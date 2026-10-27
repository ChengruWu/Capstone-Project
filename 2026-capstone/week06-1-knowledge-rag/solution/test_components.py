"""不依赖 torch、Transformers 或模型的组件测试。

运行：python -m unittest discover -s solution -p "test_components.py" -v
"""

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rag_components import (
    KnowledgeBase, MemoryStore, bm25_scores, lexical_tokens, read_chunks,
    read_file, reciprocal_rank_fusion,
)


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "data" / "memory.json"
        self.store = MemoryStore(self.path)

    def test_overwrite_delete_and_reload(self):
        self.assertEqual(self.store.load(), {})
        self.assertEqual(self.store.set("language", " 中文 "), {"language": "中文"})
        self.store.set("style", "简洁")
        self.store.set("language", "英文")
        self.assertEqual(MemoryStore(self.path).load(), {"language": "英文", "style": "简洁"})
        self.assertEqual(self.store.delete("language"), {"style": "简洁"})
        self.assertEqual(MemoryStore(self.path).load(), {"style": "简洁"})
        self.assertEqual(self.store.delete("style"), {})
        with self.assertRaises(KeyError):
            self.store.delete("missing")

    def test_reject_invalid_keys_and_values_without_writing(self):
        for key in ["", "bad key", "../bad", ".hidden", "-bad", "a" * 65, 2, None]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.store.set(key, "中文")
        for value in ["", "  ", "字" * 301, 2, None, []]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.store.set("language", value)
        self.assertFalse(self.path.exists())
        self.store.set("回答语言", "字" * 300)
        self.assertEqual(len(self.store.load()["回答语言"]), 300)

    def test_reject_bad_format(self):
        self.path.parent.mkdir()
        bad_contents = ["{bad json", "[]", '"text"', '{"language": 1}',
                        '{"language": "中文", "language": "英文"}',
                        '{"bad key": "中文"}', '{"language": " 中文 "}']
        for content in bad_contents:
            self.path.write_text(content, encoding="utf-8")
            with self.subTest(content=content), self.assertRaises(ValueError):
                self.store.load()

    def test_failed_atomic_replace_preserves_previous_file(self):
        self.store.set("language", "中文")
        before = self.path.read_bytes()
        with patch("rag_components.os.replace", side_effect=OSError("模拟替换失败")):
            with self.assertRaises(OSError):
                self.store.set("language", "英文")
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])


class FileAndChunkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "workspace"
        self.root.mkdir()

    def test_read_limit_and_unicode(self):
        file = self.root / "task.txt"
        file.write_text("字" * 4000, encoding="utf-8")
        self.assertEqual(read_file(self.root, "task.txt"), {"text": "字" * 4000, "truncated": False})
        file.write_text("字" * 4001, encoding="utf-8")
        result = read_file(self.root, "task.txt")
        self.assertEqual(len(result["text"]), 4000)
        self.assertTrue(result["truncated"])

    def test_reject_path_escape_and_windows_paths(self):
        (self.base / "secret.txt").write_text("secret", encoding="utf-8")
        for path in ["../secret.txt", "..\\secret.txt", str(self.base / "secret.txt"),
                     "C:\\Windows\\system.ini", "C:relative.txt", "\\\\server\\share\\a.txt",
                     "\\rooted.txt", "", "\x00bad"]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                read_file(self.root, path)

    def test_symlink_boundary(self):
        outside = self.base / "secret.txt"
        outside.write_text("secret", encoding="utf-8")
        link = self.root / "link.txt"
        try:
            os.symlink(outside, link)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"当前系统不能创建符号链接：{exc}")
        with self.assertRaises(ValueError):
            read_file(self.root, "link.txt")
        (self.root / "safe.md").write_text("安全资料", encoding="utf-8")
        self.assertEqual([chunk["source"] for chunk in read_chunks(self.root)], ["safe.md"])

    def test_chunks_have_overlap_ids_sources_and_hashes(self):
        (self.root / "nested").mkdir()
        (self.root / "nested" / "a.MD").write_text("  0123456789  ", encoding="utf-8")
        (self.root / "ignored.py").write_text("ignore", encoding="utf-8")
        chunks = read_chunks(self.root, size=6, overlap=2)
        self.assertEqual([c["text"] for c in chunks], ["012345", "456789"])
        self.assertEqual([c["id"] for c in chunks], ["nested/a.MD#1", "nested/a.MD#2"])
        for chunk in chunks:
            self.assertEqual(chunk["source"], "nested/a.MD")
            self.assertEqual(chunk["content_sha256"], hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest())

    def test_exact_end_no_extra_chunk_and_short_tail(self):
        file = self.root / "a.txt"
        file.write_text("abcdefgh", encoding="utf-8")
        self.assertEqual([c["text"] for c in read_chunks(self.root, 4, 0)], ["abcd", "efgh"])
        file.write_text("abcdefghi", encoding="utf-8")
        self.assertEqual([c["text"] for c in read_chunks(self.root, 4, 0)], ["abcd", "efgh", "i"])
        file.write_text("a", encoding="utf-8")
        self.assertEqual(len(read_chunks(self.root, 4, 3)), 1)

    def test_chunk_parameter_validation_and_empty_folder(self):
        for size, overlap in [(0, 0), (-1, 0), (1.5, 0), (True, 0), (4, -1), (4, 4), (4, 5), (4, True)]:
            with self.subTest(size=size, overlap=overlap), self.assertRaises(ValueError):
                read_chunks(self.root, size, overlap)
        with self.assertRaises(ValueError):
            read_chunks(self.root)
        (self.root / "empty.md").write_text(" \n ", encoding="utf-8")
        with self.assertRaises(ValueError):
            read_chunks(self.root)
        with self.assertRaises(FileNotFoundError):
            read_chunks(self.root / "missing")


class LexicalAndFusionTests(unittest.TestCase):
    def test_chinese_bigrams_and_english_identifiers(self):
        self.assertEqual(lexical_tokens("缓存目录 CACHE_DIR FileNotFoundError 500。中"),
                         ["缓存", "存目", "目录", "cache_dir", "filenotfounderror", "500", "中"])
        self.assertEqual(lexical_tokens("，！？"), [])

    def test_bm25_matches_chinese_and_case_insensitive_error_identifier(self):
        documents = ["缓存目录 CACHE_DIR 不可写会启动失败", "检查 API_KEY 是否为空", "启动日志记录正常事件"]
        scores = bm25_scores("缓存目录 cache_dir", documents)
        self.assertGreater(scores[0], 0)
        self.assertEqual(scores[1:], [0.0, 0.0])
        self.assertGreater(bm25_scores("api_key", documents)[1], 0)
        self.assertEqual(bm25_scores("", documents), [0.0] * 3)
        self.assertEqual(bm25_scores("cache_dir", []), [])
        self.assertEqual(bm25_scores("cache_dir", [""]), [0.0])

    def test_bm25_rare_term_and_length_normalization(self):
        scores = bm25_scores("timeout", ["timeout", "timeout " + "other " * 20, "normal"])
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[1], scores[2])
        self.assertEqual(bm25_scores("timeout timeout", ["timeout"]), bm25_scores("timeout", ["timeout"]))
        with self.assertRaises(ValueError):
            bm25_scores("timeout", ["timeout"], k1=0)
        with self.assertRaises(ValueError):
            bm25_scores("timeout", ["timeout"], b=2)

    def test_rrf_contributions_are_rank_based(self):
        scores = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a"]])
        self.assertAlmostEqual(scores["a"], 1 / 61 + 1 / 62)
        self.assertAlmostEqual(scores["b"], scores["a"])
        self.assertAlmostEqual(scores["c"], 1 / 63)
        self.assertEqual(reciprocal_rank_fusion([[], []]), {})
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([["a", "a"]])
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([["a"]], k=-1)


class _FakeKnowledgeBase(KnowledgeBase):
    """注入已知向量分数，只替换昂贵模型边界，保留完整 search 排序。"""

    def __init__(self, chunks, scores, **kwargs):
        super().__init__("not-needed", "not-needed", **kwargs)
        self.chunks = chunks
        self.scores = scores

    def _ensure_index(self):
        pass

    def _vector_scores(self, query):
        return self.scores


class KnowledgeBaseTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {"id": "a.md#1", "source": "a.md", "text": "启动故障排查", "content_sha256": "a" * 64},
            {"id": "b.md#1", "source": "b.md", "text": "CACHE_DIR 缓存目录权限不足", "content_sha256": "b" * 64},
            {"id": "c.md#1", "source": "c.md", "text": "服务的启动顺序", "content_sha256": "c" * 64},
        ]

    def test_lazy_constructor_and_query_validation(self):
        kb = KnowledgeBase("not-existing", "not-existing")
        self.assertIsNone(kb.model)
        self.assertIsNone(kb.vectors)
        for query in ["", "   ", "字" * 1001, None, 1]:
            with self.subTest(query=query), self.assertRaises(ValueError):
                kb.search(query)
        for options in [{"mode": "lexical"}, {"top_k": 0}, {"top_k": True}, {"chunk_size": 0}, {"overlap": 240}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                KnowledgeBase("unused", "unused", **options)

    def test_hybrid_promotes_exact_identifier_and_reports_evidence(self):
        vector = _FakeKnowledgeBase(self.chunks, [0.9, 0.8, 0.7], mode="vector", top_k=3)
        hybrid = _FakeKnowledgeBase(self.chunks, [0.9, 0.8, 0.7], mode="hybrid", top_k=3)
        self.assertEqual(vector.search("cache_dir")[0]["id"], "a.md#1")
        results = hybrid.search("cache_dir")
        first = results[0]
        self.assertEqual(first["id"], "b.md#1")
        self.assertEqual([r["rank"] for r in results], [1, 2, 3])
        self.assertEqual(first["vector_rank"], 2)
        self.assertEqual(first["lexical_rank"], 1)
        self.assertGreater(first["lexical_score"], 0)
        self.assertEqual(first["vector_score"], 0.8)
        self.assertAlmostEqual(first["score"], 1 / 62 + 1 / 61)
        self.assertEqual(first["source"], "b.md")
        self.assertEqual(first["content_sha256"], "b" * 64)

    def test_no_lexical_hit_does_not_boost_zero_score_documents(self):
        kb = _FakeKnowledgeBase(self.chunks, [0.9, 0.8, 0.7], top_k=10)
        results = kb.search("unmatched_identifier")
        self.assertEqual(len(results), 3)
        self.assertEqual([r["id"] for r in results], ["a.md#1", "b.md#1", "c.md#1"])
        self.assertAlmostEqual(results[0]["score"], 1 / 61)
        self.assertTrue(all(result["lexical_rank"] is None for result in results))

    def test_deterministic_ties_and_invalid_vector_scores(self):
        kb = _FakeKnowledgeBase(list(reversed(self.chunks)), [0.5, 0.5, 0.5], mode="vector", top_k=2)
        self.assertEqual([result["id"] for result in kb.search("unknown")], ["a.md#1", "b.md#1"])
        kb.scores = [0.5, math.nan, 0.5]
        with self.assertRaises(ValueError):
            kb.search("unknown")
        kb.scores = [0.5]
        with self.assertRaises(ValueError):
            kb.search("unknown")


if __name__ == "__main__":
    unittest.main()
