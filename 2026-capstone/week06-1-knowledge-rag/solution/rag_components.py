"""本周作业的可复用组件：显式偏好记忆、安全读文件和混合检索。

本模块导入时仅使用标准库。BGE 在第一次调用 KnowledgeBase.search 时
才从本地加载，索引只保存在进程内存中，不下载模型、不写向量缓存。
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
import tempfile
from typing import Hashable, Iterable, Sequence


MAX_MEMORY_VALUE = 300
MAX_FILE_CHARS = 4000
MAX_QUERY_CHARS = 1000
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
_MEMORY_KEY = re.compile(r"\w[\w.-]{0,63}", re.UNICODE)
_LEXICAL_TOKEN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+|[A-Za-z_][A-Za-z0-9_]*|[0-9]+")


def _memory_key(key: str) -> str:
    if not isinstance(key, str) or not _MEMORY_KEY.fullmatch(key):
        raise ValueError("记忆键须为 1—64 个字母、数字、中文、下划线、点或短横线，首字不能是点或短横线")
    return key


def _memory_value(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > MAX_MEMORY_VALUE:
        raise ValueError("记忆值须为 1—300 字的简短偏好")
    return value.strip()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"记忆文件含重复键：{key}")
        result[key] = value
    return result


class MemoryStore:
    """JSON 字典记忆，仅由用户显式调用 set/delete 修改。

    load、set、delete 均返回当前 ``dict[str, str]``。set 覆盖同名键；
    delete 不存在的键抛 KeyError。每次修改重新读取文件，再用 os.replace
    原子替换，避免中途退出留下半个 JSON；不承诺多进程并发写入合并。
    """

    def __init__(self, path):
        self.path = Path(path)

    def load(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        try:
            notes = json.loads(self.path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
        except json.JSONDecodeError as exc:
            raise ValueError(f"记忆文件不是合法 JSON：{self.path}") from exc
        if not isinstance(notes, dict):
            raise ValueError("记忆文件应为键和值均为字符串的 JSON 字典")
        for key, value in notes.items():
            _memory_key(key)
            normalized = _memory_value(value)
            if value != normalized:
                raise ValueError("记忆文件中的值不能有首尾空白，请通过 set 保存偏好")
        return notes

    def _write(self, notes: dict[str, str]) -> dict[str, str]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", dir=self.path.parent,
                prefix=f".{self.path.name}.", suffix=".tmp", delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(notes, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return dict(notes)

    def set(self, key: str, value: str) -> dict[str, str]:
        key, value = _memory_key(key), _memory_value(value)
        notes = self.load()
        notes[key] = value
        return self._write(notes)

    def delete(self, key: str) -> dict[str, str]:
        key = _memory_key(key)
        notes = self.load()
        if key not in notes:
            raise KeyError(f"记忆键不存在：{key}")
        del notes[key]
        return self._write(notes)


def read_file(folder, path) -> dict:
    """只读取 workspace 内的相对 UTF-8 文件，最多返回 4000 个字符。

    同时识别 Windows 盘符、UNC/根路径和两种路径分隔符；resolve 后再次
    检查边界，阻止 ``..`` 和符号链接访问 workspace 外的内容。
    """
    if not isinstance(path, (str, os.PathLike)):
        raise ValueError("文件路径必须是相对路径字符串")
    raw = os.fspath(path)
    if not isinstance(raw, str) or not raw.strip() or "\x00" in raw:
        raise ValueError("文件路径不能为空或包含空字符")
    windows_path = PureWindowsPath(raw)
    relative = Path(raw.replace("\\", "/"))
    if relative.is_absolute() or windows_path.drive or windows_path.root or ".." in relative.parts:
        raise ValueError("只能读取 workspace 内的相对路径")
    root = Path(folder).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"workspace 目录不存在：{root}")
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError("只能读取 workspace 内的相对路径")
    with target.open(encoding="utf-8") as handle:
        text = handle.read(MAX_FILE_CHARS + 1)
    return {"text": text[:MAX_FILE_CHARS], "truncated": len(text) > MAX_FILE_CHARS}


def _chunk_parameters(size, overlap) -> None:
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("chunk_size 必须是正整数")
    if isinstance(overlap, bool) or not isinstance(overlap, int) or not 0 <= overlap < size:
        raise ValueError("overlap 必须是满足 0 <= overlap < chunk_size 的整数")


def read_chunks(folder, size: int = 240, overlap: int = 40) -> list[dict]:
    """按字符窗口切分 Markdown/TXT，返回稳定 ID、来源和片段 SHA-256。"""
    _chunk_parameters(size, overlap)
    root = Path(folder).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"知识目录不存在：{root}")
    chunks = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        if not path.resolve().is_relative_to(root):
            continue
        text = path.read_text(encoding="utf-8").strip()
        source = path.relative_to(root).as_posix()
        for number, start in enumerate(range(0, len(text), size - overlap), 1):
            chunk_text = text[start:start + size]
            chunks.append({
                "id": f"{source}#{number}", "source": source, "text": chunk_text,
                "content_sha256": hashlib.sha256(chunk_text.encode("utf-8")).hexdigest(),
            })
            if start + size >= len(text):
                break
    if not chunks:
        raise ValueError("知识库没有非空的 Markdown/TXT 文件")
    return chunks


def lexical_tokens(text: str) -> list[str]:
    """中文连续字符取 bigram，单字保留；英文标识符忽略大小写。"""
    if not isinstance(text, str):
        raise TypeError("待分词文本必须是字符串")
    tokens = []
    for match in _LEXICAL_TOKEN.finditer(text):
        word = match.group()
        if "\u3400" <= word[0] <= "\u9fff":
            tokens.extend(word[i:i + 2] for i in range(max(1, len(word) - 1)))
        else:
            tokens.append(word.casefold())
    return tokens


def bm25_scores(query: str, documents: Sequence[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """返回与 documents 顺序一致的 BM25 分数，查询重复词只计一次。"""
    if not isinstance(k1, (int, float)) or isinstance(k1, bool) or not math.isfinite(k1) or k1 <= 0:
        raise ValueError("BM25 k1 必须是有限正数")
    if not isinstance(b, (int, float)) or isinstance(b, bool) or not math.isfinite(b) or not 0 <= b <= 1:
        raise ValueError("BM25 b 必须在 0 和 1 之间")
    terms = set(lexical_tokens(query))
    tokenized = [lexical_tokens(document) for document in documents]
    count = len(tokenized)
    if count == 0:
        return []
    average_length = sum(map(len, tokenized)) / count
    if not terms or average_length == 0:
        return [0.0] * count
    document_frequency = Counter(term for tokens in tokenized for term in set(tokens))
    scores = []
    for tokens in tokenized:
        frequencies = Counter(tokens)
        score = 0.0
        for term in sorted(terms):
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            df = document_frequency[term]
            inverse_frequency = math.log(1 + (count - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (1 - b + b * len(tokens) / average_length)
            score += inverse_frequency * frequency * (k1 + 1) / denominator
        scores.append(score)
    return scores


def reciprocal_rank_fusion(rankings: Iterable[Iterable[Hashable]], k: int = 60) -> dict[Hashable, float]:
    """融合有序文档 ID/索引列表：每次出现贡献 1 / (k + 从 1 开始的排名)。

    某一路没有匹配的文档应省略，不会因零分而得到额外贡献。最终平分时的
    ID 排序由调用者处理。单路列表不能含重复 ID。
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("RRF k 必须是非负整数")
    scores = {}
    for ranking in rankings:
        seen = set()
        for rank, document in enumerate(ranking, 1):
            if document in seen:
                raise ValueError("同一排名列表不能包含重复文档")
            seen.add(document)
            scores[document] = scores.get(document, 0.0) + 1.0 / (k + rank)
    return scores


class KnowledgeBase:
    """CPU BGE 检索，mode 为 vector 或 hybrid（BM25 + 向量 RRF）。

    search 返回 top_k 个含 id/text/source/content_sha256/score/vector_score/
    lexical_score/rank 的字典。score 只用于排序，不代表事实正确概率；
    hybrid 的 RRF 分数与 vector 的余弦分数不能直接比较。
    """

    def __init__(self, folder, model_path, mode="hybrid", top_k=3, chunk_size=240, overlap=40):
        if mode not in {"vector", "hybrid"}:
            raise ValueError("mode 必须是 vector 或 hybrid")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k 必须是正整数")
        _chunk_parameters(chunk_size, overlap)
        self.folder, self.model_path = Path(folder), os.fspath(model_path)
        self.mode, self.top_k = mode, top_k
        self.chunk_size, self.overlap = chunk_size, overlap
        self.model = self.tokenizer = self.vectors = None
        self.chunks = []

    def _ensure_index(self) -> None:
        if self.vectors is not None:
            return
        self.chunks = read_chunks(self.folder, self.chunk_size, self.overlap)
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(1)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        # 当前 Windows / torch 2.4 CPU 上 BERT 的 SDPA 会返回 NaN；eager
        # 注意力配合单线程在相同权重、输入下保持有限且可重复；本机多线程
        # eager 虽然有限，重复编码却不稳定。保持参考 CLS pooling 不变。
        self.model = AutoModel.from_pretrained(
            self.model_path, local_files_only=True, attn_implementation="eager", torch_dtype=torch.float32,
        ).to("cpu").eval()
        self.vectors = self.encode([chunk["text"] for chunk in self.chunks])

    def encode(self, texts: Sequence[str], query: bool = False):
        """与参考程序一致：查询加 BGE 指令，CLS pooling 后 L2 归一化。"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("BGE 尚未加载，请先调用 search")
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("编码文本须为非空字符串列表")
        import torch

        # PyTorch 线程设置是进程全局的；LLM 初始化或调用方可能更改它。
        # 在实际编码前确保本机验证过的配置，不能只在创建索引时设置。
        if torch.get_num_threads() != 1:
            torch.set_num_threads(1)
        prefix = BGE_QUERY_PREFIX if query else ""
        vectors = []
        for start in range(0, len(texts), 8):
            batch = self.tokenizer(
                [prefix + text for text in texts[start:start + 8]],
                padding=True, truncation=False, return_tensors="pt",
            )
            if batch["input_ids"].shape[1] > 512:
                raise ValueError("文本超过 BGE 512 token 上限，请缩短查询或减小 chunk_size")
            with torch.inference_mode():
                cls = self.model(**batch).last_hidden_state[:, 0]
            vectors.append(torch.nn.functional.normalize(cls, p=2, dim=1))
        return torch.cat(vectors)

    def _vector_scores(self, query: str) -> list[float]:
        return (self.vectors @ self.encode([query], query=True)[0]).tolist()

    def search(self, query: str) -> list[dict]:
        if not isinstance(query, str) or not query.strip() or len(query.strip()) > MAX_QUERY_CHARS:
            raise ValueError("检索查询须为 1—1000 字的字符串")
        query = query.strip()
        self._ensure_index()
        vector_scores = [float(score) for score in self._vector_scores(query)]
        if len(vector_scores) != len(self.chunks) or any(not math.isfinite(score) for score in vector_scores):
            raise ValueError("向量检索返回了无效分数")
        lexical_scores = bm25_scores(query, [chunk["text"] for chunk in self.chunks])
        vector_order = sorted(range(len(self.chunks)), key=lambda i: (-vector_scores[i], self.chunks[i]["id"]))
        lexical_order = sorted(
            (i for i, score in enumerate(lexical_scores) if score > 0),
            key=lambda i: (-lexical_scores[i], self.chunks[i]["id"]),
        )
        vector_ranks = {i: rank for rank, i in enumerate(vector_order, 1)}
        lexical_ranks = {i: rank for rank, i in enumerate(lexical_order, 1)}
        scores = (reciprocal_rank_fusion([vector_order, lexical_order], k=60)
                  if self.mode == "hybrid" else dict(enumerate(vector_scores)))
        order = sorted(scores, key=lambda i: (-scores[i], self.chunks[i]["id"]))[:self.top_k]
        return [{
            **self.chunks[i], "rank": rank, "score": scores[i],
            "vector_score": vector_scores[i], "lexical_score": lexical_scores[i],
            "vector_rank": vector_ranks[i], "lexical_rank": lexical_ranks.get(i),
        } for rank, i in enumerate(order, 1)]
