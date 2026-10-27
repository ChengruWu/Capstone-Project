"""不加载模型，核对真实运行记录的完整性与来源；不评分答案语义。"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from agent_v1 import ROOT, save_result, timestamp, validate_citations
from rag_components import read_chunks
from run_experiments import AGENT_CASES, RETRIEVAL_CASES


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def close_number(left, right, tolerance=1e-12):
    return finite_number(left) and finite_number(right) and abs(left - right) <= tolerance


def verify(folder):
    folder = Path(folder)
    summary, environment = read_json(folder / "summary.json"), read_json(folder / "environment.json")
    memory = read_json(folder / "memory_demo.json")["reloaded"]
    checks = []

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})

    check("source manifest complete", set(environment["source_sha256"])
          == {path.name for path in ROOT.glob("*.py")})
    check("knowledge manifest complete", set(environment["knowledge_sha256"])
          == {path.name for path in (ROOT / "knowledge").glob("*.md")})
    for filename, expected in environment["source_sha256"].items():
        path = ROOT / filename
        check("source hash: " + filename, path.is_file() and sha256(path) == expected)
    for filename, expected in environment["knowledge_sha256"].items():
        path = ROOT / "knowledge" / filename
        check("knowledge hash: " + filename, path.is_file() and sha256(path) == expected)

    # 重建的是字符切分，不需要加载 BGE；不只比较结果中的两份片段副本。
    original_chunks = {chunk["id"]: chunk for chunk in read_chunks(ROOT / "knowledge")}

    def original_chunk_matches(chunk):
        if not isinstance(chunk, dict) or chunk.get("id") not in original_chunks:
            return False
        original = original_chunks[chunk["id"]]
        return all(chunk.get(key) == original[key]
                   for key in ("id", "source", "text", "content_sha256"))

    memory_path = folder / "memory.json"
    check("memory snapshot matches persisted file", memory_path.is_file() and read_json(memory_path) == memory)
    expected_agent_ids = [case["id"] for case in AGENT_CASES]
    check("agent case IDs complete and unique",
          Counter(item["id"] for item in summary["cases"]) == Counter(expected_agent_ids))
    outcomes = []
    for expected_case in AGENT_CASES:
        case_id = expected_case["id"]
        items = [item for item in summary["cases"] if item["id"] == case_id]
        if len(items) != 1:
            continue
        item = items[0]
        path = folder / (case_id + ".json")
        check(case_id + ": result file exists", path.is_file())
        if not path.is_file():
            continue
        result = read_json(path)
        check(case_id + ": file hash", sha256(path) == item["sha256"])
        check(case_id + ": fixed task and configuration",
              result.get("case_id") == case_id and result["task"] == expected_case["task"]
              and result["retrieval"]["mode"] == expected_case["mode"]
              and result["retrieval"]["top_k"] == 3
              and result["citation_policy"] == expected_case["policy"])
        check(case_id + ": trace step sequence",
              [s["step"] for s in result["trace"]] == list(range(1, len(result["trace"]) + 1)))
        check(case_id + ": raw matches messages",
              [s["raw"] for s in result["trace"] if "raw" in s]
              == [m["content"] for m in result["messages"] if m["role"] == "assistant"])
        evidence = {}
        successful_chunks = []
        expected_candidates = []
        expected_tool_messages = []
        for step in result["trace"]:
            for observation in step.get("observations", []):
                expected_tool_messages.append({key: value for key, value in observation.items() if key != "call"})
                if observation["call"]["name"] == "search_knowledge" and observation["status"] == "ok":
                    successful_chunks.extend(observation["result"])
                    evidence.update({c["id"]: c for c in observation["result"]})
            if "citation_validation" in step:
                check(case_id + f": step {step['step']} citation validation replay",
                      validate_citations(step["raw"], evidence) == step["citation_validation"])
                expected_candidates.append({"step": step["step"], "answer": step["raw"],
                                            "validation": step["citation_validation"]})
        check(case_id + ": evidence from successful search only", evidence == result["evidence"])
        check(case_id + ": retrieved chunks match original knowledge",
              all(original_chunk_matches(chunk) for chunk in successful_chunks)
              and all(original_chunk_matches(chunk) for chunk in result["evidence"].values()))
        check(case_id + ": tool observations match messages", expected_tool_messages
              == [json.loads(m["content"]) for m in result["messages"] if m["role"] == "tool"])
        check(case_id + ": answer candidates preserve raw and validation",
              result["answer_candidates"] == expected_candidates)
        check(case_id + ": persisted memory loaded", result["memory_used"] == memory)
        expected_summary = {"id": case_id, "status": result["status"], "steps": len(result["trace"]),
                            "elapsed_s": result["elapsed_s"], "citation_repairs": result["citation_repairs"],
                            "citations": result.get("citation_validation", {}).get("citations", []),
                            "tool_calls": [obs["call"]["name"] for step in result["trace"]
                                           for obs in step.get("observations", [])]}
        for field, expected in expected_summary.items():
            check(case_id + ": summary " + field, item.get(field) == expected)
        repairs = result["citation_repairs"]
        check(case_id + ": at most one citation repair",
              type(repairs) is int and 0 <= repairs <= 1)
        check(case_id + ": repair count matches feedback",
              repairs == sum(m["role"] == "user" and m["content"].startswith("来源检查未通过：")
                             for m in result["messages"]))
        check(case_id + ": generation metadata present",
              all({"input_tokens", "output_tokens", "hit_token_limit"} <= set(s.get("generation", {}))
                  for s in result["trace"] if "raw" in s))
        check(case_id + ": generation not truncated",
              not any(s.get("generation", {}).get("hit_token_limit", False) for s in result["trace"]))
        if result.get("citation_validation"):
            check(case_id + ": replay citation validation",
                  validate_citations(result["answer"], evidence) == result["citation_validation"])
        if result["status"] == "ok":
            check(case_id + ": ok requires citations", bool(result.get("answer"))
                  and result.get("citation_validation", {}).get("passed") is True)
        if result["status"] in {"model_error", "max_steps"}:
            check(case_id + ": unfinished repair has no final answer", result["answer"] is None)
        outcomes.append({"id": case_id, "status": result["status"],
                         "citation_repairs": result["citation_repairs"]})

    retrieval = read_json(folder / "retrieval_comparison.json")
    expected_retrieval_ids = [case["id"] for case in RETRIEVAL_CASES]
    check("retrieval top_k is three", retrieval["top_k"] == 3)
    check("retrieval modes are vector and hybrid",
          {case["mode"] for case in retrieval["cases"]} == {"vector", "hybrid"})
    for mode in ("vector", "hybrid"):
        cases = [c for c in retrieval["cases"] if c["mode"] == mode]
        reported = retrieval["summary"][mode]
        check(mode + ": retrieval case IDs complete and unique",
              Counter(c["id"] for c in cases) == Counter(expected_retrieval_ids))
        check(mode + ": metric sample count", len(cases) == reported["n"] == len(RETRIEVAL_CASES))
        recomputed = []
        for expected_case in RETRIEVAL_CASES:
            matches = [case for case in cases if case["id"] == expected_case["id"]]
            if len(matches) != 1:
                continue
            case = matches[0]
            label = mode + "/" + case["id"]
            check(label + ": fixed query and relevance labels", case["query"] == expected_case["query"]
                  and case["expected_sources"] == expected_case["expected_sources"])
            hits = case["hits"]
            check(label + ": top_k distinct hits", len(hits) == min(3, len(original_chunks))
                  and len({hit["id"] for hit in hits}) == len(hits))
            check(label + ": rank sequence", [hit["rank"] for hit in hits] == list(range(1, len(hits) + 1)))
            check(label + ": hit chunks match original knowledge", all(original_chunk_matches(hit) for hit in hits))
            check(label + ": finite retrieval scores", all(finite_number(hit.get(field))
                  for hit in hits for field in ("score", "vector_score", "lexical_score")))
            check(label + ": scores in descending order",
                  all(hits[i]["score"] >= hits[i + 1]["score"] for i in range(len(hits) - 1)))
            expected_sources = set(expected_case["expected_sources"])
            sources = [hit["source"] for hit in hits]
            hit_at_1 = bool(sources and sources[0] in expected_sources)
            recall = len(set(sources) & expected_sources) / len(expected_sources)
            check(label + ": hit@1 recomputed from hits", case["hit_at_1"] is hit_at_1)
            check(label + ": recall@3 recomputed from hits", close_number(case["source_recall_at_3"], recall))
            recomputed.append((hit_at_1, recall))
        complete = len(recomputed) == len(RETRIEVAL_CASES)
        check(mode + ": hit@1 aggregation", complete and close_number(
              reported["hit_at_1"], sum(c[0] for c in recomputed) / len(recomputed)))
        check(mode + ": recall@3 aggregation", reported["mean_source_recall_at_3"]
              is not None and complete and close_number(reported["mean_source_recall_at_3"],
              sum(c[1] for c in recomputed) / len(recomputed)))
    stability = []
    for query_id in expected_retrieval_ids:
        pair = [c for c in retrieval["cases"] if c["id"] == query_id]
        paired = len(pair) == 2 and {c["mode"] for c in pair} == {"vector", "hybrid"}
        check(query_id + ": exactly one record per retrieval mode", paired)
        if not paired:
            continue
        check(query_id + ": paired query and labels match", pair[0]["query"] == pair[1]["query"]
              and pair[0]["expected_sources"] == pair[1]["expected_sources"])
        maps = [{h["id"]: h["vector_score"] for h in c["hits"]} for c in pair]
        overlap = set(maps[0]) & set(maps[1])
        check(query_id + ": common recalled fragments nonempty", bool(overlap))
        stable = bool(overlap) and all(close_number(maps[0][i], maps[1][i], 1e-6) for i in overlap)
        check(query_id + ": common recalled fragment vector scores stable", stable)
        stability.append({"id": query_id, "common_fragment_ids": sorted(overlap), "stable": stable})
    return {"created_at": timestamp(), "passed": all(c["passed"] for c in checks),
            "check_count": len(checks), "checks": checks, "outcomes": outcomes, "stability": stability,
            "scope": "固定案例完整性、原始知识片段、来源编号、summary字段、逐题及总体指标、共同召回片段向量分数；不代表答案语义或任务成功率"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.results_dir)
    output = args.output or args.results_dir / "verification.json"
    save_result(result, output)
    print(json.dumps({"passed": result["passed"], "checks": result["check_count"], "output": str(output)}, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
