"""一次加载本地模型，保存固定任务、检索对照、记忆操作与完整原始轨迹。"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from agent_v1 import (ROOT, MODEL_HOME, EXAMPLE, make_hf_llm, make_tools,
                      run_agent, save_result, timestamp)
from rag_components import KnowledgeBase, MemoryStore

RETRIEVAL_CASES = [
    {"id": "port", "query": "Address already in use，APP_PORT=8001 应该怎样配置并重新启动？",
     "expected_sources": ["configuration.md", "troubleshooting.md"]},
    {"id": "startup", "query": "首次使用安装依赖、准备配置文件和启动命令是什么？",
     "expected_sources": ["startup.md"]},
    {"id": "logs", "query": "运行日志路径是什么？日志默认保留多少天？",
     "expected_sources": ["logging.md"]},
    {"id": "export", "query": "导出结果保存在哪里？再次导出会覆盖吗？",
     "expected_sources": ["export.md"]},
    {"id": "sample", "query": "data/sample.csv 示例数据包含哪些列，更换数据要保留什么？",
     "expected_sources": ["sample_data.md"]},
    {"id": "config", "query": "修改 .env 中的 APP_PORT 后运行中的服务会自动生效吗？",
     "expected_sources": ["configuration.md"]},
    {"id": "backup", "query": "exports/result.csv 再次导出前如何保留旧结果？",
     "expected_sources": ["export.md"]},
    {"id": "privacy", "query": "分享 logs/app.log 前应该去除什么信息？",
     "expected_sources": ["logging.md"]},
]

AGENT_CASES = [
    {"id": "01_vector_port", "mode": "vector", "policy": "repair", "task": EXAMPLE},
    {"id": "02_hybrid_port", "mode": "hybrid", "policy": "repair", "task": EXAMPLE},
    {"id": "03_hybrid_onboarding", "mode": "hybrid", "policy": "repair",
     "task": "请调用 read_file 读取相对路径 onboarding.txt，按其中的三个问题检索资料并逐项回答，标明来源。"},
    {"id": "04_hybrid_export", "mode": "hybrid", "policy": "repair",
     "task": "请读取相对路径 export_task.txt，查询知识库后给出有来源的回答。"},
    {"id": "05_hybrid_unknown", "mode": "hybrid", "policy": "repair",
     "task": "请读取相对路径 unknown_task.txt，查询知识库后回答；资料没有说明的地方直接说不知道。"},
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def retrieval_eval(kb):
    cases = []
    for mode in ("vector", "hybrid"):
        kb.mode = mode
        for case in RETRIEVAL_CASES:
            hits = kb.search(case["query"])
            expected = set(case["expected_sources"])
            sources = [hit["source"] for hit in hits]
            cases.append({**case, "mode": mode, "hits": hits,
                          "hit_at_1": bool(sources and sources[0] in expected),
                          "source_recall_at_3": len(set(sources) & expected) / len(expected)})
    summary = {}
    for mode in ("vector", "hybrid"):
        subset = [c for c in cases if c["mode"] == mode]
        summary[mode] = {"n": len(subset),
                         "hit_at_1": sum(c["hit_at_1"] for c in subset) / len(subset),
                         "mean_source_recall_at_3": sum(c["source_recall_at_3"] for c in subset) / len(subset)}
    return {"created_at": timestamp(), "top_k": 3, "cases": cases, "summary": summary,
            "scope": "人工定义的八个小样本；按来源文件统计，不验证片段语义或生成答案，不能代表通用性能"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--model", default=str(MODEL_HOME / "Qwen2.5-7B-Instruct"))
    parser.add_argument("--embedding-model", default=str(MODEL_HOME / "bge-small-zh-v1.5"))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--quantization", choices=["nf4", "none"], default="nf4")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("实验目录已存在；请指定新的 --output-dir，保留历史原文")
    args.output_dir.mkdir(parents=True)
    import torch
    environment = {"created_at": timestamp(), "python": sys.version, "executable": sys.executable,
                   "platform": platform.platform(), "cwd": str(Path.cwd()), "argv": sys.argv,
                   "versions": {p: importlib.metadata.version(p) for p in
                                ["torch", "transformers", "accelerate", "bitsandbytes", "huggingface-hub"]},
                   "cuda_available": torch.cuda.is_available(),
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   "source_sha256": {p.name: digest(p) for p in ROOT.glob("*.py")},
                   "knowledge_sha256": {p.name: digest(p) for p in (ROOT / "knowledge").glob("*.md")}}
    gpu_command = shutil.which("nvidia-smi")
    environment["nvidia_smi"] = (subprocess.run([gpu_command], capture_output=True, text=True,
                                               check=False).stdout if gpu_command else "未安装 nvidia-smi")
    save_result(environment, args.output_dir / "environment.json")

    # 每次使用单独记忆文件，避免改动用户日常偏好；每步用新实例重新读取。
    memory_path = args.output_dir / "memory.json"
    operations = []
    for key, value in [("style", "请用中文写详细说明。"),
                       ("style", "请用中文给出简短的编号步骤，保留必要命令和来源；资料没说明就直接说明未知。"),
                       ("temporary", "这是一条待删除的临时偏好。")]:
        operations.append({"action": "set", "key": key, "value": value,
                           "snapshot": MemoryStore(memory_path).set(key, value)})
    operations.append({"action": "delete", "key": "temporary",
                       "snapshot": MemoryStore(memory_path).delete("temporary")})
    notes = MemoryStore(memory_path).load()
    save_result({"operations": operations, "reloaded": notes}, args.output_dir / "memory_demo.json")

    kb = KnowledgeBase(ROOT / "knowledge", args.embedding_model, mode="hybrid", top_k=3)
    retrieval = retrieval_eval(kb)
    save_result(retrieval, args.output_dir / "retrieval_comparison.json")
    print("[retrieval] " + json.dumps(retrieval["summary"], ensure_ascii=False), flush=True)
    llm = make_hf_llm(args.model, args.device, 768, args.quantization, 8192)
    summary = []
    for case in AGENT_CASES:
        print(f"\n[experiment] {case['id']}", flush=True)
        kb.mode = case["mode"]
        result = run_agent(case["task"], llm, make_tools(kb, ROOT / "workspace"),
                           MemoryStore(memory_path).load(), max_steps=8, citation_policy=case["policy"])
        result.update(created_at=timestamp(), model=llm.metadata, case_id=case["id"],
                      retrieval={"mode": case["mode"], "top_k": 3,
                                 "embedding_model": args.embedding_model, "chunk_size": 240, "overlap": 40,
                                 "attention": "eager", "cpu_threads": 1, "dtype": "float32"})
        target = args.output_dir / (case["id"] + ".json")
        save_result(result, target)
        item = {"id": case["id"], "status": result["status"], "steps": len(result["trace"]),
                "elapsed_s": result["elapsed_s"], "citation_repairs": result["citation_repairs"],
                "citations": result.get("citation_validation", {}).get("citations", []),
                "tool_calls": [obs["call"]["name"] for step in result["trace"]
                               for obs in step.get("observations", [])], "sha256": digest(target)}
        summary.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
        print(result["answer"], flush=True)
    save_result({"created_at": timestamp(), "cases": summary}, args.output_dir / "summary.json")
    print(f"\n完整记录已保存：{args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
