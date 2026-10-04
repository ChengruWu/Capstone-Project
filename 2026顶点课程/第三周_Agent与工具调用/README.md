# 第三周：Agent 与 Tool Calling

## 提交材料

- [提交说明](提交说明.md)：1.5B 遇到的问题、换用 7B 的原因，以及两种模型的结果与分析。
- [实验命令](实验命令.md)：全部 21 次实际执行命令及批量复现方式。
- [作业要求](作业.md)；[课程课件](Minimal%20Agent与Tool%20Calling.pptx)。

| 内容 | 位置 |
| --- | --- |
| 1.5B 的 10 组原始结果与运行日志 | [results/](results/)、[logs/](logs/) |
| 7B 的 11 组原始结果与运行日志 | [results_7b/](results_7b/)、[logs_7b/](logs_7b/) |
| Agent 和工具实现 | agent_v0.py |
| 批量复现实验 | run_experiments.py |
| 核验结果、原文和参数对照 | verify_results.py |
| JSON 停止条件测试 | test_json_stop.py |
| 依赖与固定输入 | requirements*.txt、workspace/test_report.txt |

结果与原始命令保持对应路径。统一报告已包含两种模型的分析，不再保留重复阶段报告、独立下载脚本或 audit 副本。

## 在现有仓库中使用

在本第三周目录运行。已有模型和第一周环境可以直接复用，无需 Ollama：

```powershell
$week3Python = '../第一周_LLM推理/.venv/Scripts/python.exe'
$week3Model = '../第一周_LLM推理/models/Qwen2.5-7B-Instruct'
```

## 只解压本提交包时准备环境

以下是本机已验证的 Python 3.12 / CUDA 12.1 PyTorch 组合，适用于相应的 NVIDIA GPU 环境。模型和环境不包含在 ZIP 中。

```powershell
python -m venv .venv
$week3Python = './.venv/Scripts/python.exe'
& $week3Python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
& $week3Python -m pip install -r requirements-4bit.txt

# 官方原始模型约 15.24GB，加载时进行 NF4 量化。
& $week3Python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='models/Qwen2.5-7B-Instruct')"
$week3Model = './models/Qwen2.5-7B-Instruct'
```

复现 1.5B 阶段时，以相同方法下载 `Qwen/Qwen2.5-1.5B-Instruct`，运行时选择 `--quantization none` 和 `--max-new-tokens 256`；完整参数见实验命令。历史命令使用当时的本机绝对路径，换机器时应替换为当前环境和模型路径。

## 推荐的单次运行

本机 RTX 4060 Laptop 8GB 使用 7B NF4、BF16 计算、全部模型层放 GPU；总 token 预算 4096，每步输出上限 512。采样参数已实际传入模型，但单次输出不保证随参数变化而变化，详见提交说明。

```powershell
& $week3Python -X utf8 agent_v0.py --backend hf --model $week3Model --device cuda --quantization nf4 --sample --temperature 0.3 --top-p 0.8 --top-k 20 --seed 42 --json-stop string-aware --max-context-tokens 4096 --max-new-tokens 512 --task '读取 test_report.txt，计算测试通过率' --output results_7b/manual_run.json
```

单次命令会写入指定输出文件，再次运行请使用新文件名。默认结果与报告中的原始记录保持原样。

## 批量复现和核验

批量脚本顺序运行十组对照；另一次推荐配置验证的命令见实验命令。复现时使用新的结果/日志目录，脚本拒绝覆盖已有批量结果。

```powershell
& $week3Python -X utf8 run_experiments.py --model $week3Model --quantization nf4 --max-new-tokens 512 --results-dir results_7b_rerun --logs-dir logs_7b_rerun
```

无需加载模型即可核验提交材料和运行自测：

```powershell
& $week3Python -B -X utf8 agent_v0.py --selftest
& $week3Python -B -X utf8 -m unittest test_json_stop -v
& $week3Python -B -X utf8 verify_results.py --results-dir results --logs-dir logs --expected-quantization none
& $week3Python -B -X utf8 verify_results.py --results-dir results_7b --logs-dir logs_7b --expected-quantization nf4
```

最新结果核验摘要分别为 `logs/result_validation.json` 和 `logs_7b/result_validation.json`；自测和单元测试日志集中保留在 `logs_7b/`。核验通过表示记录完整、对照正确，不代表模型的每条回答都正确。

## 提交包范围

第三周提交包包含本目录的报告、命令、四个 Python 脚本、依赖文件、固定输入、两套结果和必要日志，以及课程课件与作业要求。不包含其他周材料、模型、虚拟环境、缓存或嵌套压缩包。
