# 第六周作业：带记忆、工具与来源检查的文档助手

本项目读取教学项目的求助文件，检索本地文档，再生成带片段编号的中文回答。它接入可更新的偏好 Memory、`read_file` / `search_knowledge` 两种 Tool，以及六份 Markdown 组成的小型知识库。在课程参考实现上增加了同名记忆覆盖与删除、BM25 + BGE 混合检索、工具错误反馈、引用编号检查与至多一次纠正，并保留每轮模型原文。

姓名、学号、实际实验结果和问题分析见 [作业报告](report.md)；固定实验索引见 [results/summary.json](results/summary.json)。本页说明代码接口和复现方法。

## 目录与处理流程

| 文件或目录 | 用途 |
| --- | --- |
| `agent_v1.py` | 工具定义、Agent Loop、本地 Qwen 推理和命令行入口 |
| `rag_components.py` | JSON 偏好记忆、受目录限制的文件读取、文档切分和检索 |
| `knowledge/` | 启动、配置、排错、日志、示例数据、导出六份知识资料 |
| `workspace/` | 端口报错、综合上手、结果导出、未知问题的固定输入 |
| `run_experiments.py` | 一次加载模型，运行检索对照、记忆演示和五次 Agent 实验 |
| `verify_results.py` | 无模型核验原文、来源、文件哈希、固定案例与指标计算 |
| `test_agent.py`、`test_components.py` | 使用构造输入和 stub 的单元测试，无需加载或下载模型 |
| `requirements.txt` | 本方案依赖版本；PyTorch 沿用原环境 |
| `results/` | 原始实验 JSON、环境信息、记忆操作和结果索引 |

Agent 启动时读取用户显式保存的偏好。模型提出工具调用后，程序校验参数并执行真实函数，将成功结果或错误交回模型；检索得到的片段累积为本次证据。模型提出最终回答时检查片段编号。默认允许一次来源纠正机会，所有调用、原文和候选回答均留存。

知识库以 240 字符窗口、40 字符重叠切分，片段记录相对来源、编号和内容 SHA-256。BGE 在 CPU 上编码，显式使用 `attn_implementation="eager"`，并固定 float32 与 CPU 单线程，避免本机默认 CPU SDPA 的 NaN，以及多线程下的重复编码不稳定；使用查询指令、CLS pooling 和 L2 归一化。`vector` 按余弦相似度排序；`hybrid` 将向量排序与中文 bigram / 英文标识符 BM25 排序通过 RRF 融合，默认返回三个片段。索引只保存在进程内存中，修改资料后重新启动即可。

**六份知识资料均为课程教学虚构内容。** 其中的 `app.py`、`.env`、`exports/result.csv` 等描述另一个示例项目，不是本助手提供的应用。助手只解释资料，不执行安装、启动或进程操作。

## 复用本仓库环境

以下为 PowerShell 命令，先在仓库根目录执行。先取得绝对路径，再切换到作业目录，避免相对路径随工作目录改变：

```powershell
$py = (Resolve-Path '2026顶点课程/第一周_LLM推理/.venv/Scripts/python.exe').Path
$modelHome = (Resolve-Path '2026顶点课程/第一周_LLM推理/models').Path
Set-Location '2026-capstone/week06-1-knowledge-rag/solution'
```

运行模型默认寻找仓库根目录下 `2026顶点课程/第一周_LLM推理/models` 中的 `Qwen2.5-7B-Instruct` 与 `bge-small-zh-v1.5`。模型读取使用 `local_files_only=True`，运行时不会自动下载。已有依赖时无需重装；缺包时再执行：

```powershell
& $py -m pip install -r requirements.txt
```

依赖文件固定 Transformers 4.51.3、Accelerate 1.6.0、BitsAndBytes 0.45.5、Hugging Face Hub 0.36.2。还需要已安装的 PyTorch；本仓库沿用第一周的 Python / PyTorch CUDA 环境，不在本作业中重装 CUDA 或 PyTorch。只有记忆操作和单元测试时，代码仅使用 Python 标准库。

本仓库可复用已有 Qwen 权重。如果缺少 BGE，可使用已经安装的 `huggingface-hub` 单独下载所需文件；以下命令只下载 Embedding 模型：

```powershell
& $py -X utf8 -c "from huggingface_hub import snapshot_download; import sys; from pathlib import Path; snapshot_download('BAAI/bge-small-zh-v1.5', local_dir=str(Path(sys.argv[1]) / 'bge-small-zh-v1.5'), allow_patterns=['config.json','pytorch_model.bin','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt'])" $modelHome
```

模型位于其他地方时，在模型任务或批量脚本后加 `--model '实际/Qwen目录' --embedding-model '实际/BGE目录'`。检索命令只需指定 `--embedding-model`。

## 保存、更新与删除偏好

Memory 是 JSON 字典，仅由显式命令修改，不自动从聊天或检索资料中提取事实。同一键再次保存会覆盖旧值；删除不存在的键会报错。每个值为 1—300 字的非空字符串，写入使用临时文件加原子替换。

```powershell
& $py -X utf8 agent_v1.py --remember 'style=请用中文写详细说明。'
& $py -X utf8 agent_v1.py --remember 'style=请用中文给出简短编号步骤，保留命令和来源。'
& $py -X utf8 agent_v1.py --remember 'temporary=这是一条临时偏好。'
& $py -X utf8 agent_v1.py --forget temporary
& $py -X utf8 agent_v1.py --show-memory
```

默认保存在 `data/preferences.json`，每次新任务重新读取。可用 `--memory '其他路径.json'` 隔离不同实验的偏好。批量实验使用其输出目录中的独立记忆文件。

## 单独检索与运行 Agent

单检索只加载 CPU 上的 BGE，不加载回答模型。下面两条命令便于比较相同问题的排序与原文：

```powershell
& $py -X utf8 agent_v1.py --search 'Address already in use，改为 APP_PORT=8001 后怎样生效？' --retrieval-mode vector --top-k 3
& $py -X utf8 agent_v1.py --search 'Address already in use，改为 APP_PORT=8001 后怎样生效？' --retrieval-mode hybrid --top-k 3
```

默认用 CUDA、7B NF4 量化、贪心生成、每步最多 768 个新 token、总上下文预算 8192、最多八轮决策。完整端口任务与自定义任务如下：

```powershell
& $py -X utf8 agent_v1.py --example --output results/manual_port.json
& $py -X utf8 agent_v1.py --task '请读取 onboarding.txt，逐项检索并回答其中的问题，标注来源；没有提供的信息直接说明未知。' --output results/manual_onboarding.json
```

`--citation-policy check` 仅检查并记录失败；默认的 `repair` 会将检查问题和真实可用 id 反馈模型，至多提供一次纠正机会。工具参数错误与协议错误可以反馈后继续尝试，总轮数仍受 `--max-steps` 限制。上下文超预算时报告错误，不会静默截断。

没有 NVIDIA GPU 时，须同时设置 `--device cpu --quantization none`。7B 在 CPU 上按 float32 加载，仅权重就需要约 28 GB 内存，推理还需额外空间，建议改用已有的较小 Qwen2.5 模型；较小模型的工具遵循表现需要单独观察：

```powershell
& $py -X utf8 agent_v1.py --example --device cpu --quantization none --model (Join-Path $modelHome 'Qwen2.5-1.5B-Instruct') --output results/manual_cpu.json
```

指定的输出 JSON 只允许新建，拒绝覆盖。再次运行请更换文件名；省略 `--output` 时自动生成时间戳文件。

## 单元测试与批量复现

单元测试不需要模型，构造的响应也不代表真实模型效果：

```powershell
& $py -B -X utf8 -m unittest discover -s . -p 'test_*.py' -v
```

固定实验包含八个问题的纯检索对照，以及端口、综合上手、导出、未知问题四类场景的五次 Agent 运行。批量脚本拒绝使用已经存在的输出目录，避免覆盖原始记录：

```powershell
& $py -X utf8 run_experiments.py --output-dir results_reproduce
& $py -X utf8 verify_results.py --results-dir results_reproduce
```

再次复现时请用新的目录名。批量脚本会记录环境版本、GPU 信息、源文件与知识资料哈希、独立记忆操作、检索结果、完整模型轨迹和摘要。端口 Agent 两组使用相同任务、记忆、生成配置和 `repair` 策略，只改变检索方式；此外独立保存八题检索指标。小样本的单次运行不能说明通用成功率。

## 如何阅读结果与引用检查的边界

| JSON 字段 | 含义 |
| --- | --- |
| `task`、`memory_used` | 本次任务及实际加载的偏好快照 |
| `trace` | 每轮原始 `raw`、生成 token 信息、工具调用及真实返回、错误、来源检查 |
| `messages` | 传给模型及由模型产生的完整对话记录 |
| `evidence` | 本次成功检索返回的片段，按真实 id 索引 |
| `answer_candidates` | 各次候选回答及其检查结果，包含纠正前失败候选 |
| `answer`、`citation_validation` | 最后提交的回答及编号检查；纠正未完成时 `answer` 可为空 |
| `citation_repairs` | 实际发出的来源纠正提示次数，最多为一 |
| `model`、`retrieval`、`created_at` | 模型配置、检索配置和带 UTC+08:00 时区的记录时间 |
| `status` | `ok`、`citation_failed`、`max_steps` 或 `model_error` |

`ok` 表示产生回答且编号检查通过。检查要求至少取得非空检索证据、使用 `[文件名.md#编号]`、引用均来自本次工具返回，并拒绝不合法的方括号引用和网页链接。它**不能判断某个片段是否支持相邻事实，也不能保证答案完整、正确或确实遵循了记忆偏好**。即使编造的事实附上了真实编号，也可能通过检查，仍需人工比对 `evidence` 原文。未知问题也可能收到相似度最高但无关的片段；排名分数不是资料足够回答的概率。

八题检索评估按人工指定的来源文件计算 Hit@1 和平均 Source Recall@3，只描述这组小样本的文件召回情况。向量模式的余弦分数与混合模式的 RRF 分数含义不同，不直接比较数值大小。

## 独立解压与代码来源

提交材料不包含模型权重、虚拟环境或下载缓存。独立解压时，先准备 Python 与适合设备的 PyTorch，再安装本目录依赖。没有仓库环境时可把模型分别放在 `solution/models/Qwen2.5-7B-Instruct`、`solution/models/bge-small-zh-v1.5`，程序会回退到本地 `models/`；也可通过两个模型路径参数指定外部绝对路径。保留 `knowledge/` 与 `workspace/` 即可复现固定输入，不需要安装资料描述的虚构应用。

- Agent 与 BGE 基础流程、六份资料及端口/上手任务来自课程 [参考代码](../code/README.md) 与 [agent_v1.py](../code/agent_v1.py)。课程参考目录保持原样，本方案在独立 `solution/` 中实现扩展。
- 7B NF4 加载方案沿用本仓库 [第三周 Agent 实现](../../week03-agent-tool-calling/agent_v0.py) 和 [第三周复现说明](../../week03-agent-tool-calling/experiment-reproduction.md) 中已有的 BitsAndBytes 配置。
- 本周新增实现包括键值记忆更新/删除、安全路径检查的完善、BM25 与 RRF 检索、来源编号检查和一次纠正、候选与错误留存，以及固定实验和无模型单元测试。真实效果与限制以 [作业报告](report.md) 和原始 JSON 为准。
