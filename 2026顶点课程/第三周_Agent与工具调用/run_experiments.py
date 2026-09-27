"""在第一周环境中顺序运行十次真实推理，保存命令和未经修改的输出。"""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
TASK = '读取 test_report.txt，计算测试通过率'
PATH_TASK = ('读取工作目录根目录中的 test_report.txt，计算测试通过率。'
             '调用读取工具时，path 必须使用相对路径 test_report.txt；不要加前导斜杠，也不要写成绝对路径。')
EDGE_TASK = '无需调用工具。请给出 final，answer 必须逐字等于：右花括号 } 是普通字符。'
RUNS = [
    ('01_greedy_legacy', TASK, 'legacy', []),
    ('02_sample_t0.7_p0.8_k20', TASK, 'legacy', ['--sample']),
    ('03_sample_t0.3_p0.8_k20', TASK, 'legacy', ['--sample', '--temperature', '0.3']),
    ('04_sample_t0.3_p0.95_k20', TASK, 'legacy', ['--sample', '--temperature', '0.3', '--top-p', '0.95']),
    ('05_sample_t0.3_p0.95_k50', TASK, 'legacy', ['--sample', '--temperature', '0.3', '--top-p', '0.95', '--top-k', '50']),
    ('06_prompt_explicit_path', PATH_TASK, 'legacy', []),
    ('ext_report_legacy', TASK, 'legacy', []),
    ('ext_report_string-aware', TASK, 'string-aware', []),
    ('ext_brace_legacy', EDGE_TASK, 'legacy', []),
    ('ext_brace_string-aware', EDGE_TASK, 'string-aware', []),
]


def main():
    os.chdir(ROOT)
    Path('logs').mkdir(exist_ok=True)
    Path('workspace').mkdir(exist_ok=True)
    report = Path('workspace/test_report.txt')
    content = 'passed=47\nfailed=3\n'
    if report.exists() and report.read_text(encoding='utf-8') != content:
        raise RuntimeError('已有报告内容与固定实验不符，请先核查。')
    report.write_text(content, encoding='utf-8')
    manifest = []
    for name, task, mode, options in RUNS:
        output = f'results/{name}.json'
        if Path(output).exists():
            raise FileExistsError(f'避免覆盖原始结果：{output}')
        argv = [sys.executable, '-X', 'utf8', 'agent_v0.py', '--backend', 'hf',
                '--model', '../第一周_LLM推理/models/Qwen2.5-1.5B-Instruct', '--device', 'cuda',
                '--task', task, '--max-steps', '6', '--max-failures', '2', '--max-new-tokens', '256',
                '--workdir', 'workspace', '--seed', '42', '--json-stop', mode, '--output', output, *options]
        command = '& ' + ' '.join("'" + arg.replace("'", "''") + "'" for arg in argv)
        print(f'Running {name}', flush=True)
        result = subprocess.run(argv, capture_output=True, text=True, encoding='utf-8',
                                env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
        Path(f'logs/{name}.txt').write_text(command + '\n\nSTDOUT\n' + result.stdout +
                                         '\nSTDERR\n' + result.stderr, encoding='utf-8')
        manifest.append({'name': name, 'command': command, 'argv': argv, 'returncode': result.returncode})
        Path('logs/commands.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        if result.returncode:
            raise RuntimeError(f'{name} 运行失败，见日志')
        data = json.loads(Path(output).read_text(encoding='utf-8'))
        print(f"{name}: {data['status']}, answer={data['answer']!r}", flush=True)


if __name__ == '__main__':
    main()
