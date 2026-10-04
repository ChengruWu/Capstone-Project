"""验证实验配置、原文留存及拓展对照；不会改写实验 JSON。"""
import hashlib
import argparse
import json
import re
from pathlib import Path

from run_experiments import ROOT, RUNS, TASK, PATH_TASK, EDGE_TASK


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--logs-dir', default='logs')
    parser.add_argument('--expected-quantization', choices=['none', 'nf4'], default='none')
    parser.add_argument('--validation-output', help='另存核验摘要，保留已有实验日志（相对第三周目录或绝对路径）')
    args = parser.parse_args()
    results_dir, logs_dir = ROOT / args.results_dir, ROOT / args.logs_dir
    records = []
    checks = []
    manifest = json.loads((logs_dir / 'commands.json').read_text(encoding='utf-8'))
    assert len(manifest) == len(RUNS) == 10
    for (name, task, mode, _), command in zip(RUNS, manifest):
        path = results_dir / f'{name}.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        assert command['name'] == name and command['returncode'] == 0
        assert data['task'] == task and data['config']['json_stop'] == mode
        assert data['steps'] == len(data['trace'])
        assert [m['content'] for m in data['messages'] if m['role'] == 'assistant'] == [r['raw'] for r in data['trace']]
        assert data['environment']['dtype'] == 'torch.bfloat16'
        assert data['config'].get('quantization', 'none') == args.expected_quantization
        assert data['environment'].get('quantization', 'none') == args.expected_quantization
        if args.expected_quantization == 'nf4':
            assert data['environment']['device_map'] == {'': '0'}, '模型未完整放入 GPU'
            assert data['environment']['model_memory_bytes'] < data['environment']['gpu_total_bytes']
        assert data['config']['seed'] == 42 and data['config']['device'] == 'cuda'
        successful_tools = [r['decision']['name'] for r in data['trace'] if r.get('observation', {}).get('status') == 'ok']
        checks.append({'file': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                       'status': data['status'], 'answer': data['answer'], 'successful_tools': successful_tools,
                       'answer_contains_94_percent': bool(re.search(r'(?<![\d.])94(?:\.0+)?\s*[%％]', data['answer'] or '')),
                       'report_tool_chain_complete': task in (TASK, PATH_TASK)
                           and 'read_file' in successful_tools and 'calculator' in successful_tools
                           and any(r.get('decision', {}).get('name') == 'calculator'
                                   and r.get('observation', {}).get('status') == 'ok'
                                   and r['observation'].get('result') in (94, 0.94) for r in data['trace']),
                       'edge_exact_match': data['answer'] == '右花括号 } 是普通字符。' if task == EDGE_TASK else None})
        records.append(data)
    for index, field in enumerate(['sample', 'temperature', 'top_p', 'top_k'], 1):
        before, after = records[index - 1]['config'], records[index]['config']
        changed = {key for key in before if before[key] != after[key]} - {'output'}
        assert changed == {field}, changed
    for a, b in [(6, 7), (8, 9)]:
        changed = {key for key in records[a]['config'] if records[a]['config'][key] != records[b]['config'][key]} - {'output'}
        assert changed == {'json_stop'}, changed
    assert all(r['task'] == TASK for r in records[:5])
    assert len({r['config']['model'] for r in records}) == 1
    assert records[5]['task'] == PATH_TASK
    prompt_changed = {key for key in records[0]['config']
                      if records[0]['config'][key] != records[5]['config'][key]} - {'output'}
    assert prompt_changed == {'task'}, prompt_changed
    assert (ROOT / 'workspace/test_report.txt').read_text(encoding='utf-8') == 'passed=47\nfailed=3\n'
    output = {'verification': 'PASS', 'checks': checks,
              'prompt_pair': {'original_file': str((results_dir / '01_greedy_legacy.json').relative_to(ROOT)),
                              'original_status': records[0]['status'], 'original_answer': records[0]['answer'],
                              'replacement_file': str((results_dir / '06_prompt_explicit_path.json').relative_to(ROOT)),
                              'replacement_status': records[5]['status'],
                              'replacement_answer': records[5]['answer'],
                              'only_changed_config': sorted(prompt_changed)}}
    recommended = results_dir / '07_recommended_sample_string-aware.json'
    if recommended.exists():
        data = json.loads(recommended.read_text(encoding='utf-8'))
        changed = {key for key in records[2]['config']
                   if records[2]['config'][key] != data['config'][key]} - {'output'}
        assert changed == {'json_stop'} and data['config']['json_stop'] == 'string-aware'
        assert [m['content'] for m in data['messages'] if m['role'] == 'assistant'] == [r['raw'] for r in data['trace']]
        assert data['status'] == 'ok' and re.search(r'(?<![\d.])94(?:\.0+)?\s*[%％]', data['answer'])
        assert [r.get('decision', {}).get('name') for r in data['trace']
                if r.get('observation', {}).get('status') == 'ok'] == ['read_file', 'calculator']
        output['recommended'] = {'file': str(recommended.relative_to(ROOT)),
                                 'sha256': hashlib.sha256(recommended.read_bytes()).hexdigest(),
                                 'status': data['status'], 'answer': data['answer'],
                                 'only_changed_config_vs_run_3': sorted(changed)}
    validation_output = ROOT / args.validation_output if args.validation_output else logs_dir / 'result_validation.json'
    validation_output.parent.mkdir(parents=True, exist_ok=True)
    validation_output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
