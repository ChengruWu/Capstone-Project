"""验证实验配置、原文留存及拓展对照；不会改写实验 JSON。"""
import hashlib
import json
from pathlib import Path

from run_experiments import ROOT, RUNS, TASK, PATH_TASK, EDGE_TASK


def main():
    records = []
    checks = []
    manifest = json.loads((ROOT / 'logs/commands.json').read_text(encoding='utf-8'))
    assert len(manifest) == len(RUNS) == 10
    for (name, task, mode, _), command in zip(RUNS, manifest):
        path = ROOT / 'results' / f'{name}.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        assert command['name'] == name and command['returncode'] == 0
        assert data['task'] == task and data['config']['json_stop'] == mode
        assert data['steps'] == len(data['trace'])
        assert [m['content'] for m in data['messages'] if m['role'] == 'assistant'] == [r['raw'] for r in data['trace']]
        assert data['environment']['dtype'] == 'torch.bfloat16'
        assert data['config']['seed'] == 42 and data['config']['device'] == 'cuda'
        successful_tools = [r['decision']['name'] for r in data['trace'] if r.get('observation', {}).get('status') == 'ok']
        checks.append({'file': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                       'status': data['status'], 'answer': data['answer'], 'successful_tools': successful_tools,
                       'answer_contains_94_percent': '94%' in (data['answer'] or ''),
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
    assert records[5]['task'] == PATH_TASK
    prompt_changed = {key for key in records[0]['config']
                      if records[0]['config'][key] != records[5]['config'][key]} - {'output'}
    assert prompt_changed == {'task'}, prompt_changed
    assert (ROOT / 'workspace/test_report.txt').read_text(encoding='utf-8') == 'passed=47\nfailed=3\n'
    output = {'verification': 'PASS', 'checks': checks,
              'prompt_pair': {'original_file': 'results/01_greedy_legacy.json',
                              'original_status': records[0]['status'], 'original_answer': records[0]['answer'],
                              'replacement_file': 'results/06_prompt_explicit_path.json',
                              'replacement_status': records[5]['status'],
                              'replacement_answer': records[5]['answer'],
                              'only_changed_config': sorted(prompt_changed)}}
    (ROOT / 'logs/result_validation.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
