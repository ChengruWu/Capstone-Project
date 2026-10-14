"""确定性测试：逐字符模拟生成，检查停止发生在完整 JSON 边界。"""
import json
import unittest

from agent_v0 import _BalancedJSONStop


class CharacterTokenizer:
    def decode(self, ids, **kwargs):
        return ''.join(chr(i) for i in ids)


class TokenIds:
    def __init__(self, text):
        self.ids = list(map(ord, text))

    def __getitem__(self, key):
        row, positions = key
        assert row == 0
        return self.ids[positions]


def stopped(text, mode="string-aware", prompt=""):
    return _BalancedJSONStop(CharacterTokenizer(), len(prompt), mode)(TokenIds(prompt + text), None)


class JSONStopTests(unittest.TestCase):
    def test_complete_objects_and_all_prefixes(self):
        for answer in ['plain', '}', '{', '{}', '引号 " 后面 }', '反斜杠 \\ 后面 {', '\\"}']:
            payload = json.dumps({'action': 'final', 'answer': answer}, ensure_ascii=False)
            with self.subTest(answer=answer):
                for i in range(len(payload)):
                    self.assertFalse(stopped(payload[:i]), payload[:i])
                self.assertTrue(stopped(payload))

    def test_nested_object(self):
        payload = '{"action":"tool","arguments":{"nested":{"x":1}}}'
        for i in range(len(payload)):
            self.assertFalse(stopped(payload[:i]))
        self.assertTrue(stopped(payload))

    def test_incomplete_and_no_object(self):
        for payload in ['', '没有对象', '{', '{"answer":"}', '{"answer":"abc\\"']:
            self.assertFalse(stopped(payload))

    def test_first_object_and_prefix(self):
        self.assertTrue(stopped('```json\n{"answer":"ok"}\n```'))
        self.assertTrue(stopped('{"a":1}{"b":2}'))

    def test_prompt_is_excluded(self):
        self.assertFalse(stopped('', prompt='{"already":"complete"}'))
        self.assertFalse(stopped('{"answer":"}', prompt='}}}'))
        self.assertTrue(stopped('{"answer":"ok"}', prompt='{{{'))

    def test_legacy_regression(self):
        self.assertTrue(stopped('{"action":"final","answer":"}', 'legacy'))
        self.assertFalse(stopped('{"action":"final","answer":"}'))
        payload = '{"action":"final","answer":"{"}'
        self.assertFalse(stopped(payload, 'legacy'))
        self.assertTrue(stopped(payload))


if __name__ == '__main__':
    unittest.main(verbosity=2)
