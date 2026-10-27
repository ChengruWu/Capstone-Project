"""Agent loop regression tests using scripted responses, without loading models.

These are constructed protocol tests, not measurements of model performance.
Run from this directory: python -B -m unittest test_agent -v
"""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from agent_v1 import parse_tool_calls, run_agent, save_result, validate_citations


CHUNK = {
    "id": "configuration.md#1",
    "text": "在 .env 中设置 APP_PORT=8001，然后正常退出旧实例并重新启动。",
    "score": 0.9,
}
ANSWER = "设置 APP_PORT=8001，再正常退出并重新启动。[configuration.md#1]"


def tool_call(name="search_knowledge", **arguments):
    return "<tool_call>" + json.dumps(
        {"name": name, "arguments": arguments}, ensure_ascii=False
    ) + "</tool_call>"


class ScriptedLLM:
    """Return fixed raw responses and snapshot the actual conversation received."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.inputs = []
        self.last_generation = {}

    def __call__(self, messages):
        self.inputs.append(copy.deepcopy(messages))
        if not self.responses:
            raise AssertionError("Agent requested an unexpected additional response")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        self.last_generation = {"stub": True, "response_number": len(self.inputs)}
        return response


class AgentLoopTests(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def read(path):
            self.calls.append(("read_file", path))
            return {"text": "请处理端口占用问题", "truncated": False}

        def search(query):
            self.calls.append(("search_knowledge", query))
            return [dict(CHUNK)]

        self.tools = {"read_file": read, "search_knowledge": search}
        self.notes = {"language": "中文", "style": "简短步骤"}

    def run_script(self, *responses, **kwargs):
        llm = ScriptedLLM(*responses)
        result = run_agent("请读取任务并查询项目资料", llm, self.tools, self.notes, **kwargs)
        return result, llm

    def test_success_executes_both_tools_and_preserves_sources_and_memory(self):
        result, llm = self.run_script(
            tool_call("read_file", path="task.txt"),
            tool_call(query="端口如何修改"), ANSWER,
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["answer"], ANSWER)
        self.assertEqual(self.calls, [("read_file", "task.txt"),
                                     ("search_knowledge", "端口如何修改")])
        self.assertEqual(result["evidence"], {CHUNK["id"]: CHUNK})
        self.assertEqual(result["memory_used"], self.notes)
        self.assertIsNot(result["memory_used"], self.notes)
        self.assertEqual(result["trace"][0]["generation"]["response_number"], 1)
        self.assertIn("简短步骤", llm.inputs[0][0]["content"])
        self.assertEqual(result["citation_repairs"], 0)

    def test_one_citation_repair_keeps_both_raw_responses(self):
        invalid = "把端口改成 8001。[fiction.md#9]"
        result, llm = self.run_script(tool_call(query="端口"), invalid, ANSWER)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["citation_repairs"], 1)
        self.assertEqual([x["raw"] for x in result["trace"]],
                         [tool_call(query="端口"), invalid, ANSWER])
        self.assertFalse(result["trace"][1]["citation_validation"]["passed"])
        self.assertTrue(result["trace"][2]["citation_validation"]["passed"])
        feedback = llm.inputs[2][-1]
        self.assertEqual(feedback["role"], "user")
        self.assertIn("configuration.md#1", feedback["content"])

    def test_second_invalid_answer_fails_without_second_repair(self):
        invalid = "改端口。[fiction.md#1]"
        result, llm = self.run_script(tool_call(query="端口"), invalid, invalid)
        self.assertEqual(result["status"], "citation_failed")
        self.assertEqual(result["citation_repairs"], 1)
        self.assertEqual(len(llm.inputs), 3)
        self.assertEqual(result["answer"], invalid)

    def test_check_policy_records_failure_without_repair(self):
        invalid = "改为 8001。"
        result, llm = self.run_script(tool_call(query="端口"), invalid,
                                      citation_policy="check")
        self.assertEqual(result["status"], "citation_failed")
        self.assertEqual(result["answer"], invalid)
        self.assertEqual(result["citation_repairs"], 0)
        self.assertEqual(len(llm.inputs), 2)

    def test_fictional_id_is_rejected_even_when_another_id_is_valid(self):
        invalid = ANSWER + "[fiction.md#3]"
        result, _ = self.run_script(tool_call(query="端口"), invalid,
                                    citation_policy="check")
        self.assertEqual(result["status"], "citation_failed")
        self.assertEqual(result["citation_validation"]["unknown"], ["fiction.md#3"])

    def test_web_source_is_rejected_with_valid_fragment_id(self):
        invalid = ANSWER + " 来源：https://example.invalid/manual"
        result, _ = self.run_script(tool_call(query="端口"), invalid,
                                    citation_policy="check")
        self.assertEqual(result["status"], "citation_failed")
        self.assertTrue(result["citation_validation"]["external_links"])

    def test_no_retrieval_cannot_pass_using_a_plausible_id(self):
        result, _ = self.run_script(ANSWER, citation_policy="check")
        self.assertEqual(result["status"], "citation_failed")
        self.assertEqual(result["evidence"], {})
        self.assertEqual(self.calls, [])
        self.assertIn("尚未成功检索知识库", result["citation_validation"]["issues"])

    def test_repair_can_request_missing_retrieval_then_answer(self):
        result, _ = self.run_script(ANSWER, tool_call(query="端口"), ANSWER)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["citation_repairs"], 1)
        self.assertEqual(len(result["trace"]), 3)

    def test_invalid_tool_parameter_is_feedback_and_can_be_fixed(self):
        invalid = tool_call(query=123)
        result, llm = self.run_script(invalid, tool_call(query="端口"), ANSWER)
        self.assertEqual(result["status"], "ok")
        failure = result["trace"][0]["observations"][0]
        self.assertEqual(failure["status"], "error")
        self.assertIn("ValueError", failure["error"])
        self.assertEqual(self.calls, [("search_knowledge", "端口")])
        observation = json.loads(llm.inputs[1][-1]["content"])
        self.assertEqual(llm.inputs[1][-1]["role"], "tool")
        self.assertEqual(observation["status"], "error")

    def test_extra_tool_argument_is_rejected_before_execution(self):
        result, _ = self.run_script(tool_call(query="端口", unexpected="x"),
                                    tool_call(query="端口"), ANSWER)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["trace"][0]["observations"][0]["status"], "error")
        self.assertEqual(len(self.calls), 1)

    def test_unknown_tool_does_not_execute(self):
        result, _ = self.run_script(tool_call("shell", command="echo hi"),
                                    tool_call(query="端口"), ANSWER)
        self.assertEqual(result["status"], "ok")
        self.assertIn("未知工具名", result["trace"][0]["observations"][0]["error"])
        self.assertEqual(len(self.calls), 1)

    def test_tool_exception_is_feedback_and_leaves_no_evidence(self):
        def failing_search(query):
            raise OSError("构造的检索失败")

        self.tools["search_knowledge"] = failing_search
        result, _ = self.run_script(tool_call(query="端口"), ANSWER,
                                    citation_policy="check")
        self.assertEqual(result["status"], "citation_failed")
        self.assertEqual(result["evidence"], {})
        self.assertIn("OSError", result["trace"][0]["observations"][0]["error"])

    def test_malformed_retrieval_cannot_partially_add_evidence(self):
        self.tools["search_knowledge"] = lambda query: [dict(CHUNK), {"id": "broken.md#1"}]
        result, _ = self.run_script(tool_call(query="端口"), ANSWER,
                                    citation_policy="check")
        self.assertEqual(result["status"], "citation_failed")
        self.assertEqual(result["evidence"], {})
        self.assertEqual(result["trace"][0]["observations"][0]["status"], "error")

    def test_failed_retrieval_does_not_overwrite_previous_valid_evidence(self):
        responses = iter([[dict(CHUNK)], [{**CHUNK, "text": "不应覆盖"}, {"id": 1}]])
        self.tools["search_knowledge"] = lambda query: next(responses)
        result, _ = self.run_script(tool_call(query="端口"), tool_call(query="更多"), ANSWER)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["evidence"][CHUNK["id"]]["text"], CHUNK["text"])
        self.assertEqual(result["trace"][1]["observations"][0]["status"], "error")

    def test_empty_retrieval_result_cannot_pass(self):
        self.tools["search_knowledge"] = lambda query: []
        result, _ = self.run_script(tool_call(query="端口"), ANSWER,
                                    citation_policy="check")
        self.assertEqual(result["status"], "citation_failed")
        self.assertEqual(result["evidence"], {})
        self.assertEqual(result["trace"][0]["observations"][0]["status"], "ok")

    def test_multiple_tool_calls_execute_in_order(self):
        raw = tool_call("read_file", path="task.txt") + "\n" + tool_call(query="端口")
        result, _ = self.run_script(raw, ANSWER)
        self.assertEqual(result["status"], "ok")
        self.assertEqual([x[0] for x in self.calls], ["read_file", "search_knowledge"])
        self.assertEqual(len(result["trace"][0]["observations"]), 2)

    def test_too_many_calls_are_rejected_without_execution(self):
        raw = "\n".join(tool_call(query="端口") for _ in range(5))
        result, _ = self.run_script(raw, tool_call(query="端口"), ANSWER)
        self.assertEqual(result["status"], "ok")
        self.assertIn("四", result["trace"][0]["format_error"])
        self.assertEqual(len(self.calls), 1)

    def test_malformed_tool_protocol_can_be_fixed(self):
        for malformed in [
            '<tool_call>{"name": "search_knowledge"}',
            '<tool_call>not json</tool_call>',
            '<tool_call>{"name":"search_knowledge"}</tool_call>',
            '<tool_call>[]</tool_call>',
        ]:
            with self.subTest(raw=malformed):
                result, llm = self.run_script(malformed, tool_call(query="端口"), ANSWER)
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["trace"][0]["raw"], malformed)
                self.assertIn("format_error", result["trace"][0])
                self.assertEqual(llm.inputs[1][-1]["role"], "user")

    def test_empty_model_response_is_recorded_then_retried(self):
        result, _ = self.run_script("", tool_call(query="端口"), ANSWER)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["trace"][0]["raw"], "")
        self.assertIn("为空", result["trace"][0]["format_error"])

    def test_max_steps_records_exhaustion_without_answer(self):
        result, llm = self.run_script(tool_call(query="端口"), max_steps=1)
        self.assertEqual(result["status"], "max_steps")
        self.assertIsNone(result["answer"])
        self.assertEqual(len(llm.inputs), 1)
        self.assertEqual(len(result["evidence"]), 1)

    def test_invalid_final_at_last_step_fails_without_spending_repair(self):
        result, _ = self.run_script("没有引用", max_steps=1)
        self.assertEqual(result["status"], "citation_failed")
        self.assertEqual(result["citation_repairs"], 0)

    def test_repair_exhaustion_keeps_candidate_without_claiming_final_answer(self):
        invalid = "没有检索依据的旧回答"
        result, _ = self.run_script(invalid, tool_call(query="端口"), max_steps=2)
        self.assertEqual(result["status"], "max_steps")
        self.assertIsNone(result["answer"])
        self.assertNotIn("citation_validation", result)
        self.assertEqual(result["answer_candidates"][0]["answer"], invalid)
        self.assertFalse(result["answer_candidates"][0]["validation"]["passed"])

    def test_model_error_during_repair_keeps_candidate_without_final_answer(self):
        invalid = "没有检索依据的旧回答"
        result, _ = self.run_script(invalid, RuntimeError("纠正阶段失败"))
        self.assertEqual(result["status"], "model_error")
        self.assertIsNone(result["answer"])
        self.assertNotIn("citation_validation", result)
        self.assertEqual(result["answer_candidates"][0]["answer"], invalid)
        self.assertEqual(result["trace"][1]["error"], "RuntimeError: 纠正阶段失败")

    def test_model_exception_is_preserved(self):
        result, _ = self.run_script(RuntimeError("构造的模型失败"))
        self.assertEqual(result["status"], "model_error")
        self.assertIsNone(result["answer"])
        self.assertEqual(result["trace"][0]["error"], "RuntimeError: 构造的模型失败")
        self.assertNotIn("raw", result["trace"][0])
        self.assertGreaterEqual(result["elapsed_s"], 0)

    def test_invalid_loop_configuration_is_rejected(self):
        for options in [{"max_steps": 0}, {"citation_policy": "ignore"}]:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    self.run_script(**options)


class CitationAndPersistenceTests(unittest.TestCase):
    def test_semantically_wrong_answer_can_pass_id_check_scope_is_explicit(self):
        answer = "日志保留 999 天。[configuration.md#1]"
        result = validate_citations(answer, {CHUNK["id"]: CHUNK})
        self.assertTrue(result["passed"])
        self.assertIn("不保证事实", result["scope"])

    def test_malformed_bracket_is_reported(self):
        result = validate_citations(ANSWER + "[configuration.md]", {CHUNK["id"]: CHUNK})
        self.assertFalse(result["passed"])
        self.assertEqual(result["malformed"], ["configuration.md"])

    def test_empty_retrieval_does_not_count_as_evidence(self):
        result = validate_citations("资料未提供相关信息。", {})
        self.assertFalse(result["passed"])
        self.assertIn("尚未成功检索知识库", result["issues"])

    def test_tool_parser_retains_json_arguments_exactly(self):
        raw = tool_call(query='含双引号 " 和花括号 { } 的问题')
        self.assertEqual(parse_tool_calls(raw)[0]["arguments"]["query"],
                         '含双引号 " 和花括号 { } 的问题')

    def test_result_save_preserves_unicode_and_refuses_overwrite(self):
        result = {"answer": ANSWER, "trace": [{"raw": "保留模型原文"}]}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "nested" / "result.json"
            save_result(result, output)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), result)
            with self.assertRaises(FileExistsError):
                save_result({"answer": "覆盖"}, output)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), result)


if __name__ == "__main__":
    unittest.main()
