import json
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.lamina_toolcalls import ToolStream, escaped, hold_suffix  # noqa: E402

TOOLS = [
    {"type": "function", "function": {"name": "write", "parameters": {"type": "object", "required": ["path", "content"],
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "grep", "parameters": {"type": "object", "required": ["pattern"],
        "properties": {"pattern": {"type": "string"}, "limit": {"type": "integer"}, "regex": {"type": "boolean"},
                       "paths": {"type": "array", "items": {"type": "string"}}}}}},
]


def call(name, **parameters):
    body = "".join(f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in parameters.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def replay(text, step, tools=TOOLS, truncated=False, choice="auto"):
    """Feeds text in pieces of `step` characters; returns (content, calls) rebuilt from the deltas."""
    stream = ToolStream(tools, choice)
    content, calls = "", {}
    deltas = []
    for end in list(range(step, len(text), step)):
        deltas += stream.feed(text[:end])
    deltas += stream.finish(text, truncated)
    for delta in deltas:
        content += delta.get("content", "")
        for piece in delta.get("tool_calls", []):
            entry = calls.setdefault(piece["index"], {"name": None, "arguments": ""})
            entry["name"] = piece.get("function", {}).get("name", entry["name"])
            entry["arguments"] += piece.get("function", {}).get("arguments", "")
    return content, [calls[i] for i in sorted(calls)], stream


class ToolStreamTest(unittest.TestCase):
    def check(self, text, expected_calls, expected_text=""):
        for step in (1, 2, 3, 5, 7, 11, 64, 10_000):
            content, calls, stream = replay(text, step)
            self.assertEqual(content.strip(), expected_text, f"text, step {step}")
            self.assertEqual([(c["name"], json.loads(c["arguments"])) for c in calls], expected_calls, f"calls, step {step}")
            self.assertEqual([c["function"]["arguments"] for c in stream.calls], [c["arguments"] for c in calls])
            for rebuilt in calls:  # the streamed fragments are the same text as the finished arguments
                self.assertEqual(rebuilt["arguments"], json.dumps(json.loads(rebuilt["arguments"]), ensure_ascii=False))

    def test_plain_text_passes_through(self):
        self.check("Just an answer with <b>tags</b> and a < sign.", [], "Just an answer with <b>tags</b> and a < sign.")

    def test_single_call_with_text_before(self):
        self.check("I will create the file.\n\n" + call("write", path="index.html", content="<h1>Hi</h1>\nline 2"),
                   [("write", {"path": "index.html", "content": "<h1>Hi</h1>\nline 2"})], "I will create the file.")

    def test_escapes_quotes_backslashes_and_unicode(self):
        value = 'say "hi"\\n\ttab \u0436\u0438\u0432\u043e\u0439 \U0001F600'
        self.check(call("write", path="a.txt", content=value), [("write", {"path": "a.txt", "content": value})])

    def test_non_string_parameters_are_json(self):
        text = call("grep", pattern="foo", limit="25", regex="true", paths='["a", "b"]')
        self.check(text, [("grep", {"pattern": "foo", "limit": 25, "regex": True, "paths": ["a", "b"]})])

    def test_two_calls(self):
        text = "Checking.\n" + call("grep", pattern="a") + "\n" + call("write", path="x", content="y")
        self.check(text, [("grep", {"pattern": "a"}), ("write", {"path": "x", "content": "y"})], "Checking.")

    def test_value_that_mentions_the_closing_tag(self):
        value = "first line\n</param>\nsecond"
        self.check(call("write", path="p", content=value), [("write", {"path": "p", "content": value})])

    def test_arguments_arrive_while_the_value_is_written(self):
        text = call("write", path="a", content="x" * 400)
        stream = ToolStream(TOOLS)
        early = stream.feed(text[:len(text) - 60])  # the call is not closed yet
        sent = "".join(p["function"].get("arguments", "") for d in early for p in d.get("tool_calls", []))
        self.assertTrue(sent.startswith('{"path": "a", "content": "xxx'), sent)
        self.assertGreater(len(sent), 300)

    def test_missing_closing_tags_are_accepted_after_a_normal_stop(self):
        text = "<tool_call>\n<function=write>\n<parameter=path>\na\n</parameter>\n<parameter=content>\nb\n</parameter>\n"
        content, calls, stream = replay(text, 4)
        self.assertEqual(json.loads(calls[0]["arguments"]), {"path": "a", "content": "b"})
        self.assertEqual(len(stream.calls), 1)

    def test_cut_off_call_is_an_error_that_names_the_limit(self):
        text = call("write", path="a", content="long " * 50)
        with self.assertRaisesRegex(RuntimeError, "max_tokens"):
            replay(text[:len(text) // 2], 9, truncated=True)

    def test_stop_in_the_middle_of_a_value_without_a_limit_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, "incomplete generated tool call"):
            replay(call("write", path="a", content="long " * 50)[:90], 9)

    def test_unknown_tool_duplicate_parameter_and_invalid_arguments(self):
        with self.assertRaisesRegex(RuntimeError, "unknown tool"):
            replay(call("missing", x="1"), 6)
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            replay("<tool_call>\n<function=grep>\n<parameter=pattern>\na\n</parameter>\n<parameter=pattern>\nb\n</parameter>\n</function>\n</tool_call>", 6)
        with self.assertRaisesRegex(RuntimeError, "invalid generated tool arguments"):
            replay(call("write", path="only"), 6)

    def test_unknown_format_is_reported(self):
        with self.assertRaisesRegex(RuntimeError, "format Lamina does not understand"):
            replay('<tool_call>\n{"name": "write", "arguments": {}}\n</tool_call>', 5)

    def test_required_and_forced_choice(self):
        with self.assertRaisesRegex(RuntimeError, "required"):
            replay("no call here", 4, choice="required")
        forced = {"type": "function", "function": {"name": "grep"}}
        with self.assertRaisesRegex(RuntimeError, "selected tool"):
            replay(call("write", path="a", content="b"), 4, choice=forced)

    def test_helpers(self):
        self.assertEqual(hold_suffix("abc <tool_c", ["<tool_call>"]), "abc ")
        self.assertEqual(hold_suffix("abc <b", ["<tool_call>"]), "abc <b")
        self.assertEqual(escaped('a"b\n'), 'a\\"b\\n')


if __name__ == "__main__":
    unittest.main()
