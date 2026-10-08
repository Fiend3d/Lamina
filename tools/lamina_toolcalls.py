"""Streaming parser for Qwen3.6's XML-style tool calls.

The model writes a call as

    <tool_call>
    <function=write>
    <parameter=path>
    index.html
    </parameter>
    </function>
    </tool_call>

ToolStream turns the growing text of an answer into OpenAI streaming deltas while the model is still
generating: ordinary text passes through at once, and the arguments of a call are sent as they are
written, so a client can show a long file growing instead of waiting for the whole answer. Held back
are only the few characters that might still turn into a tag. The finished arguments equal what the
buffered parser (lamina_chat.tool_message) produces.
"""
import json
import re
import uuid


def hold_suffix(text, markers):
    """Text without a trailing piece that could still grow into one of the markers."""
    for length in range(min(max(map(len, markers), default=0), len(text)), 0, -1):
        if any(marker.startswith(text[-length:]) for marker in markers):
            return text[:-length]
    return text


def escaped(text):
    """The inside of a JSON string literal for text (no surrounding quotes)."""
    return json.dumps(text, ensure_ascii=False)[1:-1]


OPEN, END_PARAMETER = "<tool_call>", "</parameter>"
HEADER = re.compile(r"<tool_call>\s*<function=([^>\n]+)>")
PARAMETER = re.compile(r"\s*(?:<parameter=([^>\n]+)>|(</function>))")
CLOSE = re.compile(r"\s*</tool_call>")


class ToolStream:
    def __init__(self, tools, choice="auto"):
        self.available = {t["function"]["name"]: t["function"] for t in tools}
        self.choice = choice
        self.calls = []          # finished calls, in the shape of an OpenAI message's tool_calls
        self.state = "text"      # text, params, value or close (header is matched in one piece)
        self.pos = 0             # everything before pos has been handled
        self.name = None
        self.args = {}
        self.fragments = []
        self.key = None
        self.is_string = False
        self.value_start = 0
        self.value_sent = 0
        self.call_id = None
        self.started = 0

    def feed(self, content):
        """Deltas for the answer text seen so far (content only ever grows)."""
        out = []
        while True:
            if self.state == "text":
                found = content.find(OPEN, self.pos)
                if found < 0:
                    safe = hold_suffix(content, [OPEN])
                    if len(safe) > self.pos:
                        out.append({"content": safe[self.pos:]})
                        self.pos = len(safe)
                    break
                end = len(content[:found].rstrip())
                if end > self.pos:
                    out.append({"content": content[self.pos:end]})
                match = HEADER.match(content, found)
                if not match:
                    self.pos = max(self.pos, end)
                    break
                name = match.group(1).strip()
                if name not in self.available:
                    raise RuntimeError(f"model requested an unknown tool: {name}")
                self.name, self.args, self.fragments = name, {}, []
                self.call_id = "call_" + uuid.uuid4().hex
                out.append({"tool_calls": [{"index": self.started, "id": self.call_id, "type": "function",
                                            "function": {"name": name, "arguments": ""}}]})
                self.started += 1
                self.pos = match.end()
                self.state = "params"
                self._emit(out, "{")
            elif self.state == "params":
                match = PARAMETER.match(content, self.pos)
                if not match:
                    break
                self.pos = match.end()
                if match.group(2):  # </function>
                    self._emit(out, "}")
                    self.state = "close"
                    continue
                key = match.group(1).strip()
                if key in self.args:
                    raise RuntimeError("duplicate tool parameter")
                properties = self.available[self.name].get("parameters", {}).get("properties", {})
                self.key, self.is_string = key, properties.get(key, {}).get("type") == "string"
                self.args[key] = None
                separator = ", " if len(self.fragments) > 1 else ""
                self._emit(out, f"{separator}{json.dumps(key, ensure_ascii=False)}: " + ('"' if self.is_string else ""))
                self.value_start, self.value_sent = self.pos, 0
                self.state = "value"
            elif self.state == "value":
                end = content.find(END_PARAMETER, self.value_start)
                if end >= 0:
                    value = content[self.value_start:end].strip("\n")
                    if self.is_string:
                        self.args[self.key] = value
                        self._emit(out, escaped(value[self.value_sent:]) + '"')
                    else:
                        try:
                            parsed = json.loads(value)
                        except ValueError:
                            parsed = value
                        self.args[self.key] = parsed
                        self._emit(out, json.dumps(parsed, ensure_ascii=False))
                    self.pos = end + len(END_PARAMETER)
                    self.state = "params"
                    continue
                if self.is_string:
                    # the part of the value that cannot change any more: no leading or trailing newlines,
                    # nothing that might be the start of the closing tag
                    stable = hold_suffix(content[self.value_start:].lstrip("\n"), [END_PARAMETER]).rstrip("\n")
                    if len(stable) > self.value_sent:
                        self._emit(out, escaped(stable[self.value_sent:]))
                        self.value_sent = len(stable)
                break
            else:  # close
                match = CLOSE.match(content, self.pos)
                if not match:
                    break
                self._finish_call()
                self.pos = match.end()
                self.state = "text"
        return out

    def finish(self, content, truncated):
        """Final deltas once generation has ended; raises when a call cannot be completed."""
        out = self.feed(content)
        cut_off = ("The answer reached the max_tokens limit while the model was writing a tool call, so the call "
                   "is incomplete. Allow longer answers (up to the context length) or ask for a smaller step.")
        if self.state == "text":
            rest = content[self.pos:].rstrip()
            if OPEN in rest:  # a call that never got as far as its first parameter
                raise RuntimeError(cut_off if truncated else
                                   "the model wrote a tool call in a format Lamina does not understand: " + rest[:300])
            if rest:
                out.append({"content": rest})
        elif truncated:
            raise RuntimeError(cut_off + f" (tool: {self.name})")
        elif not content[self.pos:].strip() and (self.state == "close" or (self.state == "params" and len(self.fragments) > 1)):
            # the model stopped normally but left out the closing tags
            if self.state == "params":
                self._emit(out, "}")
            self._finish_call()
        else:
            raise RuntimeError("incomplete generated tool call")
        forced = self.choice.get("function", {}).get("name") if isinstance(self.choice, dict) else None
        if (self.choice == "required" or forced) and not self.calls:
            raise RuntimeError("model did not produce the required tool call")
        if forced and any(call["function"]["name"] != forced for call in self.calls):
            raise RuntimeError("model did not call the selected tool")
        return out

    def _emit(self, out, text):
        if not text:
            return
        self.fragments.append(text)
        out.append({"tool_calls": [{"index": self.started - 1, "function": {"arguments": text}}]})

    def _finish_call(self):
        import jsonschema
        try:
            jsonschema.validate(self.args, self.available[self.name].get("parameters", {"type": "object"}))
        except jsonschema.ValidationError as error:
            raise RuntimeError("invalid generated tool arguments: " + error.message) from error
        self.calls.append({"id": self.call_id, "type": "function",
                           "function": {"name": self.name, "arguments": "".join(self.fragments)}})
        self.state = "text"
