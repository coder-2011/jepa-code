from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "benchmark_local.py"
SPEC = importlib.util.spec_from_file_location("benchmark_local_script", SCRIPT_PATH)
benchmark_local = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(benchmark_local)


class FakeThinkingTokenizer:
    chat_template = "fake"

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(kwargs)
        prompt = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
        if kwargs.get("add_generation_prompt"):
            prompt += "\nassistant:"
        if kwargs.get("enable_thinking") is False:
            prompt += "\n<think>\n\n</think>\n\n"
        return prompt


class FakeStrictTokenizer(FakeThinkingTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        if "enable_thinking" in kwargs:
            raise TypeError("unexpected keyword argument")
        return super().apply_chat_template(messages, **kwargs)


def test_render_prompt_can_disable_qwen_thinking_template():
    tokenizer = FakeThinkingTokenizer()
    prompt = benchmark_local.render_prompt(
        tokenizer,
        [{"role": "user", "content": "Question?"}],
        disable_thinking=True,
    )

    assert tokenizer.calls[0]["enable_thinking"] is False
    assert "<think>" in prompt


def test_render_prompt_falls_back_when_tokenizer_rejects_enable_thinking():
    tokenizer = FakeStrictTokenizer()
    prompt = benchmark_local.render_prompt(
        tokenizer,
        [{"role": "user", "content": "Question?"}],
        disable_thinking=True,
    )

    assert "assistant:" in prompt
    assert tokenizer.calls[0].get("enable_thinking") is None
