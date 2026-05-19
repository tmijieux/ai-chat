from llama_cpp import Llama
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class Message:
    role: str
    content: str
    tokens: int = 0


@dataclass
class ContextStats:
    max_context: int
    used_tokens: int = 0
    messages: List[Message] = field(default_factory=list)

    def remaining(self) -> int:
        return max(0, self.max_context - self.used_tokens)

    def usage_ratio(self) -> float:
        return self.used_tokens / self.max_context if self.max_context else 0.0

    def overflow_risk(self) -> str:
        r = self.usage_ratio()
        if r < 0.5:
            return "low"
        elif r < 0.8:
            return "medium"
        return "high"


class OllamaTokenTracker:
    """
    Token tracker that matches Ollama/llama.cpp tokenization exactly.
    """

    def __init__(self, model_path: str, max_context: int = 4096):
        self.llm = Llama(model_path=model_path, verbose=False, logits_all=False, n_ctx=1, vocab_only=True)
        self.ctx = ContextStats(max_context=max_context)

    # -------------------------
    # Core tokenizer (exact)
    # -------------------------
    def count_tokens(self, text: str) -> int:
        return len(self.llm.tokenize(text.encode("utf-8")))

    # -------------------------
    # Add message to context
    # -------------------------
    def add_message(self, role: str, content: str) -> Message:
        tokens = self.count_tokens(content)

        msg = Message(role=role, content=content, tokens=tokens)
        self.ctx.messages.append(msg)
        self.ctx.used_tokens += tokens

        return msg

    # -------------------------
    # Simulate assistant reply (optional tracking)
    # -------------------------
    def add_assistant_response(self, content: str) -> Message:
        return self.add_message("assistant", content)

    # -------------------------
    # Full context summary
    # -------------------------
    def summary(self) -> Dict:
        return {
            "used_tokens": self.ctx.used_tokens,
            "remaining_tokens": self.ctx.remaining(),
            "usage_ratio": self.ctx.usage_ratio(),
            "overflow_risk": self.ctx.overflow_risk(),
            "message_count": len(self.ctx.messages),
        }

    # -------------------------
    # Reset context
    # -------------------------
    def reset(self):
        self.ctx = ContextStats(max_context=self.ctx.max_context)