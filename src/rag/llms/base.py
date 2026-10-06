"""
Base class for language models.

In RAG the LLM is the "G" (Generation): it reads the retrieved context and
writes the final answer.  We keep it behind a tiny interface so the pipeline is
provider-agnostic:

    generate(prompt, system=...) -> str
    chat(messages)               -> str

`messages` is the standard chat format:
    [{"role": "system", "content": "..."},
     {"role": "user", "content": "..."}]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class BaseLLM(ABC):
    """Abstract chat/generation model."""

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send a list of chat messages, return the assistant's reply text."""
        raise NotImplementedError

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """
        Convenience wrapper for a single user prompt with an optional system
        instruction.
        """
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages)
