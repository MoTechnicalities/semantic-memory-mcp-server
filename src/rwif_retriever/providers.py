from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ActivationProvider(Protocol):
    def encode_text(self, text: str) -> np.ndarray: ...

    def encode_texts(self, texts: list[str]) -> np.ndarray: ...


@dataclass
class ArrayActivationProvider:
    activations: dict[str, np.ndarray]

    def encode_text(self, text: str) -> np.ndarray:
        if text not in self.activations:
            raise KeyError(f"No activation registered for text: {text}")
        return np.asarray(self.activations[text], dtype=np.float64)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        vectors = [self.encode_text(text) for text in texts]
        if not vectors:
            raise ValueError("texts must not be empty")
        return np.stack(vectors, axis=0)


@dataclass
class TransformersActivationProvider:
    model_id: str
    layer_index: int = -1
    pooling: str = "last_token"
    device: str | None = None
    max_length: int = 256

    def __post_init__(self) -> None:
        self._tokenizer = None
        self._model = None
        self._torch = None

    def encode_text(self, text: str) -> np.ndarray:
        return self.encode_texts([text])[0]

    def warmup(self, sample_text: str = "semantic memory warmup") -> None:
        self.encode_text(sample_text)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise ValueError("texts must not be empty")
        tokenizer, model, torch = self._ensure_loaded()
        encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        encoded = {name: tensor.to(self._resolved_device()) for name, tensor in encoded.items()}
        with torch.inference_mode():
            outputs = model(**encoded, output_hidden_states=True)
        hidden = outputs.hidden_states[self.layer_index]
        attention_mask = encoded["attention_mask"]
        if self.pooling == "last_token":
            lengths = attention_mask.sum(dim=1) - 1
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            pooled = hidden[rows, lengths]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            raise ValueError(f"Unsupported pooling strategy: {self.pooling}")
        return pooled.detach().cpu().numpy().astype(np.float64)

    def _ensure_loaded(self):
        if self._tokenizer is not None and self._model is not None and self._torch is not None:
            return self._tokenizer, self._model, self._torch

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "TransformersActivationProvider requires both transformers and torch to be installed."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        model = AutoModel.from_pretrained(self.model_id)
        model.to(self._resolved_device(torch))
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch
        return tokenizer, model, torch

    def _resolved_device(self, torch_module=None) -> str:
        if self.device:
            return self.device
        torch = torch_module or self._torch
        if torch is None:
            import torch as imported_torch

            torch = imported_torch
        return "cuda" if torch.cuda.is_available() else "cpu"