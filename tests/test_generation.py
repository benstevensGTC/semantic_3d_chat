from types import SimpleNamespace

import torch
from torch import nn

from semantic_3d_chat.language.generation import generate_from_embeddings


class DeterministicToyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, inputs_embeds=None, input_ids=None, **kwargs):
        self.calls += 1
        token = min(self.calls, 3)
        sequence = inputs_embeds.shape[1] if inputs_embeds is not None else input_ids.shape[1]
        logits = torch.full((1, sequence, 5), -10.0)
        logits[:, -1, token] = 10.0
        return SimpleNamespace(logits=logits, past_key_values=(self.calls,))


def test_custom_generation_uses_prefix_once_and_stops() -> None:
    model = DeterministicToyLM()
    result = generate_from_embeddings(
        model,
        inputs_embeds=torch.zeros(1, 7, 4),
        attention_mask=torch.ones(1, 7, dtype=torch.long),
        max_new_tokens=5,
        eos_token_ids=3,
    )
    assert result.tolist() == [[1, 2, 3]]
    assert model.calls == 3
