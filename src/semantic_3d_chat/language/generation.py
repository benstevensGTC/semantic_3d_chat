from __future__ import annotations

from collections.abc import Sequence

import torch


@torch.inference_mode()
def generate_from_embeddings(
    model: torch.nn.Module,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    eos_token_ids: int | Sequence[int] | None,
) -> torch.Tensor:
    """Deterministic greedy decoding that preserves an `inputs_embeds` prefix in KV cache."""
    if inputs_embeds.ndim != 3 or inputs_embeds.shape[:2] != attention_mask.shape:
        raise ValueError("inputs_embeds [B,T,H] and attention_mask [B,T] must align")
    if inputs_embeds.shape[0] != 1:
        raise ValueError("The interactive generation path currently supports batch size one")
    stop_ids = set()
    if eos_token_ids is not None:
        stop_ids = {int(eos_token_ids)} if isinstance(eos_token_ids, int) else set(eos_token_ids)

    outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=True)
    next_id = outputs.logits[:, -1].float().argmax(dim=-1, keepdim=True)
    generated = [next_id]
    past = outputs.past_key_values
    if int(next_id.item()) in stop_ids:
        return next_id

    for _ in range(max_new_tokens - 1):
        attention_mask = torch.cat(
            (attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)),
            dim=1,
        )
        outputs = model(
            input_ids=next_id,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
        )
        past = outputs.past_key_values
        next_id = outputs.logits[:, -1].float().argmax(dim=-1, keepdim=True)
        generated.append(next_id)
        if int(next_id.item()) in stop_ids:
            break
    return torch.cat(generated, dim=1)
