from types import SimpleNamespace

import pytest
import torch
from torch import nn

from semantic_3d_chat.language.local_lm import LocalLanguageModel
from semantic_3d_chat.language.prefix_injection import (
    ContinuousPrefixComposer,
    prefix_sha256,
    scene_prefix_after_bos_contract_mismatch,
    scene_prefix_after_bos_setting,
    stack_prefix_batches,
)


def test_prefix_precedes_prompt_and_loss_only_covers_answer() -> None:
    torch.manual_seed(3)
    embedding = nn.Embedding(20, 12)
    composer = ContinuousPrefixComposer(12)
    scene = torch.randn(1, 8, 12)
    prompt = torch.tensor([[2, 3, 4]])
    answer = torch.tensor([[5, 6]])
    batch = composer.compose(scene, prompt, embedding, answer)
    assert batch.inputs_embeds.shape == (1, 15, 12)
    assert batch.scene_prefix_length == 10
    assert torch.all(batch.labels[:, :13] == -100)
    assert torch.equal(batch.labels[:, 13:], answer)


def test_generic_control_tokens_follow_complete_prompt_and_preserve_scene_identity() -> None:
    torch.manual_seed(30)
    embedding = nn.Embedding(20, 4)
    composer = ContinuousPrefixComposer(4)
    scene = torch.randn(1, 2, 4)
    prompt = torch.tensor([[2, 3, 4]])
    answer = torch.tensor([[5, 6]])
    first_control = torch.randn(1, 2, 4)
    second_control = torch.randn(1, 3, 4)
    scene_hash = prefix_sha256(composer.scene_prefix(scene))

    trained = composer.compose(
        scene,
        prompt,
        embedding,
        answer,
        control_tokens=first_control,
    )
    generated = composer.compose(
        scene,
        prompt,
        embedding,
        control_tokens=second_control,
    )
    scene_prefix = composer.scene_prefix(scene)

    assert trained.scene_prefix_length == generated.scene_prefix_length == 4
    assert prefix_sha256(scene_prefix) == scene_hash
    assert torch.equal(trained.inputs_embeds[:, :4], scene_prefix)
    assert torch.equal(trained.inputs_embeds[:, 4:7], embedding(prompt))
    assert torch.equal(trained.inputs_embeds[:, 7:9], first_control)
    assert torch.equal(trained.inputs_embeds[:, 9:], embedding(answer))
    assert torch.equal(trained.labels[:, :9], torch.full((1, 9), -100))
    assert torch.equal(trained.labels[:, 9:], answer)
    assert generated.labels is None
    assert torch.equal(generated.inputs_embeds[:, :4], scene_prefix)
    assert torch.equal(generated.inputs_embeds[:, 4:7], embedding(prompt))
    assert torch.equal(generated.inputs_embeds[:, 7:], second_control)


@pytest.mark.parametrize(
    ("control_tokens", "message"),
    [
        (torch.randn(1, 4), "shape"),
        (torch.randn(2, 1, 4), "batch sizes"),
        (torch.randn(1, 1, 5), "hidden size"),
        (torch.tensor([[[float("nan"), 0.0, 0.0, 0.0]]]), "finite"),
    ],
)
def test_generic_control_tokens_reject_invalid_inputs(
    control_tokens: torch.Tensor,
    message: str,
) -> None:
    composer = ContinuousPrefixComposer(4)
    with pytest.raises(ValueError, match=message):
        composer.compose(
            torch.randn(1, 2, 4),
            torch.tensor([[2, 3]]),
            nn.Embedding(20, 4),
            control_tokens=control_tokens,
        )


def test_opt_in_generic_layout_keeps_native_bos_before_continuous_scene_prefix() -> None:
    torch.manual_seed(31)
    embedding = nn.Embedding(20, 4)
    composer = ContinuousPrefixComposer(
        4,
        scene_prefix_after_bos=True,
        bos_token_id=2,
    )
    scene = torch.randn(1, 2, 4)
    prompt = torch.tensor([[2, 3, 4]])
    answer = torch.tensor([[5, 6]])
    scene_prefix = composer.scene_prefix(scene)

    batch = composer.compose(scene, prompt, embedding, answer)

    assert batch.inputs_embeds.shape == (1, 9, 4)
    assert batch.scene_prefix_length == 4
    assert torch.equal(batch.inputs_embeds[:, :1], embedding(prompt[:, :1]))
    assert torch.equal(batch.inputs_embeds[:, 1:5], scene_prefix)
    assert torch.equal(batch.inputs_embeds[:, 5:7], embedding(prompt[:, 1:]))
    assert torch.equal(batch.inputs_embeds[:, 7:], embedding(answer))
    assert torch.equal(batch.labels[:, :7], torch.full((1, 7), -100))
    assert torch.equal(batch.labels[:, 7:], answer)


def test_opt_in_generic_layout_rejects_missing_or_misplaced_bos() -> None:
    embedding = nn.Embedding(20, 4)
    scene = torch.randn(1, 2, 4)
    missing_identity = ContinuousPrefixComposer(4, scene_prefix_after_bos=True)
    with pytest.raises(ValueError, match="BOS token ID"):
        missing_identity.compose(scene, torch.tensor([[2, 3]]), embedding)

    composer = ContinuousPrefixComposer(
        4,
        scene_prefix_after_bos=True,
        bos_token_id=2,
    )
    with pytest.raises(ValueError, match="start with bos_token_id=2"):
        composer.compose(scene, torch.tensor([[3, 2]]), embedding)


def test_bos_prefix_config_and_legacy_checkpoint_contract_are_strict() -> None:
    assert not scene_prefix_after_bos_setting({"language": {}})
    assert scene_prefix_after_bos_setting(
        {"language": {"scene_prefix_after_bos": True}}
    )
    with pytest.raises(TypeError, match="must be a boolean"):
        scene_prefix_after_bos_setting(
            {"language": {"scene_prefix_after_bos": "true"}}
        )

    assert scene_prefix_after_bos_contract_mismatch({}, False) is None
    assert scene_prefix_after_bos_contract_mismatch(
        {"scene_prefix_after_bos": False}, False
    ) is None
    assert scene_prefix_after_bos_contract_mismatch({}, True) == {
        "checkpoint": "<missing; legacy false>",
        "runtime": True,
    }
    assert scene_prefix_after_bos_contract_mismatch(
        {"scene_prefix_after_bos": True}, False
    ) == {"checkpoint": True, "runtime": False}


def test_generic_generation_uses_same_bos_first_layout_as_training() -> None:
    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(20, 4)
            self.config = SimpleNamespace(hidden_size=4, bos_token_id=2)

        def get_input_embeddings(self):
            return self.embedding

    model = TinyModel()
    language = LocalLanguageModel(
        model=model,
        tokenizer=SimpleNamespace(bos_token_id=2),
        device=torch.device("cpu"),
    )
    scene_prefix = torch.randn(1, 4, 4)
    prompt_ids = torch.tensor([[2, 3, 4]])
    captured: dict[str, torch.Tensor] = {}

    def fallback(_model, inputs, attention, _maximum, _eos):
        captured["inputs"] = inputs.detach().clone()
        captured["attention"] = attention.detach().clone()
        return torch.tensor([[7]])

    generated = language.generate_from_scene_prefix(
        scene_prefix,
        prompt_ids,
        max_new_tokens=1,
        eos_token_ids=None,
        scene_prefix_after_bos=True,
        fallback=fallback,
    )

    prompt_embeddings = model.get_input_embeddings()(prompt_ids)
    assert torch.equal(generated, torch.tensor([[7]]))
    assert torch.equal(captured["inputs"][:, :1], prompt_embeddings[:, :1])
    assert torch.equal(captured["inputs"][:, 1:5], scene_prefix)
    assert torch.equal(captured["inputs"][:, 5:], prompt_embeddings[:, 1:])
    assert torch.equal(captured["attention"], torch.ones(1, 7, dtype=torch.long))


def test_prefix_hash_is_question_independent() -> None:
    torch.manual_seed(4)
    composer = ContinuousPrefixComposer(8)
    scene = torch.randn(1, 5, 8)
    before = prefix_sha256(composer.scene_prefix(scene))
    _unrelated_questions = ["Is there a chair?", "Where is the bowl?"]
    after = prefix_sha256(composer.scene_prefix(scene))
    assert before == after


def test_generic_variable_length_stack_preserves_qwen_path() -> None:
    torch.manual_seed(5)
    embedding = nn.Embedding(20, 8)
    composer = ContinuousPrefixComposer(8)
    scene = torch.randn(1, 3, 8, requires_grad=True)
    short = composer.compose(
        scene,
        torch.tensor([[2]]),
        embedding,
        torch.tensor([[5, 6]]),
    )
    long = composer.compose(
        scene,
        torch.tensor([[2, 3, 4]]),
        embedding,
        torch.tensor([[7, 8, 9]]),
    )
    stacked = stack_prefix_batches([short, long], torch.device("cpu"))

    assert stacked.inputs_embeds.shape == (2, 11, 8)
    assert stacked.per_layer_inputs is None
    assert stacked.mm_token_type_ids is None
    assert torch.equal(stacked.attention_mask[0, 8:], torch.zeros(3, dtype=torch.long))
    assert torch.equal(stacked.labels[0, 8:], torch.full((3,), -100))
    assert torch.equal(stacked.labels[0, 6:8], torch.tensor([5, 6]))
    stacked.inputs_embeds.sum().backward()
    assert scene.grad is not None and scene.grad.abs().sum() > 0
