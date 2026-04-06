from __future__ import annotations

import torch

from classifer_training.extract_hidden_states import _extract_last_token_vector, _extract_messages


def test_extract_last_token_vector_handles_standard_hidden_state_shapes() -> None:
    tensor3d = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    extracted3d = _extract_last_token_vector(tensor3d)
    assert torch.equal(extracted3d, torch.tensor([3.0, 4.0]))
    assert extracted3d._base is None

    tensor2d = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    extracted2d = _extract_last_token_vector(tensor2d)
    assert torch.equal(extracted2d, torch.tensor([7.0, 8.0]))
    assert extracted2d._base is None

    tensor1d = torch.tensor([9.0, 10.0])
    extracted1d = _extract_last_token_vector(tensor1d)
    assert torch.equal(extracted1d, torch.tensor([9.0, 10.0]))
    assert extracted1d._base is None


def test_extract_messages_uses_messages_or_user_input() -> None:
    record_with_messages = {
        "messages": [{"role": "user", "content": "hello"}],
    }
    assert _extract_messages(record_with_messages) == [{"role": "user", "content": "hello"}]

    record_with_user_input = {
        "user_input": "fallback prompt",
    }
    assert _extract_messages(record_with_user_input) == [{"role": "user", "content": "fallback prompt"}]
