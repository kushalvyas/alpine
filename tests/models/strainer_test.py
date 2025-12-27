import pytest
import torch
import numpy as np

from alpine.models.strainer import Strainer  # adjust import path if needed


@pytest.fixture
def coords():
    return torch.randn(5, 2)  # batch of 5, 2 input features


def test_forward_with_return_features(coords):
    num_decoders = 10
    num_shared_layers = 5
    model = Strainer(
        in_features=2,
        out_features=3,
        hidden_features=178,
        hidden_layers=6,
        num_decoders=num_decoders,
        num_shared_layers=num_shared_layers,
        outermost_linear=True,
    )
    out = model(coords, return_features=True)
    assert "output" in out
    assert "features" in out

    encoder_features = out["features"]["encoder"]
    decoders_features = out["features"]["decoder"]

    assert len(encoder_features) == 10  # 5 layers with pre and post activations
    assert len(decoders_features) == num_decoders
