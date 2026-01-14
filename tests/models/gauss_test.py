import pytest
import torch
import numpy as np

from alpine.models.gauss import Gauss
from alpine.models.nonlin import Gauss as GaussActivation

@pytest.fixture
def coords():
    return torch.randn(5, 3)  # batch of 5, 3 input features

def test_gauss_initialization():
    model = Gauss(in_features=3, hidden_features=8, hidden_layers=3, out_features=2)
    # There should be 3 linear + 2 sine layers before last linear + possible sine layer if outermost_linear False
    assert isinstance(model.model[0], torch.nn.Linear)
    assert isinstance(model.model[1], GaussActivation)
    assert isinstance(model.model[-1], torch.nn.Linear)
    
def test_forward_with_return_features(coords):
    model = Gauss(in_features=3, hidden_features=8, hidden_layers=3, out_features=2)
    model.compile()
    
    out = model(coords, return_features=True)
    assert "output" in out
    assert "features" in out

    layers_dict = out["features"]["model"]
    assert "pre_activation" in layers_dict["0"]
    assert "post_activation" in layers_dict["1"]
    assert "pre_activation" in layers_dict["2"]
    assert "post_activation" in layers_dict["3"]
    assert "pre_activation" in layers_dict["4"]
    
def test_load_weights():
    model = Gauss(in_features=3, hidden_features=8, hidden_layers=3, out_features=2)
    # Save current weights
    state_dict = model.state_dict()
    # Modify weights
    for param in model.parameters():
        param.data.fill_(0.5)
    # Load previous weights
    model.load_weights(state_dict)
    # Check that weights restored
    for key, param in model.state_dict().items():
        assert torch.allclose(param, state_dict[key])