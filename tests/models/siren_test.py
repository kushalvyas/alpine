import pytest
import torch
import numpy as np

from alpine.models.siren import Siren
from alpine.models.nonlin import Sine

@pytest.fixture
def coords():
    return torch.randn(5, 3)  # batch of 5, 3 input features

# --- Tests ---
def test_siren_initialization():
    model = Siren(in_features=3, hidden_features=8, hidden_layers=3, out_features=2)
    # There should be 3 linear + 2 sine layers before last linear + possible sine layer if outermost_linear False
    assert isinstance(model.model[0], torch.nn.Linear)
    assert isinstance(model.model[1], Sine)
    assert isinstance(model.model[-1], torch.nn.Linear)

def test_forward_output_shape(coords):
    model = Siren(3, 8, 3, 2)
    out = model(coords)
    assert 'output' in out
    assert out['output'].shape == (5, 2)
        

def test_forward_with_return_features(coords):
    model = Siren(3, 8, 3, 2)
    out = model(coords, return_features=True)
    assert 'output' in out
    assert 'features' in out
    
    # Features should have linear, sine, linear, sine, linear
    # Check that we have the expected activations
    assert 'linear_0_pre' in out['features']  
    assert out['features']['linear_0_pre'].shape == (5, 3)
    assert 'sine_0_post' in out['features']   
    assert out['features']['sine_0_post'].shape == (5, 8)
    assert 'linear_1_pre' in out['features']
    assert out['features']['linear_1_pre'].shape == (5, 8)
    assert 'sine_1_post' in out['features']
    assert out['features']['sine_1_post'].shape == (5, 8)
    assert 'linear_2_pre' in out['features']
    assert out['features']['linear_2_pre'].shape == (5, 8)
    
    
def test_outermost_linear_toggle():
    model = Siren(3, 8, 3, 2, outermost_linear=False)
    assert isinstance(model.model[-1], Sine)  # last layer is Sine now

def test_custom_omegas():
    omegas = [10.0, 20.0, 30.0]
    model = Siren(3, 8, 3, 2, omegas=omegas)
    assert model.omegas == omegas

def test_load_weights():
    model = Siren(3, 8, 3, 2)
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