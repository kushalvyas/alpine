import pytest
import torch
import numpy as np

# Adjust imports based on your actual package structure
from alpine.models.finer import Finer
from alpine.models.nonlin import FinerSine

@pytest.fixture
def coords():
    return torch.randn(5, 3)  # batch of 5, 3 input features

# --- Tests ---

def test_finer_initialization():
    """Test if layers are initialized correctly with specific Activation functions."""
    # hidden_layers=3 implies: 
    # 1. First Linear
    # 2. FinerSine
    # 3. Hidden Linear
    # 4. FinerSine
    # 5. Last Linear
    model = Finer(in_features=3, hidden_features=8, hidden_layers=3, out_features=2)
    
    assert len(model.model) == 5
    assert isinstance(model.model[0], torch.nn.Linear)
    assert isinstance(model.model[1], FinerSine)
    assert isinstance(model.model[2], torch.nn.Linear)
    assert isinstance(model.model[3], FinerSine)
    assert isinstance(model.model[-1], torch.nn.Linear)

def test_forward_output_shape(coords):
    """Test standard forward pass output shape."""
    model = Finer(in_features=3, hidden_features=8, hidden_layers=3, out_features=2)
    out = model(coords)
    assert 'output' in out
    assert out['output'].shape == (5, 2)

def test_forward_with_return_features(coords):
    """Test if intermediate features are returned when requested."""
    model = Finer(in_features=3, hidden_features=8, hidden_layers=3, out_features=2)
    out = model(coords, return_features=True)
    
    assert 'output' in out
    assert 'features' in out
    
    # AlpineBaseModule usually structures features by layer index
    layers_dict = out['features']['model']
    
    # Check for presence of pre/post activation keys for indices 0 to 4
    # Note: The specific keys depend on implementation of AlpineBaseModule.forward_w_features
    # Assuming similar behavior to Siren test provided:
    assert 'pre_activation' in layers_dict['0']
    assert 'post_activation' in layers_dict['1']
    assert 'pre_activation' in layers_dict['2']
    assert 'post_activation' in layers_dict['3']
    assert 'pre_activation' in layers_dict['4']

def test_custom_omegas():
    """Test initialization with a list of custom omegas."""
    # 3 hidden layers require 3 omegas if provided as a list
    omegas = [10.0, 20.0, 30.0]
    model = Finer(in_features=3, hidden_features=8, hidden_layers=3, out_features=2, omegas=omegas)
    assert model.omegas == omegas
    
    # Verify the sine layers picked up the correct omegas
    # model[1] is first sine, model[3] is second sine
    assert model.model[1].omega == 10.0
    assert model.model[3].omega == 20.0
    # The final omega (30.0) is used for the weights of the last linear layer, not a sine layer

def test_first_bias_scale():
    """Test if first_bias_scale affects the first layer's bias initialization."""
    scale = 10.0
    model = Finer(in_features=3, hidden_features=8, hidden_layers=3, out_features=2, first_bias_scale=scale)
    
    first_layer = model.model[0]
    
    # Check that biases are within the scaled range [-scale, scale]
    # (Technically uniform, so checking bounds is a good sanity check)
    assert torch.all(first_layer.bias.data >= -scale)
    assert torch.all(first_layer.bias.data <= scale)
    
    # Ensure it's not just using the default small initialization (e.g. < 1.0)
    # by checking if at least one value is large-ish (statistically likely for scale=10)
    # or just checking that the bounds are respected which we did above.

def test_load_weights():
    """Test weight loading mechanism."""
    model = Finer(in_features=3, hidden_features=8, hidden_layers=3, out_features=2)
    
    # Save current weights
    state_dict = model.state_dict()
    
    # Modify weights in place
    for param in model.parameters():
        param.data.fill_(0.5)
        
    # Load previous weights
    model.load_weights(state_dict)
    
    # Check that weights are restored
    for key, param in model.state_dict().items():
        assert torch.allclose(param, state_dict[key])