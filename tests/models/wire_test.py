import alpine
import pytest
import torch

@pytest.fixture
def coords():
    return torch.randn(5, 2)

def test_forward_with_return_features(coords):
    wire_model = alpine.models.Wire(
        in_features=2,
        out_features=1,
        hidden_features=300,
        hidden_layers=4,
        omegas=[10.0],
        sigmas=[10.0,],
    )
    
    out = wire_model(coords, return_features=True)
    assert 'output' in out 
    assert 'features' in out
    
    model_features = out['features']['model']
    assert len(model_features) == 7 #3 Linear+Wavelet pairs and final Linear layer
    assert 'pre_activation' in model_features['0']
    assert 'post_activation' in model_features['1']
    assert 'pre_activation' in model_features['2']
    assert 'post_activation' in model_features['3']
    assert 'pre_activation' in model_features['4']
    assert 'post_activation' in model_features['5']
    assert 'pre_activation' in model_features['6']
    
    