import alpine
import pytest
import torch

@pytest.fixture
def coords():
    return torch.randn(5, 2)

def test_forward_with_return_features(coords):
    ffn_model = alpine.models.FFN(
        in_features=2,
        out_features=3,
        hidden_features=256,
        hidden_layers=4,
        positional_encoding='fourier'
    )
    
    out = ffn_model(coords, return_features=True)
    assert 'output' in out
    assert 'features' in out
    
    model_features = out['features']['model']
    assert len(model_features) == 7  # PosEncoding, 3 Linear+ReLU pairs, and final Linear layer
    
    assert 'pre_activation' in model_features['1']
    assert 'post_activation' in model_features['2']
    assert 'pre_activation' in model_features['3']
    assert 'post_activation' in model_features['4']
    assert 'pre_activation' in model_features['5']
    assert 'post_activation' in model_features['6']
    assert 'pre_activation' in model_features['7']