import pytest
import torch

from alpine.models.fourier_reparametrized import FourierReparameterization, Sine, ReLU, FourierLinear


def make_model(nonlinearity, **kwargs):
    base = dict(
        in_features=3,
        hidden_features=8,
        hidden_layers=3,
        out_features=2,
        num_frequencies=4,
        num_phases=6,
    )
    return FourierReparameterization(nonlinearity=nonlinearity, **base, **kwargs)


@pytest.fixture
def coords():
    return torch.randn(5, 3) # batch of 5, 3 input features


@pytest.fixture
def relu_model():
    return make_model("relu")


@pytest.fixture
def sine_model():
    return make_model("sine", omegas=[30.0])


# --- Tests ---


def test_relu_initialization(relu_model):
    layers = [layer for layer in relu_model.model]
    # The first layer is a normal Linear layer, followed by a ReLU then FourierLinear
    assert isinstance(layers[0], torch.nn.Linear)
    assert isinstance(layers[1], ReLU)
    assert isinstance(layers[2], FourierLinear)
    assert isinstance(layers[-1], torch.nn.Linear)


def test_sine_initialization(sine_model):
    layers = [layer for layer in sine_model.model]
    # The first layer is a normal Linear layer, followed by a Sine then FourierLinear
    assert isinstance(layers[0], torch.nn.Linear)
    assert isinstance(layers[1], Sine)
    assert isinstance(layers[2], FourierLinear)
    assert isinstance(layers[-1], torch.nn.Linear)


def test_omega_assertion():
    # Empty omegas only valid for ReLU
    with pytest.raises(AssertionError, match='Omegas must be specified for non-ReLU nonlinearities'):
        make_model('sine')


def test_nonlin_assertion():
    with pytest.raises(ValueError, match='Unknown nonlinearity: foo'):
        make_model('foo')

@pytest.mark.parametrize("model_fixture", ["relu_model", "sine_model"])
def test_forward_output_shape(request, model_fixture, coords):
    model = request.getfixturevalue(model_fixture)
    out = model(coords)
    assert "output" in out
    assert out["output"].shape == (5, 2)

@pytest.mark.parametrize("model_fixture", ["relu_model", "sine_model"])
def test_forward_with_return_features(request, model_fixture, coords):

    model = request.getfixturevalue(model_fixture)
    out = model(coords, return_features=True)
    assert "output" in out
    assert "features" in out

    layers_dict = out["features"]["model"]
    assert "pre_activation" in layers_dict["0"]
    assert "post_activation" in layers_dict["1"]
    assert "pre_activation" in layers_dict["2"]
    assert "post_activation" in layers_dict["3"]
    assert "pre_activation" in layers_dict["4"]


def test_outermost_linear_toggle():
    model = make_model(nonlinearity='sine', outermost_linear=False, omegas=[30.0])
    assert isinstance(model.model[-1], Sine)  # last layer is Sine now


def test_custom_omegas():
    omegas = [10.0, 20.0, 30.0]
    model = make_model(nonlinearity='sine', omegas=omegas, outermost_linear=False)
    # (Linear[None], Sine[10.0], FourierLinear[20.0], Sine[20.0], Linear[None], Sine[30.0])
    expected_pattern = [10.0, 20.0, 20.0, 30.0]
    omegas_in_model = [layer.omega for layer in model.model if hasattr(layer, "omega")]
    # Check to make sure omega is propagating correctly through the sine layers.
    assert model.omegas == omegas
    assert omegas_in_model == expected_pattern


@pytest.mark.parametrize("model_fixture", ["relu_model", "sine_model"])
def test_load_weights(request, model_fixture):
    model = request.getfixturevalue(model_fixture)
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



