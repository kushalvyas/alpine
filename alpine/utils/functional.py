import torch
from collections import OrderedDict


def functional_model_call(model, params, coords):
    return torch.func.functional_call(model, params, coords)


def get_stacked_weights_and_bias_from_statedicts(params, N):
    #  from a list of model state dicts, get stacked k, v pirs of w and bias
    stacked_params = OrderedDict(
        (key, torch.stack([params[i][key] for i in range(N)]))
        for key in params[0].keys()
    )
    return stacked_params
