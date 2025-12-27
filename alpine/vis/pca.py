import torch
from sklearn.decomposition import PCA
import numpy as np


def normalize_fn(x, axis=None):
    if axis is None:
        return (x - x.min()) / (x.max() - x.min())
    else:
        return (x - x.min(axis=axis, keepdims=True)) / (
            x.max(axis=axis, keepdims=True) - x.min(axis=axis, keepdims=True)
        )


def check_tensor_dtype(x):
    if isinstance(x, torch.Tensor):
        return x.clone().detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise ValueError("Input must be a torch.Tensor or numpy.ndarray")


def return_as_torch(x):
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    else:
        raise ValueError("Input must be a torch.Tensor or numpy.ndarray")


def compute_pca_layerwise(
    inr_features,
    num_components,
    signal_shape,
    pca_preprocess_whiten=True,
    min_max_normalize=True,
    normalization_axis=(-1,),
    **kwargs,
):
    batch = inr_features.shape[0]
    assert batch == 1, "Batch size must be 1"
    pts = np.prod(inr_features.shape[1:-1])
    feature_dims = inr_features.shape[-1]
    pca = PCA(n_components=num_components, whiten=pca_preprocess_whiten, **kwargs)

    inr_features = inr_features.reshape(batch * pts, feature_dims)
    pca_features = pca.fit_transform(inr_features)
    pca_features = pca_features.reshape(batch, pts, num_components)
    pca_features_signal = pca_features.reshape(
        (batch,) + signal_shape + (num_components,)
    )

    if min_max_normalize:
        pca_features_signal = normalize_fn(pca_features_signal, axis=normalization_axis)

    return return_as_torch(pca_features_signal)


def compute_pca_features(
    inr_features,
    num_components,
    signal_shape,
    pca_preprocess_whiten=True,
    min_max_normalize=True,
    normalization_axis=(-1,),
    **kwargs,
):
    """_summary_

    Args:
        inr_features (_type_): _description_
        num_components (_type_): _description_
        signal_shape (_type_): _description_
        pca_preprocess_whiten (bool, optional): _description_. Defaults to True.
        min_max_normalize (bool, optional): _description_. Defaults to True.
        kwargs: keyword arguments for sklearn.decomposition.PCA
    """

    assert (
        len(inr_features.shape) > 3
    ), "Input features must have shape (1, layers, pts, features)"
    batch = inr_features.shape[0]
    assert batch == 1, "Batch size must be 1"
    num_layers = inr_features.shape[1]
    pts = inr_features.shape[2]
    feature_dims = inr_features.shape[-1]

    inr_features = check_tensor_dtype(inr_features)

    pca_features_signal = torch.stack(
        [
            compute_pca_layerwise(
                inr_features[:, l, ...],
                num_components=num_components,
                signal_shape=signal_shape,
                pca_preprocess_whiten=pca_preprocess_whiten,
                min_max_normalize=min_max_normalize,
                normalization_axis=normalization_axis,
                **kwargs,
            )
            for l in range(num_layers)
        ]
    ).squeeze(1)
    return return_as_torch(pca_features_signal)


def compute_pca_features_batch(
    inr_features,
    num_components,
    signal_shape,
    pca_preprocess_whiten=True,
    min_max_normalize=True,
    normalization_axis=(-1,),
    **kwargs,
):
    """_summary_

    Args:
        inr_features (_type_): _description_
        num_components (_type_): _description_
        signal_shape (_type_): _description_
        pca_preprocess_whiten (bool, optional): _description_. Defaults to True.
        min_max_normalize (bool, optional): _description_. Defaults to True.
        kwargs: keyword arguments for sklearn.decomposition.PCA
    """

    assert (
        len(inr_features.shape) > 2
    ), "Input features must have shape (1, pts, features)"
    batch = inr_features.shape[0]
    pts = inr_features.shape[1]
    feature_dims = inr_features.shape[-1]

    pca_features = torch.stack(
        [
            compute_pca_features(
                inr_features[b].unsqueeze(0),
                num_components=num_components,
                signal_shape=signal_shape,
                pca_preprocess_whiten=pca_preprocess_whiten,
                min_max_normalize=min_max_normalize,
                normalization_axis=normalization_axis,
                **kwargs,
            )
            for b in range(batch)
        ]
    ).squeeze(1)

    return pca_features
