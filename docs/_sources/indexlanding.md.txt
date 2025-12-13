Alpine is a flexible, user-friendly, and distributed PyTorch library for Implicit Neural Representations. Alpine provides a systematic way to build, train, and visualize neural fields or neural representations across a wide variety of scientific and medical data. Alpine is user-frieldly, scalable, and easily integrates with exiting PyTorch workflows. 

### Getting started

```{code-block} python
import alpine
import torch
import skimage.io, skimage.transform, skimage.data

# load data signal
H, W = 256, 256
data = skimage.transform.resize(skimage.data.astronaut(), (H,W))

# create a pytorch tensor for data signal
data_tensor = torch.from_numpy(data).float().cuda()

# initialize coordinate grid for fitting the signal 
coords = alpine.utils.coords_spatial_nd(H,W)[None,...].cuda()

# initialize neural representation
model = alpine.models.Siren(in_features=2, out_features=3).cuda()

# fit signal
result = model.fit_signal(input = coords, signal = data_tensor, n_iters=1000)
```

For more instructions on how to install `alpine` please see our [installation instructions](installation.md). Alpine also provides a wide variety of [examples](examples.md) detailing various features and functionalities of `alpine`.


### Key Features

::::{grid} 4
:::{grid-item-card}  Flexible
Alpine provides modular interfaces for prototyping and testing neural representations. Alpine also offers highly reconfigurable training workflows that allow for custom forward model via closures and loss functions.
:::
:::{grid-item-card}  Distributed
We use PyTorch lightning to scale Alpine across multiple GPUs. Alpine offers seamless integration with Pytorch lightning using its custom lightining trainer.
:::
:::{grid-item-card}  Interpretability
Alpine provides a rich set of visualization tools ( and many more to be added) that allow you to peer deep into the features learned by the INR.
:::
:::{grid-item-card}  Extensible
Written in PyTorch, Alpine can be seamlessly integrated with other PyTorch libraries and models. 
:::
::::


:::{important}
Alpine is work in progress. We appreciate any constructive community feedback and support. We invite all the researchers across all disciplines to explore, and suggest any features you find particularly useful. Our goal is to make `Alpine` the go-to place for scientific computing using INRs!
:::
