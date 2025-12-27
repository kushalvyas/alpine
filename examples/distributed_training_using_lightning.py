"""Code for using DDP strategy for training a large INR on large data using Alpine"""

import os

gpu_ids = "2,3,5"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

# %%
# make all necessary imports
import alpine
import skimage.io, skimage, data, skimage.transform, skimage
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from functools import partial
import lightning as pl
import torch, torch.nn as nn

# %%
NUM_EPOCHS = 10

# %%
siren_model = alpine.models.Siren(
    in_features=2,
    out_features=3,
    hidden_features=1024,
    hidden_layers=5,
    outermost_linear=True,
).float()
siren_model.compile(
    scheduler=partial(
        optim.lr_scheduler.LambdaLR,
        lr_lambda=lambda epoch: 0.95 ** min(epoch / NUM_EPOCHS, 1),
    ),
)
print(siren_model)

# %%


# %%
H, W = 4096, 4096

# %%
img = skimage.data.astronaut()
img = skimage.transform.resize(img, (H, W), anti_aliasing=True)
img = skimage.img_as_float(img)
gt_signal = torch.from_numpy(img).float()
print(gt_signal.shape)

# %%
train_dataset = alpine.dataloaders.BatchedNDSignalLoader(
    signal=gt_signal,
    grid_dims=[H, W],
    bounds=(-1, 1),
    vectorized=True,
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=100 * 100,
    num_workers=16,
    pin_memory=True,
    shuffle=True,
)

test_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=100 * 100,
    num_workers=16,
    pin_memory=True,
    shuffle=False,
)

# %%
siren_lightning = alpine.trainers.LightningTrainer(
    model=siren_model, return_features=False, log_results=False
)

# %%
trainer = pl.Trainer(
    accelerator="cuda",
    devices=len(gpu_ids.split(",")),
    max_epochs=NUM_EPOCHS,
    strategy="ddp",
)

# %%
trainer.fit(siren_lightning, train_dataloaders=train_dataloader)

trainer2 = pl.Trainer(accelerator="cuda", devices=1, max_epochs=1)
trainer2.test(siren_lightning, dataloaders=test_dataloader)

# %%
import skimage.io

skimage.io.imsave(
    "siren_lightning.png",
    np.uint8(255 * siren_lightning.stacked_test_output.reshape(H, W, 3)),
)
