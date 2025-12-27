"""Code for using DDP strategy for training a large INR on large data using Alpine"""

import os

gpu_ids = "3,4"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

# %%
# make all necessary imports
import alpine
import numpy as np
from functools import partial
import lightning as pl
import torch
from scipy import io
from scipy import ndimage
from alpine.utils import volutils

# %%
import argparse

parser = argparse.ArgumentParser(description="Distributed training of INR")
parser.add_argument(
    "--num_epochs", type=int, default=20, help="Number of epochs to train"
)
parser.add_argument(
    "--batch_size", type=float, default=(1e6), help="Number of epochs to train"
)
args = parser.parse_args()

# %%
NUM_EPOCHS = int(args.num_epochs)
batch_size = int(args.batch_size)
mcubes_thresh = 0.5
scale = 1.0
savename_original = "./data/occupancy/output/original_thai_statue_3d_distributed.dae"
savename = "./data/occupancy/output/thai_statue_3d_distributed4_lrdecay_v3_lrdecay095_10000iter.dae"

# %%
# epochs 20, bs 1e6, lr maybe 1e-4 not 5e-5 : loss 0.00301, outputfile: 2.dae
#  epochs 50, bs 1e6, lr 5e-5 :

# %%
wire_model = alpine.models.Siren(
    in_features=3,
    out_features=1,
    hidden_features=512,
    hidden_layers=5,
    # omegas=[10.0],
    # sigmas = [40.0],
    outermost_linear=True,
).float()
wire_model.compile(
    learning_rate=1e-4,
    scheduler=partial(
        torch.optim.lr_scheduler.LambdaLR,
        lr_lambda=lambda epoch: 0.95 ** (min(epoch / NUM_EPOCHS, 1)),
    ),
)
print(wire_model)

# %%
im = io.loadmat("./data/occupancy/thai_statue.mat")["hypercube"].astype(np.float32)
im = ndimage.zoom(im / im.max(), [scale, scale, scale], order=0)

hidx, widx, tidx = np.where(im > 0.99)
im = im[hidx.min() : hidx.max(), widx.min() : widx.max(), tidx.min() : tidx.max()]

volutils.march_and_save(im, mcubes_thresh, savename_original, True)
H, W, T = im.shape
gt_signal = torch.tensor(im).float().reshape(H * W * T, 1)
# %%
train_dataset = alpine.dataloaders.BatchedNDSignalLoader(
    signal=gt_signal,
    grid_dims=[H, W, T],
    bounds=(-1, 1),
    vectorized=True,
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=16,
    pin_memory=True,
    shuffle=True,
)

test_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=16,
    pin_memory=True,
    shuffle=False,
)

# %%
inr_lightning = alpine.trainers.LightningTrainer(
    model=wire_model, return_features=False, log_results=False, is_distributed=True
)

# %%
trainer = pl.Trainer(
    accelerator="cuda",
    devices=len(gpu_ids.split(",")),
    max_epochs=NUM_EPOCHS,
    strategy="ddp",
)

# %%
trainer.fit(inr_lightning, train_dataloaders=train_dataloader)

trainer2 = pl.Trainer(accelerator="cuda", devices=1, max_epochs=1)
trainer2.test(inr_lightning, dataloaders=test_dataloader)

# %%
final_output = inr_lightning.stacked_test_output.reshape(H, W, T)
volutils.march_and_save(final_output, mcubes_thresh, savename, True)

# %%
