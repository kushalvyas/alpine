## Alpine - A PyTorch Library for Implicit Neural Representations


![alpine_logo](./assets/alpine_logo.png)


Welcome to Alpine! We provide an easy and extensible way to rapidly prototype implicit neural representations or neural fields (INRs) in PyTorch. Alpine offers modular and object-oriented interfaces allowing you to get started with INRs from the get go. We provide clean interfaces minimizing overhead and boilerplate code. 

Alpine also offers a powerful visualization library that helps you take a peek under the hood. We provide interpretable visualizations such as PCA of learned features, gradient monitoring and histograms.

> [!IMPORTANT]
> Alpine is work in progress. We appreciate any constructive community feedback and support. We invite all the researchers across all disciplines to explore, and suggest any features you find particularly useful. Our goal is to make `Alpine` the go-to place for scientific computing using INRs!


## Documentation
Please refer to the [hosted documentation](https://kushalvyas.github.io/alpine-docs) for API documentation.

## Setup instructions

To only install the core Alpine library (lightweight):

```bash
git clone git@github.com:kushalvyas/alpine.git
cd alpine/
pip install . # Core installation (lightweight)
```

For development mode (editable install):
```bash
pip install -e .
```

### Optional Dependencies

Alpine provides optional extras for specialized features:

| Extra | Install Command | Features |
|-------|-----------------|----------|
| `bio` | `pip install ".[bio]"` | NIfTI/PDB file loading (`nibabel`, `biopandas`, `pandas`) |
| `mesh` | `pip install ".[mesh]"` | 3D mesh generation (`open3d`, `mcubes`, `scikit-image`) |
| `vis` | `pip install ".[vis]"` | PCA visualization (`scikit-learn`) |
| `all` | `pip install ".[all]"` | All optional dependencies |

Example usage with extras:
```python
from alpine.bio import load_nii_gz, load_pdb   # Requires: pip install ".[bio]"
from alpine.mesh import march_and_save         # Requires: pip install ".[mesh]"
from alpine.vis import pca                     # Requires: pip install ".[vis]"
```


## Contribution
We'll have our contribution guidelines up soon. Meanwhile, please feel free to report issues or raise any PRs.

## Examples

We provide extensive examples across the tasks of ( in progress )
- Audio (1-D)
- Images (2-D) 
- Volumes (3-D) fitting
- 3D Protein structure modelling (using the RCSB protein databank)
- Fitting gigapixel signals (using MINER)
- Representing hyperspectral volumes
- Solving phase recovery in optics (inverse problem)
- Solving CT reconstruction from sparse measurements (inverse problem)



## Citation

If you find Alpine useful, please consider citing us!
    
    @misc{vyas_alpine_2025,
        author = {Vyas, Kushal and Saragadam, Vishwanath and Veeraraghavan, Ashok and Balakrishnan, Guha},
        title = {Alpine: A Flexible, User-Friendly, and Distributed PyTorch Library for Implicit Neural Representation Development},
        booktitle={CVPR Workshop Neural Fields Beyond Conventional Cameras},
        year={2025},
        url={https://github.com/kushalvyas/alpine}
    }

    @software{vyas_alpine_2025
        author={ Vyas, Kushal and Kim, Daniel and Saragadam, Vishwanath and Veeraraghavan, Ashok and Balakrishnan, Guha},
        title = {Alpine: A Flexible, User-Friendly, and Distributed PyTorch Library for Implicit Neural Representation Development},
        year={2025},
        url={https://github.com/kushalvyas/alpine}
    }

---------


### Acknowledgements
Logo designed by Isha Chakraborty.