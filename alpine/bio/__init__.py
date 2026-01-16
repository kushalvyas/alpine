"""Alpine Bio subpackage for biological data formats.

This subpackage requires additional dependencies. Install with:
    pip install alpine[bio]
"""

from .loaders import load_nii_gz, load_pdb

__all__ = ["load_nii_gz", "load_pdb"]
