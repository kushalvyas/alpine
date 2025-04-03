from setuptools import setup, find_packages
import os, os.path as osp

# Determine the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.dirname(__file__))


setup(
    name="alpine",  # Replace with your package name
    version="1.0.0",  # Update version as needed
    description="A python package for working with INRs",
    author="Kushal Vyas, Vishwanath Saragadam, Ashok Veeraraghavan, Guha Balakrishnan",  # Replace with your name
    author_email="kushalkvyas@gmail.com",  # Replace with your email
    # url="https://github.com/kushalvyas/alpine",  # Replace with your URL if hosted
    packages=find_packages(exclude=['*tests*', '*examples*','*docs*']),  # Automatically find packages in your directory
    
    install_requires=[
        # Add package dependencies here, e.g.:
        # 'numpy>=1.19.0',
    ],
    python_requires=">=3.10",  # Specify compatible Python versions
)
