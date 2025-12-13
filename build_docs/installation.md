Installation
=============

Alpine is being actively built on [GitHub](https://github.com/kushalvyas/alpine). Currently, Alpine can be installed via pip or can be built from source.

## pip
```{code-block} shell
git clone https://github.com/kushalvyas/alpine.git # if using https

OR

git clone git@github.com:kushalvyas/alpine.git # if using git-ssh
```

Once you download the alpine source code, you can use pip to install alpine and its dependencies from the `pyproject.toml' file.

```{code-block} shell
cd alpine/

pip install .
```

## Build from source
You can also build alpine from source by using the available `setup.py` file.

```{code-block} shell
cd alpine

python setup.py install
```