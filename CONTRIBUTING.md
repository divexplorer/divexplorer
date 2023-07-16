# Contributing

All work and changes can be directed to `main` directly or through branches.

## Publishing

### TestPyPI

This is the testing version of PyPI that allows you to test the package before publishing it as the real deal.

To push to TestPyPI, draft and publish a new release from the [releases tab](https://github.com/divexplorer/divexplorer/releases) by tagging it as **a pre-release**. Make sure to **increment each** version (`0.1.2a0` -> `0.1.2a1`) in the `pyproject.toml` config file, and then with the pre-release.

```toml
version = "0.1.2a4"
```

![TestPyPI](https://i.gyazo.com/cc623fb89ee19ceb6561b798eb14bb21.png)

### PyPI

This is the production version of the package, essentially the real deal.

To push to PyPI, draft and publish a new release from the [releases tab](https://github.com/divexplorer/divexplorer/releases) by tagging it as **the latest release**. Make sure to **increment each** version (`0.1.0` -> `0.1.1`) in the `pyproject.toml` config file, and then with the release.

```toml
version = "0.1.2"
```

![PyPI](https://i.gyazo.com/44cab1b7e0b60133cf1eb3e43b9f1eee.png)

## Testing

### TestPyPI

Find all releases [here](https://test.pypi.org/project/DivExplorer/#history)

https://test.pypi.org/project/DivExplorer/VERSION/

```bash
pip install -i https://test.pypi.org/simple/ DivExplorer==VERSION
```

### PyPI

Find all releases [here](https://pypi.org/project/divexplorer/#history)

https://pypi.org/project/DivExplorer

```bash
pip install DivExplorer
```
