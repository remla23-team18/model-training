# Model Training

[![Tests](https://github.com/remla23-team18/model-training/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/remla23-team18/model-training/actions/workflows/tests.yml)
[![Codecov](https://codecov.io/gh/remla23-team18/model-training/branch/main/graph/badge.svg)][codecov]
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

[codecov]: https://app.codecov.io/gh/remla23-team18/model-training
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

## Requirements

- python >= 3.10
- poetry >= 1.5

## Installation

You can install the package locally for development using `poetry`:

```console
$ poetry install
```

## Usage

### Direct

After installing the package, you can run the CLI using `poetry`:

```console
$ poetry run model-training --help
```

will print the usage information for the CLI.

To get the training/testing data you'll need to use `dvc`. See the following section for more information.

### DVC

After installing the package as mentioned above, you may also run `dvc pull` and `dvc repro` to let `dvc` automatically execute the stages of the pipeline for you.

All of the stages of the pipeline are executed by running various subcommands of the CLI.

By default you have read access to the Google Cloud Storage bucket where the data is stored.

## mllint

To check `mllint` manually, first install with `poetry` as described above, then run:

```console
pytest --junitxml=tests-report.xml --cov-report=xml --cov=model_training
```

and finally, to check the report:

```console
mllint run
```

### Experiment

The performance of the model is currently reported in the `classification_report.json` file in the root directory.

If you want to run an experiment and observe the results, then you can run the following command:

```console
$ dvc exp run --set-param split_random_state=1
```

This will run the experiment with a fixed random state for the train-test split.

You can then compare the results of this experiment with the results of other experiments by running:

```console
$ dvc exp diff
```

### Testing

```console
$ poetry run pytest
```

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [MIT license][license],
_Model Training_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

This project was generated from [@cjolowicz]'s [Hypermodern Python Cookiecutter] template.

[@cjolowicz]: https://github.com/cjolowicz
[pypi]: https://pypi.org/
[hypermodern python cookiecutter]: https://github.com/cjolowicz/cookiecutter-hypermodern-python
[file an issue]: https://github.com/remla23-team18/model-training/issues
[pip]: https://pip.pypa.io/

<!-- github-only -->

[license]: https://github.com/remla23-team18/model-training/blob/main/LICENSE
[contributor guide]: https://github.com/remla23-team18/model-training/blob/main/CONTRIBUTING.md
