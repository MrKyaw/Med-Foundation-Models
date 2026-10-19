# Contributing to Med-Foundation-Models

Thank you for your interest in contributing to Med-Foundation-Models! We welcome contributions from the community.

## How to Contribute

### Reporting Bugs

- Check if the issue already exists in [GitHub Issues](../../issues)
- Include a minimal reproducible example
- Specify your environment (Python version, PyTorch version, CUDA version)

### Suggesting Features

- Open a [GitHub Discussion](../../discussions) first to gather feedback
- Clearly describe the use case and expected behavior
- Consider submitting a pull request with a proposed implementation

### Pull Requests

1. Fork the repository
2. Create a new branch from `main`
3. Make your changes
4. Write/update tests
5. Run the test suite: `pytest tests/ -v`
6. Run linting: `black --check . && isort --check .`
7. Submit a pull request

## Code Style

- Follow [PEP 8](https://pep8.org/) with a line length of 100
- Use [black](https://github.com/psf/black) for formatting
- Use [isort](https://github.com/PyCQA/isort) for import sorting
- Add type hints to all public functions and classes

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add JEPA-style pretraining for MRI
fix: resolve data loader issue with multi-channel inputs
docs: add installation instructions for CUDA 12.1
test: add unit tests for multimodal fusion module
```

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Open a [GitHub Discussion](../../discussions) or reach out to the maintainers.
