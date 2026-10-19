import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def sample_batch():
    return torch.randn(2, 1, 64, 64)


@pytest.fixture
def sample_labels():
    return torch.randint(0, 5, (2, 64, 64))


@pytest.fixture
def sample_classification_labels():
    return torch.randint(0, 3, (2,))


@pytest.fixture
def sample_multiclass_predictions():
    return torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])


@pytest.fixture
def sample_multiclass_targets():
    return torch.tensor([0, 1, 2])
