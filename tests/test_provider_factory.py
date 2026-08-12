import pytest

from src.providers.mock_provider import MockProvider
from src.providers.provider_factory import create_provider


def test_mock_provider_is_created():
    provider = create_provider("mock")

    assert isinstance(provider, MockProvider)
    assert provider.provider_name == "mock"


def test_unknown_provider_is_rejected():
    with pytest.raises(
        ValueError,
        match="Unknown provider",
    ):
        create_provider("does_not_exist")
