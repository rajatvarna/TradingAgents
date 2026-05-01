import pytest

from cli.main import extract_content_string


@pytest.mark.unit
def test_extract_content_string_preserves_literal_looking_text():
    assert extract_content_string("0") == "0"
    assert extract_content_string("False") == "False"


@pytest.mark.unit
def test_extract_content_string_treats_empty_containers_as_empty():
    assert extract_content_string([]) is None
    assert extract_content_string({}) is None
    assert extract_content_string("") is None
