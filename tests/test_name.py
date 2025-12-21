import re

import hypothesis as hp
import hypothesis.strategies as st
import pytest as pt
import xarray.testing.strategies as xt

from xarray_jsonschema import NameModel, ValidationError

REGEX_PATTERNS = [
    r'[A-Za-z]{3,10}',  # Letters only, 3-10 chars
    r'\d{2,4}',  # 2-4 digits
    r'[A-Z][a-z]+\d{2}',  # Capitalized word with 2 digits
    r'[a-z]+[-_][a-z]+',  # Two lowercase words with dash/underscore
    r'[A-Z0-9]{8}',  # 8 chars of uppercase letters and numbers
    r'^[a-z]+_\d{3}$',  # Lowercase word with 3 digits suffix
    r'^\d{2}-[A-Z]{3}$',  # 2 digits followed by 3 uppercase letters
    r'^[A-Z][a-z]+$',  # Single capitalized word
    r'^\d{4}-[A-Z]{2}$',  # 4 digits followed by 2 uppercase letters
    r'^v\d+\.\d+$',  # Version number format
]


@st.composite
def patterns(draw: st.DrawFn) -> re.Pattern:
    """Generate regular expression patterns."""
    return re.compile(draw(st.sampled_from(REGEX_PATTERNS)))


class TestName:
    @hp.given(data=st.data())
    def test_generates_valid_schema(self, data: st.DataObject) -> None:
        """Should produce valid JSON Model."""
        _name = data.draw(st.one_of(patterns(), xt.names()))
        name = getattr(_name, 'pattern', _name)

        model = NameModel(name)
        assert NameModel.check_schema(model.to_schema()) is None

    def test_default_value(self) -> None:
        """Should produce a default schema if no attrs are provided."""
        model = NameModel()
        assert model.to_schema() == {
            '$schema': 'https://json-schema.org/draft/2020-12/schema',
            'type': 'string',
        }

    @hp.given(expected=xt.names())
    def test_argument_is_not_kw_only(self, expected: str) -> None:
        assert NameModel(expected) == NameModel(name=expected)

    @hp.given(expected=xt.names())
    def test_string_validation(self, expected: str):
        """Should pass if the instance name matches the expected string."""
        NameModel(expected).validate(expected)

    @hp.given(expected=xt.names(), actual=xt.names())
    def test_string_invalidation(self, expected: str, actual: str):
        """Should fail if the instance name does not match the expected string."""
        hp.assume(actual != expected)
        with pt.raises(ValidationError):
            NameModel(expected).validate(actual)

    @hp.given(data=st.data())
    def test_regex_validation(self, data: st.DataObject):
        """Should pass if the instance name matches the expected regex."""
        expected = data.draw(patterns())
        actual = data.draw(st.from_regex(expected.pattern))
        NameModel(expected).validate(actual)

    @hp.given(data=st.data())
    def test_regex_invalidation(self, data: st.DataObject):
        """Should fail if the instance name does not match the expected regex."""
        actual = 'actual'
        expected = data.draw(patterns())
        hp.assume(re.match(expected, actual) is None)
        with pt.raises(ValidationError):
            NameModel(expected).validate(actual)
