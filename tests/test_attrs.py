from collections.abc import Mapping
from typing import Any

import hypothesis as hp
import pytest as pt

from xarray_jsonschema import AttrsModel, ValidationError

from .strategies import attrs


class TestAttrs:
    @hp.given(expected=attrs())
    def test_generates_valid_schema(self, expected: Mapping) -> None:
        """Should produce valid JSON Model."""
        model = AttrsModel(expected)
        assert AttrsModel.check_schema(model.to_schema()) is None

    def test_default_value(self) -> None:
        """Should produce a default schema if no attrs are provided."""
        model = AttrsModel()
        assert model.to_schema() == {
            '$schema': 'https://json-schema.org/draft/2020-12/schema',
            'type': 'object',
        }

    @hp.given(expected=attrs())
    def test_argument_is_not_kw_only(self, expected: Mapping) -> None:
        assert AttrsModel(expected) == AttrsModel(attrs=expected)

    @hp.given(expected=attrs())
    def test_validation(self, expected: Mapping[str, Any]) -> None:
        """Should pass if the instance attrs matches the expected mapping."""
        AttrsModel(expected).validate(expected)

    def test_validation_custom(self) -> None:
        """Should pass if the instance attrs matches the expected mapping."""
        expected = {
            'a': str,
            'b': 'b',
            'c': int,
            'd': 42,
            'e': float,
            'f': 3.14,
            'g': bool,
            'h': True,
            'i': {
                'a': str,
                'b': 'b',
                'c': {
                    'c': int,
                    'd': 42,
                },
            },
        }
        actual = {
            'a': 'a',
            'b': 'b',
            'c': 42,
            'd': 42,
            'e': 3.14,
            'f': 3.14,
            'g': True,
            'h': True,
            'i': {
                'a': 'a',
                'b': 'b',
                'c': {
                    'c': 42,
                    'd': 42,
                },
            },
        }
        AttrsModel(expected).validate(actual)

    @hp.given(expected=attrs(min_items=1), actual=attrs(min_items=1))
    def test_invalidation(self, expected: Mapping, actual: Mapping) -> None:
        """Should fail if the instance attrs do not match the expected mapping."""
        hp.assume(all(k not in actual for k in expected.keys()))

        with pt.raises(ValidationError):
            AttrsModel(expected).validate(actual)
