from typing import Sequence

import hypothesis as hp
import pytest as pt
import xarray.testing.strategies as xt

from xarray_jsonschema import DimsModel, ValidationError


class TestDims:
    @hp.given(expected=xt.dimension_names(min_dims=1))
    def test_generates_valid_schema(self, expected: Sequence) -> None:
        """Should produce valid JSON Model."""
        model = DimsModel(expected)
        assert DimsModel().check_schema(model.to_schema()) is None

    def test_default_value(self) -> None:
        """Should produce a default schema if no attrs are provided."""
        model = DimsModel()
        assert model.to_schema() == {
            '$schema': 'https://json-schema.org/draft/2020-12/schema',
            'type': 'array',
        }

    @hp.given(expected=xt.dimension_names(min_dims=1))
    def test_argument_is_not_kw_only(self, expected: Sequence) -> None:
        assert DimsModel(expected) == DimsModel(dims=expected)

    @hp.given(expected=xt.dimension_names(min_dims=1))
    def test_validation(self, expected: Sequence) -> None:
        """Should pass if the instance dims matches the expected sequence."""
        DimsModel(expected).validate(expected)

    @hp.given(
        expected=xt.dimension_names(min_dims=1), actual=xt.dimension_names()
    )
    def test_invalidation(self, expected: Sequence, actual: Sequence) -> None:
        """Should fail if the instance dims do not match the expected sequence."""
        hp.assume(expected != actual)
        print(expected)
        print(actual)
        with pt.raises(ValidationError):
            DimsModel(expected).validate(actual)
