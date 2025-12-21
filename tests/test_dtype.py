import hypothesis as hp
import numpy as np
import pytest as pt

from xarray_jsonschema import DTypeModel, ValidationError
from xarray_jsonschema.model import DTypeLike

from .strategies import supported_dtype_likes


class TestDType:
    @hp.given(expected=supported_dtype_likes())
    def test_generates_valid_schema(self, expected: DTypeLike) -> None:
        """Should produce valid JSON Model."""
        model = DTypeModel(expected)
        assert DTypeModel.check_schema(model.to_schema()) is None

    def test_default_value(self) -> None:
        """Should produce a default schema if no attrs are provided."""
        model = DTypeModel()
        assert model.to_schema() == {
            '$schema': 'https://json-schema.org/draft/2020-12/schema',
            'const': 'float64',
        }

    @hp.given(expected=supported_dtype_likes())
    def test_argument_is_not_kw_only(self, expected: DTypeLike) -> None:
        assert DTypeModel(expected) == DTypeModel(dtype=expected)

    @hp.given(expected=supported_dtype_likes())
    def test_converter(self, expected: DTypeLike) -> None:
        """Should convert attribute values"""
        model = DTypeModel(expected)
        assert model.dtype == str(np.dtype(expected))

    @hp.given(expected=supported_dtype_likes())
    def test_validation(self, expected: DTypeLike) -> None:
        """Should pass if the dtype matches the expected dtype-like."""
        actual = np.dtype(expected)
        DTypeModel(expected).validate(actual)

    @hp.given(expected=supported_dtype_likes(), actual=supported_dtype_likes())
    def test_invalidation(
        self, expected: DTypeLike, actual: DTypeLike
    ) -> None:
        """Should fail if the dtype does not match the expected dtype-like."""
        actual = np.dtype(actual)
        expected = np.dtype(expected)

        hp.assume(actual != expected)

        with pt.raises(ValidationError):
            DTypeModel(expected).validate(np.dtype(actual))
