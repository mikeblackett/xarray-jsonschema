import hypothesis as hp
import numpy as np
import pytest as pt
from hypothesis import strategies as st
from jsonschema import ValidationError
from numpy.typing import DTypeLike
import xarray.testing.strategies as xrst

import xarray_jsonschema.testing as xmst
from xarray_jsonschema import DTypeSchema
from xarray_jsonschema.testing import supported_dtype_likes


class TestDType:
    @hp.given(dtype_like=supported_dtype_likes())
    def test_dtype_schema_is_valid(self, dtype_like: DTypeLike) -> None:
        """Should always produce a valid JSON Schema"""
        schema = DTypeSchema(dtype_like)
        assert schema.check_schema() is None

    @hp.given(expected=xmst.supported_dtype_likes())
    def test_validation_passes(self, expected: DTypeLike) -> None:
        """Should pass if the instance matches a dtype-like."""
        DTypeSchema(expected).validate(np.dtype(expected))

    @hp.given(data=st.data())
    def test_validation_fails(self, data: st.DataObject) -> None:
        """Should fail if the instance does not match a dtype-like."""
        expected = data.draw(xmst.supported_dtype_likes())
        instance = data.draw(xrst.supported_dtypes())
        # # np.dtype(None) returns 'float64', but xrm.DimsSchema(None) is a wildcard...
        hp.assume(np.dtype(expected) != instance)
        with pt.raises(ValidationError):
            DTypeSchema(expected).validate(instance)
