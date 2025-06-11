from collections.abc import Hashable

import hypothesis as hp
from jsonschema import ValidationError
import pytest as pt
import xarray.testing.strategies as xrst
from hypothesis import strategies as st

from xarray_jsonschema import DimsSchema


class TestDims:
    @hp.given(data=st.data())
    def test_dims_schema_is_valid(self, data: st.DataObject) -> None:
        """Should always produce a valid JSON Schema"""
        dims = data.draw(st.one_of(st.none(), xrst.dimension_names()))
        contains = data.draw(st.one_of(st.none(), st.text()))
        min_dims = data.draw(st.integers(min_value=0))
        max_dims = data.draw(st.integers(min_value=min_dims))
        schema = DimsSchema(
            dims, contains=contains, max_dims=max_dims, min_dims=min_dims
        )
        assert schema.check_schema() is None

    @hp.given(expected=xrst.dimension_names())
    def test_validation_with_sequence_passes(self, expected: list[Hashable]):
        """Should pass if the instance matches a sequence of names."""
        DimsSchema(expected).validate(expected)

    @hp.given(
        expected=xrst.dimension_names(min_dims=1),
        instance=xrst.dimension_names(),
    )
    def test_validation_with_sequence_fails(
        self, expected: list[Hashable], instance: list[Hashable]
    ):
        """Should fail if the instance does not match a sequence of names."""
        hp.assume(expected != instance)
        with pt.raises(ValidationError):
            DimsSchema(expected).validate(instance)

    @hp.given(instance=xrst.dimension_names(min_dims=1))
    def test_validation_with_contains_passes(self, instance: list[Hashable]):
        """Should pass if the instance contains a name."""
        expected = str(next(iter(instance)))
        DimsSchema(contains=expected).validate(instance)

    @hp.given(expected=xrst.names(), instance=xrst.dimension_names(min_dims=1))
    def test_validation_with_contains_fails(
        self, expected: str, instance: list[Hashable]
    ):
        """Should fail if the instance does not contain a name."""
        hp.assume(expected not in instance)
        with pt.raises(ValidationError):
            DimsSchema(contains=expected).validate(instance)

    @hp.given(
        data=st.data(),
    )
    def test_validation_with_size_passes(self, data: st.DataObject):
        """Should pass if the instance is within the size constraints."""
        min_dims = data.draw(st.integers(min_value=0, max_value=1))
        max_dims = data.draw(st.integers(min_value=min_dims + 1, max_value=5))
        instance = data.draw(
            xrst.dimension_names(min_dims=min_dims, max_dims=max_dims)
        )
        DimsSchema(min_dims=min_dims, max_dims=max_dims).validate(instance)

    @hp.given(data=st.data())
    def test_validation_with_size_constraints_fails(self, data: st.DataObject):
        """Should fail if the instance is outside the size constraints."""
        min_dims = data.draw(st.integers(min_value=0, max_value=1))
        max_dims = data.draw(st.integers(min_value=min_dims + 1, max_value=5))
        instance = data.draw(
            xrst.dimension_names(min_dims=max_dims + 1, max_dims=max_dims + 1)
        )
        with pt.raises(ValidationError):
            DimsSchema(max_dims=min_dims).validate(instance)
