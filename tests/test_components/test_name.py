import re

import hypothesis as hp
import pytest as pt
import xarray as xr
import xarray.testing.strategies as xrst
from hypothesis import strategies as st
from jsonschema.exceptions import ValidationError

from xarray_jsonschema import NameSchema
from xarray_jsonschema.testing import data_arrays, patterns


class TestName:
    @hp.given(data=st.data())
    def test_name_schema_is_valid(self, data: st.DataObject) -> None:
        """Should always produce a valid JSON Schema"""
        regex = data.draw(st.booleans())
        if regex:
            name = data.draw(patterns()).pattern
        else:
            name = data.draw(
                st.one_of(st.none(), st.text(), st.lists(st.text()))
            )
        min_length = data.draw(st.integers(min_value=0))
        max_length = data.draw(st.integers(min_value=min_length))
        hp.assume(min_length < max_length)

        schema = NameSchema(
            name, regex=regex, min_length=min_length, max_length=max_length
        )
        assert schema.check_schema() is None

    @hp.given(instance=data_arrays())
    def test_validation_with_string_passes(self, instance: xr.DataArray):
        """Should pass if the name matches a string."""
        name = str(instance.name)
        NameSchema(name).validate(instance)

    @hp.given(expected=xrst.names(), instance=data_arrays())
    def test_validation_with_string_fails(
        self, expected: str, instance: xr.DataArray
    ):
        """Should fail if the name does not match a string."""
        hp.assume(expected != instance.name)
        with pt.raises(ValidationError):
            NameSchema(expected).validate(instance)

    @hp.given(
        data=st.data(),
        expected=st.lists(xrst.names(), min_size=1),
    )
    def test_validation_with_sequence_passes(
        self,
        data: st.DataObject,
        expected: list[str],
    ):
        """Should pass if the name is in a sequence."""
        instance = data.draw(data_arrays(name=st.sampled_from(expected)))
        NameSchema(expected).validate(instance)

    @hp.given(
        expected=st.lists(xrst.names(), min_size=1), instance=data_arrays()
    )
    def test_validation_with_sequence_fails(
        self, expected: list[str], instance: xr.DataArray
    ):
        """Should fail if the name is not in a sequence."""
        hp.assume(instance.name not in expected)
        with pt.raises(ValidationError):
            NameSchema(expected).validate(instance)

    @hp.given(data=st.data(), expected=patterns())
    def test_validation_with_regex_passes(
        self, data: st.DataObject, expected: re.Pattern
    ):
        """Should pass if the name matches a regex pattern"""
        instance = data.draw(data_arrays(name=st.from_regex(expected.pattern)))
        NameSchema(expected.pattern, regex=True).validate(instance)

    @hp.given(instance=data_arrays())
    def test_validation_with_regex_fails(self, instance: xr.DataArray):
        """Should fail if the name matches a regex pattern"""
        expected = r'^expected$'
        with pt.raises(ValidationError):
            NameSchema(expected).validate(instance)

    @hp.given(data=st.data())
    def test_validation_with_length_constraints_passes(
        self, data: st.DataObject
    ):
        """Should pass if the name satisfies the length constraints"""
        min_length = data.draw(st.integers(min_value=0, max_value=3))
        max_length = data.draw(st.integers(min_value=min_length, max_value=6))
        names = st.text(min_size=min_length, max_size=max_length)
        instance = data.draw(data_arrays(name=names))
        NameSchema(min_length=min_length, max_length=max_length).validate(
            instance
        )


    @hp.given(data=st.data())
    def test_validation_with_length_constraints_fails(
        self, data: st.DataObject
    ):
        """Should fail if the name does not satisfy the length constraints"""
        max_length = data.draw(st.integers(min_value=1, max_value=6))
        names = st.text(min_size=max_length + 1)
        instance = data.draw(data_arrays(name=names))
        with pt.raises(ValidationError):
            NameSchema(max_length=max_length).validate(instance)
