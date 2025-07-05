import re
from collections.abc import Mapping, Sequence
from typing import Any

import hypothesis as hp
import numpy as np
import pytest as pt
import xarray as xr
from hypothesis import strategies as st
from jsonschema import ValidationError

from xarray_jsonschema import AttrSchema, AttrsSchema
from xarray_jsonschema.testing import (
    attrs,
    patterns,
    data_arrays,
)


class TestAttrs:
    @hp.given(
        value=st.one_of(
            st.none(), st.booleans(), st.integers(), st.floats(), st.text()
        ),
        regex=st.booleans(),
        required=st.booleans(),
        data=st.data(),
    )
    def test_attr_schema_is_valid(
        self,
        value: None | bool | int | float | str | type,
        regex: bool,
        required: bool,
        data: st.DataObject,
    ):
        """Should produce a valid JSON Schema with any combination of arguments."""
        if data.draw(st.booleans()) and value is not None:
            value = type(value)
        schema = AttrSchema(value, regex=regex, required=required)
        schema.check_schema()

    @hp.given(
        expected=attrs(),
        strict=st.booleans(),
    )
    def test_attrs_schema_is_valid(
        self,
        expected: dict[str, Any],
        strict: bool,
    ):
        """Should produce a valid JSON Schema with any combination of arguments."""
        schema = AttrsSchema(expected, strict=strict)
        assert schema.check_schema() is None

    @hp.given(instance=data_arrays(attrs=attrs(min_items=1)))
    def test_strict_validation_fails(self, instance: xr.DataArray):
        """Should fail if the instance has unspecified attributes."""
        with pt.raises(ValidationError):
            AttrsSchema(strict=True).validate(instance)

    @hp.given(instance=data_arrays())
    def test_strict_validation_passes(self, instance: xr.DataArray):
        """Should pass if the instance has unspecified attributes."""
        AttrsSchema(strict=False).validate(instance)

    @hp.given(instance=data_arrays(attrs=attrs(min_items=1)))
    def test_validation_with_required_keys_passes(
        self, instance: xr.DataArray
    ):
        """Should pass if the instance contains the required attribute keys."""
        key = next(iter(instance.attrs.keys()))
        AttrsSchema({key: AttrSchema(required=True)}).validate(instance)

    @hp.given(instance=data_arrays())
    def test_validation_with_required_keys_fails(self, instance: xr.DataArray):
        """Should fail if the instance does not contain a required attribute key."""
        key = 'random'
        hp.assume(key not in instance.attrs)
        with pt.raises(ValidationError):
            AttrsSchema({key: AttrSchema(required=True)}).validate(instance)

    @hp.given(instance=data_arrays(attrs=attrs(min_items=1)))
    def test_validation_with_optional_keys_passes(
        self, instance: xr.DataArray
    ):
        """Should pass with/without optional attribute keys."""
        missing = 'random'
        present = next(iter(instance.attrs.keys()))
        hp.assume(missing != present)
        AttrsSchema(
            {
                present: AttrSchema(required=False),
                missing: AttrSchema(required=False),
            }
        ).validate(instance)

    @hp.given(data=st.data(), pattern=patterns())
    def test_validation_with_key_regex_match_passes(
        self,
        data: st.DataObject,
        pattern: re.Pattern,
    ):
        """Should pass if the instance contains an attribute key that matches the specified pattern."""
        key = data.draw(st.from_regex(pattern.pattern))
        expected = {pattern.pattern: AttrSchema(regex=True)}
        instance = data.draw(data_arrays()).assign_attrs({key: 'value'})
        AttrsSchema(expected).validate(instance)

    @hp.given(instance=data_arrays(attrs=attrs(min_items=1)))
    def test_validation_with_key_regex_fails(self, instance: xr.DataArray):
        """Should fail if the instance does not contain any keys that match the specified pattern."""
        expected = {
            r'^expected$': AttrSchema(regex=True, required=True),
        }
        with pt.raises(ValidationError):
            AttrsSchema(
                expected,
            ).validate(instance)

    @hp.given(instance=data_arrays(attrs=attrs(min_items=1)))
    def test_validation_with_key_value_pairs_passes(
        self, instance: xr.DataArray
    ):
        """Should pass if the instance contains the specified attribute key-value pair."""
        key, value = next(iter(instance.attrs.items()))
        # TODO (mike): Update once we supported nested attrs
        hp.assume(not isinstance(value, (Mapping, Sequence)))
        expected = {
            key: value,
        }
        AttrsSchema(expected).validate(instance)

    @hp.given(instance=data_arrays(attrs=attrs(min_items=1)), data=st.data())
    def test_validation_with_key_value_pairs_fails(
        self, instance: xr.DataArray, data: st.DataObject
    ):
        """Should fail if the instance does not contain the specified attribute key-value pair."""
        key, value = next(iter(instance.attrs.items()))
        expected_value = data.draw(
            st.one_of(
                st.booleans(),
                st.integers(min_value=1),
                st.floats(allow_infinity=False),
                st.text(),
            )
        )
        hp.assume(
            not isinstance(value, np.ndarray) and value != expected_value
        )
        with pt.raises(ValidationError):
            AttrsSchema({key: AttrSchema(expected_value)}).validate(instance)

    @hp.given(instance=data_arrays(attrs=attrs(min_items=1)))
    def test_validation_with_key_type_pairs_passes(
        self, instance: xr.DataArray
    ):
        """Should pass if the instance contains the specified attribute key-value pair."""
        key, value = next(iter(instance.attrs.items()))
        expected = {
            key: type(value),
        }
        AttrsSchema(expected).validate(instance)

    @hp.given(instance=data_arrays(attrs=attrs(min_items=1)), data=st.data())
    def test_validation_with_key_type_pairs_fails(
        self, instance: xr.DataArray, data: st.DataObject
    ):
        """Should fail if the instance does not contain the specified attribute key-value pair."""
        key, value = next(iter(instance.attrs.items()))
        expected_type = data.draw(st.sampled_from([int, str, bool]))
        hp.assume(type(value) is not expected_type)
        if isinstance(value, (int, float)):
            # JSON Schema considers numbers like 1 and 1.0 both integers...
            try:
                # If float is NaN, type conversion will fail...
                hp.assume(value != expected_type(value))
            except ValueError:
                return
        expected = {key: expected_type}
        with pt.raises(ValidationError):
            AttrsSchema(expected).validate(instance)
