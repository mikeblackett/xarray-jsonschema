import re
from collections.abc import Mapping
from typing import Any

import hypothesis as hp
import numpy as np
import pytest as pt
from hypothesis import strategies as st
from jsonschema import ValidationError

from xarray_jsonschema import AttrSchema, AttrsSchema
from xarray_jsonschema.testing import attrs, patterns, readable_text


class TestAttrs:
    @hp.given(
        key=st.one_of(readable_text(), patterns()),
        value=st.one_of(
            st.none(), st.booleans(), st.integers(), st.floats(), st.text()
        ),
        required=st.booleans(),
        data=st.data(),
    )
    def test_attr_schema_is_valid(
        self,
        key: str | re.Pattern,
        value: Any,
        required: bool,
        data: st.DataObject,
    ):
        """Should always produce a valid JSON Schema"""
        if data.draw(st.booleans()) and value is not None:
            value = type(value)
        schema = AttrSchema(value, required=required)
        schema.check_schema()

    @hp.given(allow_extra_items=st.booleans(), data=st.data())
    def test_attrs_schema_is_valid(
        self,
        allow_extra_items: bool,
        data: st.DataObject,
    ):
        """Should always produce a valid JSON Schema"""
        expected = data.draw(
            st.dictionaries(
                keys=st.one_of(st.text(), patterns()),
                values=st.one_of(
                    st.none(),
                    st.booleans(),
                    st.floats(),
                    st.integers(),
                    readable_text(),
                ),
            )
        )
        schema = AttrsSchema(expected, allow_extra_items=allow_extra_items)
        print(schema.json)
        assert schema.check_schema() is None

    @hp.given(instance=attrs())
    def test_validation_with_allow_extra_items_passes(
        self, instance: Mapping[str, Any]
    ):
        """Should always pass if ``AttrsSchema.allow_extra_items`` is true."""
        AttrsSchema(allow_extra_items=True).validate(instance)

    @hp.given(instance=attrs(min_items=1))
    def test_validation_with_allow_extra_items_fails(
        self, instance: Mapping[str, Any]
    ):
        """Should fail if the instance has unspecified attributes."""
        with pt.raises(ValidationError):
            AttrsSchema(allow_extra_items=False).validate(instance)

    @hp.given(instance=attrs(min_items=1))
    def test_validation_with_required_keys_passes(
        self, instance: Mapping[str, Any]
    ):
        """Should pass if the instance contains the required attribute keys."""
        key = next(iter(instance.keys()))
        AttrsSchema({key: AttrSchema(required=True)}).validate(instance)

    @hp.given(instance=attrs(min_items=1))
    def test_validation_with_optional_keys_passes(
        self, instance: Mapping[str, Any]
    ):
        """Should pass if the instance contains the optional attribute keys."""
        key = next(iter(instance.keys()))
        missing_key = 'random'
        hp.assume(missing_key != key)
        expected = {
            key: AttrSchema(required=False),
            missing_key: AttrSchema(required=False),
        }
        AttrsSchema(expected).validate(instance)

    @hp.given(instance=attrs())
    def test_validation_with_required_keys_fails(
        self, instance: Mapping[str, Any]
    ):
        """Should fail if the instance does not contain a required attribute key."""
        key = 'random'
        hp.assume(key not in instance)
        expected = {key: None}
        with pt.raises(ValidationError):
            AttrsSchema(expected).validate(instance)

    @hp.given(pattern=patterns(), data=st.data())
    def test_validation_with_key_regex_match_passes(
        self,
        pattern: re.Pattern,
        data: st.DataObject,
    ):
        """Should pass if the instance contains an attribute key that matches the specified pattern."""
        key = data.draw(st.from_regex(pattern.pattern))
        instance = {key: 'value'}
        expected = {pattern: None}
        AttrsSchema(expected).validate(instance)

    @hp.given(instance=attrs(min_items=1))
    def test_validation_with_key_regex_fails(
        self, instance: Mapping[str, Any]
    ):
        """Should fail if the instance does not contain any keys that match the specified pattern."""
        # The ``required`` attribute doesn't apply to pattern properties,
        # so we need to apply `allow_extra_items=False`
        expected = {
            r'^expected$': None,
        }
        with pt.raises(ValidationError):
            AttrsSchema(
                expected,
                allow_extra_items=False,
            ).validate(instance)

    @hp.given(instance=attrs(min_items=1))
    def test_validation_with_key_value_pairs_passes(
        self, instance: Mapping[str, Any]
    ):
        """Should pass if the instance contains the specified attribute key-value pair."""
        key, value = next(iter(instance.items()))
        expected = {
            key: value,
        }
        AttrsSchema(expected).validate(instance)

    @hp.given(instance=attrs(min_items=1), data=st.data())
    def test_validation_with_key_value_pairs_fails(
        self, instance: Mapping[str, Any], data: st.DataObject
    ):
        """Should fail if the instance does not contain the specified attribute key-value pair."""
        key, value = next(iter(instance.items()))
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

    @hp.given(instance=attrs(min_items=1))
    def test_validation_with_key_type_pairs_passes(
        self, instance: Mapping[str, Any]
    ):
        """Should pass if the instance contains the specified attribute key-value pair."""
        key, value = next(iter(instance.items()))
        expected = {
            key: type(value),
        }
        AttrsSchema(expected).validate(instance)

    @hp.given(instance=attrs(min_items=1), data=st.data())
    def test_validation_with_key_type_pairs_fails(
        self, instance: Mapping[str, Any], data: st.DataObject
    ):
        """Should fail if the instance does not contain the specified attribute key-value pair."""
        key, value = next(iter(instance.items()))
        expected_type = data.draw(st.sampled_from([float, int, str, bool]))
        hp.assume(type(value) is not expected_type)
        if isinstance(value, (int, float)):
            # JSON Schema considers numbers like 1 and 1.0 both integers...
            try:
                # If float is NaN type conversion will fail...
                hp.assume(value != expected_type(value))
            except ValueError:
                return
        expected = {key: expected_type}
        with pt.raises(ValidationError):
            AttrsSchema(expected).validate(instance)

    @hp.given(instance=attrs(min_items=1))
    def test_validation_with_duplicate_keys_passes(
        self, instance: Mapping[str, Any]
    ):
        """Should handle AttrsSchema with duplicate keys"""
        key, value = next(iter(instance.items()))
        expected = {key: None, key: value}
        AttrsSchema(expected).validate(instance)
