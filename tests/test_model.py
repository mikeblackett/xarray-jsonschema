"""Tests for the base Model class implemented in jsonschema.model"""

from __future__ import annotations

import attrs as at
import hypothesis as hp
import hypothesis.strategies as st
import pytest as pt
from genson.schema.builder import json

from xarray_jsonschema import Model, SchemaError, ValidationError


@at.define(frozen=True)
class SimpleModel(Model):
    uid: str | type[str] | None = None
    count: int | type[int] | None = None

    def build(self) -> None:
        return super().build()

    def validate(self, obj) -> None:
        return super().validate(obj)


@at.define(frozen=True)
class ComplexModel(Model):
    uid: str | type[str] | None = str
    count: int | type[int] | None = int
    nested: SimpleModel | None = at.field(
        default=None,
        converter=at.converters.optional(SimpleModel),
    )


class TestModelClass:
    def test_check_schema(self) -> None:
        """Should validate the schema against the model's meta-schema."""
        model = SimpleModel()
        with pt.raises(SchemaError):
            model.check_schema({'type': 'banana'})

    def test_attrs_with_none_values_are_omitted_from_schema(self) -> None:
        model = SimpleModel(uid=None, count=None)
        assert model.to_schema() == {
            '$schema': 'https://json-schema.org/draft/2020-12/schema',
            'type': 'object',
        }

    def test_validates(self) -> None:
        """Should validate an object against the model's schema."""
        model = SimpleModel(uid=str, count=42)

        obj = {
            'uid': 'abc',
            'count': 42,
        }

        model.validate(obj)

    def test_invalidates(self) -> None:
        """Should validate an object against the model's schema."""
        model = SimpleModel(uid=str, count=int)

        obj = {
            'uid': 42,
            'count': 'abc',
        }

        with pt.raises(ValidationError):
            model.validate(obj)

    @hp.given(uid=st.text(), count=st.integers())
    def test_from_dict_to_dict_roundtrip(self, uid: str, count: int) -> None:
        """Should be round-trippable through to_dict and from_dict."""
        data = {
            'uid': uid,
            'count': count,
        }
        model = SimpleModel.from_dict(data)
        assert model.to_dict() == data

    def test_to_schema(self) -> None:
        """Should generate a schema from the model."""
        model = SimpleModel(uid=str, count=42)
        schema = model.to_schema()
        assert schema == {
            '$schema': 'https://json-schema.org/draft/2020-12/schema',
            'type': 'object',
            'properties': {
                'uid': {'type': 'string'},
                'count': {'const': 42},
            },
            'required': ['count', 'uid'],
        }

    def test_to_json(self) -> None:
        """Should generate a JSON string from the model."""
        model = SimpleModel(uid=str, count=42)
        assert json.dumps(model.to_schema()) == model.to_json()
