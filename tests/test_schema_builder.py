"""Tests for the custom genson.SchemaBuilder and genson.SchemaStrategy classes
defined in xarray_jsonschema.schema."""

import hypothesis as hp
import hypothesis.strategies as st

from tests.strategies import patterns, readable_text
from xarray_jsonschema.schema import (
    SchemaBuilder,
    Tuple,
)

scalars = st.one_of(
    st.integers(), st.floats(allow_nan=False), readable_text(), st.booleans()
)


class TestTuple:
    @hp.given(data=st.data())
    def test_add_object(self, data: st.DataObject) -> None:
        builder = SchemaBuilder(None)  # type: ignore[reportArgumentType]
        sequence = data.draw(st.sampled_from(Tuple.PYTHON_TYPE))
        obj = sequence(data.draw(st.iterables(scalars, min_size=1)))

        builder.add_object(obj)
        schema = builder.to_schema()

        assert schema.get('type') == 'array'
        assert isinstance(schema.get('prefixItems'), list)
        assert (
            len(schema['prefixItems'])
            == schema.get('minItems')
            == schema.get('maxItems')
        )

    @hp.given(data=st.data())
    def test_add_schema(self, data: st.DataObject) -> None:
        builder = SchemaBuilder(None)  # type: ignore[reportArgumentType]

        schema = {
            'type': 'array',
            'prefixItems': [
                {'type': 'integer'},
                {'const': 'help'},
                {'const': True},
            ],
            'minItems': 3,
            'maxItems': 3,
        }
        builder.add_schema(schema)

        assert builder.to_schema() == schema


class TestPattern:
    @hp.given(data=st.data())
    def test_add_object(self, data: st.DataObject) -> None:
        builder = SchemaBuilder(None)  # type: ignore[reportArgumentType]
        obj = data.draw(patterns())

        builder.add_object(obj)
        schema = builder.to_schema()

        assert schema.get('type') == 'string'
        assert schema.get('pattern') == obj.pattern

    @hp.given(data=st.data())
    def test_add_schema(self, data: st.DataObject) -> None:
        builder = SchemaBuilder(None)  # type: ignore[reportArgumentType]

        schema = {'type': 'string', 'pattern': '$\\d{2,4}^'}
        builder.add_schema(schema)

        assert builder.to_schema() == schema


class TestConst:
    @hp.given(data=st.data())
    def test_add_object(self, data: st.DataObject) -> None:
        builder = SchemaBuilder(None)  # type: ignore[reportArgumentType]
        obj = data.draw(scalars)

        builder.add_object(obj)
        schema = builder.to_schema()

        assert 'const' in schema
        assert schema.get('const') == obj

    @hp.given(data=st.data())
    def test_add_schema(self, data: st.DataObject) -> None:
        builder = SchemaBuilder(None)  # type: ignore[reportArgumentType]

        schema = {'const': 42}
        builder.add_schema(schema)

        assert builder.to_schema() == schema


class TestString:
    @hp.given(data=st.data())
    def test_add_object(self, data: st.DataObject) -> None:
        builder = SchemaBuilder(None)  # type: ignore[reportArgumentType]
        obj = str

        builder.add_object(obj)
        schema = builder.to_schema()

        assert schema.get('type') == 'string'

    @hp.given(data=st.data())
    def test_add_schema(self, data: st.DataObject) -> None:
        builder = SchemaBuilder(None)  # type: ignore[reportArgumentType]

        schema = {'type': 'string'}
        builder.add_schema(schema)

        assert builder.to_schema() == schema


class TestBoolean:
    @hp.given(data=st.data())
    def test_add_object(self, data: st.DataObject) -> None:
        builder = SchemaBuilder(None)  # type: ignore[reportArgumentType]
        obj = bool

        builder.add_object(obj)
        schema = builder.to_schema()

        assert schema.get('type') == 'boolean'

    @hp.given(data=st.data())
    def test_add_schema(self, data: st.DataObject) -> None:
        builder = SchemaBuilder(None)  # type: ignore[reportArgumentType]

        schema = {'type': 'boolean'}
        builder.add_schema(schema)

        assert builder.to_schema() == schema


class TestInteger:
    @hp.given(data=st.data())
    def test_add_object(self, data: st.DataObject) -> None:
        builder = SchemaBuilder(None)  # type: ignore[reportArgumentType]
        obj = int

        builder.add_object(obj)
        schema = builder.to_schema()

        assert schema.get('type') == 'integer'

    @hp.given(data=st.data())
    def test_add_schema(self, data: st.DataObject) -> None:
        builder = SchemaBuilder(None)  # type: ignore[reportArgumentType]

        schema = {'type': 'integer'}
        builder.add_schema(schema)

        assert builder.to_schema() == schema


class TestNumber:
    @hp.given(data=st.data())
    def test_add_object(self, data: st.DataObject) -> None:
        builder = SchemaBuilder(None)  # type: ignore[reportArgumentType]
        obj = float

        builder.add_object(obj)
        schema = builder.to_schema()

        assert schema.get('type') == 'number'

    @hp.given(data=st.data())
    def test_add_schema(self, data: st.DataObject) -> None:
        builder = SchemaBuilder(None)  # type: ignore[reportArgumentType]

        schema = {'type': 'number'}
        builder.add_schema(schema)

        assert builder.to_schema() == schema


class TestNull:
    @hp.given(data=st.data())
    def test_add_object(self, data: st.DataObject) -> None:
        builder = SchemaBuilder(None)  # type: ignore[reportArgumentType]
        obj = None

        builder.add_object(obj)
        schema = builder.to_schema()
        print(schema)
        assert schema.get('type') == 'null'

    @hp.given(data=st.data())
    def test_add_schema(self, data: st.DataObject) -> None:
        builder = SchemaBuilder(None)  # type: ignore[reportArgumentType]

        schema = {'type': 'null'}
        builder.add_schema(schema)

        assert builder.to_schema() == schema


class TestWildcard:
    @hp.given(data=st.data())
    def test_add_object(self, data: st.DataObject) -> None:
        builder = SchemaBuilder(None)  # type: ignore[reportArgumentType]
        obj = ...

        builder.add_object(obj)

        assert builder.to_schema() == {}

    # Wildcard does not support add_schema()
