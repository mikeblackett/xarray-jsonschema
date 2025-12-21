"""This module provides a custom SchemaBuilder and SchemaStrategy classes for handling xarray data structures.

The classes defined here differ significantly from their default ``genson`` parent classes.

The most important difference is the way Python types/scalars are handled.

In ``genson``, Python scalars generate "type" keywords

builder = genson.SchemaBuilder(None)
builder.add_object('i am a string')
builder.to_schema() # {'type': 'string'}

In ``xarray_jsonschema``, Python scalars generate "const" keywords

builder = xarray_jsonschema.SchemaBuilder(None)
builder.add_object('i am a string')
builder.to_schema() # {'const': 'i am a string'}

In ``genson``, Python types produce ``SchemaGenerationError``:

builder = gs.SchemaBuilder(None)
builder.add_object(str)

In ``xarray_jsonschema``, Python types generate "type" keywords:

builder = xarray_jsonschema.SchemaBuilder(None)
builder.add_object(str)
builder.to_schema() # {'type': 'string'}

This module also adds:

- Very basic Draft 6 tuple validation support;
- Regular expression support for string validation.
"""

import re
from collections.abc import Mapping

import genson as gs
import genson.schema.strategies as st

__all__ = ['SchemaBuilder']


class Tuple(st.SchemaStrategy):
    KEYWORDS = (
        *st.SchemaStrategy.KEYWORDS,
        'prefixItems',
        'minItems',
        'maxItems',
    )
    PYTHON_TYPE = (list, tuple, set)

    @classmethod
    def match_schema(cls, schema: Mapping):
        return schema.get('type') == 'array' and 'prefixItems' in schema

    @classmethod
    def match_object(cls, obj: object) -> bool:
        return isinstance(obj, cls.PYTHON_TYPE)

    def __init__(self, node_class: type[gs.SchemaNode]):
        super().__init__(node_class)
        self.prefix_items = [node_class()]

    def add_schema(self, schema: Mapping) -> None:
        super().add_schema(schema)
        self._add(schema['prefixItems'], 'add_schema')

    def add_object(self, obj: list | tuple | set) -> None:
        self._add(obj, 'add_object')

    def _add(self, items: list | tuple | set, func: str) -> None:
        while len(self.prefix_items) < len(items):
            self.prefix_items.append(self.node_class())

        for subschema, item in zip(self.prefix_items, items):
            getattr(subschema, func)(item)

    def items_to_schema(self) -> list[Mapping]:
        return [item.to_schema() for item in self.prefix_items]

    def to_schema(self) -> dict:
        schema = super().to_schema()
        schema['type'] = 'array'
        items = self.items_to_schema()
        size = len(items)
        if size > 0 and not items == [{}]:
            schema['prefixItems'] = items
            schema['minItems'] = len(items)
            schema['maxItems'] = len(items)
        return schema


class Pattern(st.String):
    KEYWORDS = (*st.String.KEYWORDS, 'pattern')
    PYTHON_TYPE = re.Pattern

    @classmethod
    def match_schema(cls, schema: Mapping) -> bool:
        return 'string' in schema and 'pattern' in schema

    @classmethod
    def match_object(cls, obj: object) -> bool:
        return isinstance(obj, cls.PYTHON_TYPE)

    def __init__(self, node_class: type[gs.SchemaNode]) -> None:
        super().__init__(node_class)
        self.pattern = None

    def add_schema(self, schema: Mapping) -> None:
        super().add_schema(schema)
        self.pattern = re.compile(schema.get('pattern', self.pattern))

    def add_object(self, obj: re.Pattern) -> None:
        super().add_object(obj)
        self.pattern = obj

    def to_schema(self) -> dict:
        schema = super().to_schema()
        schema['pattern'] = getattr(self.pattern, 'pattern', self.pattern)
        return schema


class Const(st.SchemaStrategy):
    KEYWORDS = ('const',)
    PYTHON_TYPE = (str, int, float, bool)

    @classmethod
    def match_schema(cls, schema: Mapping) -> bool:
        return 'const' in schema

    @classmethod
    def match_object(cls, obj: object) -> bool:
        return isinstance(obj, cls.PYTHON_TYPE)

    def __init__(self, node_class: type[gs.SchemaNode]) -> None:
        super().__init__(node_class)
        self.const = None

    def add_schema(self, schema: Mapping) -> None:
        super().add_schema(schema)
        self.const = schema.get('const', self.const)

    def add_object(self, obj: str | int | float | bool) -> None:
        super().add_object(obj)
        self.const = obj

    def to_schema(self) -> dict:
        schema = super().to_schema()
        schema['const'] = self.const
        return schema


class _Type(st.TypedSchemaStrategy):
    JS_TYPE: str
    PYTHON_TYPE: type | tuple[type, ...]

    @classmethod
    def match_schema(cls, schema: Mapping) -> bool:
        return schema.get('type') == cls.JS_TYPE

    @classmethod
    def match_object(cls, obj: object):
        return isinstance(obj, type) and issubclass(obj, cls.PYTHON_TYPE)

    def to_schema(self) -> dict:
        schema = super().to_schema()
        schema['type'] = self.JS_TYPE
        return schema


class String(_Type):
    JS_TYPE = 'string'
    PYTHON_TYPE = (str,)


class Array(_Type):
    JS_TYPE = 'array'
    PYTHON_TYPE = (tuple, list, set)


class Boolean(_Type):
    JS_TYPE = 'boolean'
    PYTHON_TYPE = (bool,)


class Integer(_Type):
    JS_TYPE = 'integer'
    PYTHON_TYPE = (int,)


class Number(_Type):
    JS_TYPE = 'number'
    PYTHON_TYPE = (float,)


class Wildcard(st.SchemaStrategy):
    PYTHON_TYPE = (...,)

    @classmethod
    def match_schema(cls, schema):
        return False

    @classmethod
    def match_object(cls, obj):
        return obj is ...


class SchemaBuilder(gs.SchemaBuilder):
    DEFAULT_URI = 'https://json-schema.org/draft/2020-12/schema'
    EXTRA_STRATEGIES = (
        Boolean,
        Const,
        Integer,
        Number,
        Pattern,
        String,
        Tuple,
        Wildcard,
    )
