import re
from collections.abc import Sequence
from types import NoneType

import genson.schema.strategies as st
import numpy as np


class Tuple(st.SchemaStrategy):
    KEYWORDS = ('type',)
    PYTHON_TYPE = (Sequence,)

    @classmethod
    def match_schema(cls, schema):
        return schema.get('type') == 'array' and isinstance(
            schema.get('prefixItems'), list
        )

    @classmethod
    def match_object(cls, obj):
        return isinstance(obj, list | tuple)

    def __init__(self, node_class):
        super().__init__(node_class)
        self.prefix_items = [node_class()]

    def add_schema(self, schema):
        super().add_schema(schema)
        if 'prefixItems' in schema:
            self._add(schema['prefixItems'], 'add_schema')

    def add_object(self, obj):
        self._add(obj, 'add_object')

    def _add(self, items, func):
        while len(self.prefix_items) < len(items):
            self.prefix_items.append(self.node_class())

        for subschema, item in zip(self.prefix_items, items):
            getattr(subschema, func)(item)

    def items_to_schema(self):
        return [item.to_schema() for item in self.prefix_items]

    def to_schema(self):
        schema = super().to_schema()
        schema['type'] = 'array'
        if size := len(self.prefix_items):
            schema['prefixItems'] = self.items_to_schema()
            schema['minItems'] = size
            schema['maxItems'] = size

        return schema


class Pattern(st.TypedSchemaStrategy):
    JS_TYPE = 'string'
    PYTHON_TYPE = re.Pattern

    def __init__(self, node_class):
        super().__init__(node_class)
        self.pattern = None

    def add_schema(self, schema):
        super().add_schema(schema)
        self.pattern = re.compile(schema.get('pattern', self.pattern))

    def add_object(self, obj):
        super().add_object(obj)
        self.pattern = obj

    def to_schema(self):
        schema = super().to_schema()
        schema['pattern'] = getattr(self.pattern, 'pattern', self.pattern)
        return schema


class Constant(st.SchemaStrategy):
    KEYWORDS = ('const',)
    PYTHON_TYPE = (str, int, float, bool, NoneType)

    @classmethod
    def match_schema(cls, schema):
        return 'const' in schema

    @classmethod
    def match_object(cls, obj) -> bool:
        return isinstance(obj, cls.PYTHON_TYPE)

    def __init__(self, node_class):
        super().__init__(node_class)

        self.const = None

    def add_schema(self, schema):
        super().add_schema(schema)
        print(f'added by {self.__class__.__name__} ')
        self.const = schema.get('const', self.const)

    def add_object(self, obj):
        super().add_object(obj)
        self.const = obj

    def to_schema(self):
        schema = super().to_schema()
        schema['const'] = self.const
        return schema


class NumpyDType(st.SchemaStrategy):
    KEYWORDS = ('const',)
    PYTHON_TYPE = (np.dtype,)

    @classmethod
    def match_schema(cls, schema):
        return 'const' in schema

    @classmethod
    def match_object(cls, obj) -> bool:
        return isinstance(obj, cls.PYTHON_TYPE)

    def __init__(self, node_class):
        super().__init__(node_class)

        self.dtype = None

    def add_schema(self, schema):
        super().add_schema(schema)
        self.dtype = np.dtype(schema.get('const', self.dtype))

    def add_object(self, obj):
        super().add_object(obj)
        self.dtype = np.dtype(obj)

    def to_schema(self):
        schema = super().to_schema()
        schema['const'] = str(self.dtype)
        return schema


class Typed(st.SchemaStrategy):
    JS_TYPE: str | tuple[str]
    PYTHON_TYPE: type | tuple[type]

    @classmethod
    def match_schema(cls, schema):
        return schema.get('type') == cls.JS_TYPE

    @classmethod
    def match_object(cls, obj):
        return isinstance(obj, type) and issubclass(obj, cls.PYTHON_TYPE)

    def to_schema(self):
        schema = super().to_schema()
        schema['type'] = self.JS_TYPE
        return schema


class String(Typed):
    JS_TYPE = 'string'
    PYTHON_TYPE = (str,)


class Boolean(Typed):
    JS_TYPE = 'boolean'
    PYTHON_TYPE = (bool,)


class Integer(Typed):
    JS_TYPE = 'integer'
    PYTHON_TYPE = (int,)


class Number(Typed):
    JS_TYPE = 'number'
    PYTHON_TYPE = (float,)


class Null(Typed):
    JS_TYPE = 'null'
    PYTHON_TYPE = (NoneType,)
