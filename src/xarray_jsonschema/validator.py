"""
A custom JSON Schema validator for xarray-jsonschema.
"""

from typing import Any

from jsonschema import (
    Draft202012Validator,
    TypeChecker,
    validators,
)
from jsonschema.protocols import Validator

__all__ = ['XarrayValidator']


def is_array_like(checker: TypeChecker, instance: Any):
    """Check if an instance is an array-like object."""
    return Draft202012Validator.TYPE_CHECKER.is_type(
        instance, 'array'
    ) or isinstance(instance, tuple)


XarrayValidator: Validator = validators.extend(
    validator=Draft202012Validator,
    type_checker=Draft202012Validator.TYPE_CHECKER.redefine(
        'array', is_array_like
    ),
)
"""
A custom JSON Schema validator for xarray objects.

This validator extends the ``Draft202012Validator`` with the following features:

- accepts the ``tuple`` type as an instance of the 'array' type
- supports the custom ``requiredPatternProperties`` keyword, which validates
  that the keys of an object contain at least one property matching a given
  regular expression pattern.
"""
