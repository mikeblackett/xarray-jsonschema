"""This module provides a custom JSON Schema validator for xarray-jsonschema."""

from jsonschema import (
    Draft202012Validator,
    TypeChecker,
    validators,
)
from jsonschema.exceptions import SchemaError, ValidationError
from jsonschema.protocols import Validator

__all__ = ['XarrayValidator', 'SchemaError', 'ValidationError']


def is_array_like(checker: TypeChecker, instance: object):
    """Check if an instance is an array-like object."""
    return Draft202012Validator.TYPE_CHECKER.is_type(
        instance, 'array'
    ) or isinstance(instance, tuple)


XarrayValidator: type[Validator] = validators.extend(
    validator=Draft202012Validator,
    type_checker=Draft202012Validator.TYPE_CHECKER.redefine(
        'array', is_array_like
    ),
)
"""A custom JSON Schema validator for xarray objects.

This validator extends the ``Draft202012Validator`` with the following features:

- interprets the ``tuple`` Python type as a valid instance of the 'array' data type
"""
