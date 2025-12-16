"""
A custom JSON Schema validator for xarray-jsonschema.
"""

import re
from collections.abc import Iterable, Mapping
from typing import Any

from jsonschema import (
    Draft202012Validator,
    TypeChecker,
    ValidationError,
    validators,
)
from jsonschema.protocols import Validator

__all__ = ['XarrayValidator']


def required_pattern_properties(
    validator: Validator,
    patterns: Iterable[str],
    instance: Any,
    schema: Mapping | bool,
):
    """Check that an object contains at least one property matching a regular
    expression pattern."""
    if not validator.is_type(instance, 'object'):
        return
    for pattern in patterns:
        regex = re.compile(pattern)
        if not any(regex.fullmatch(key) for key in instance):
            yield ValidationError(
                f'{pattern!r} is a required pattern property.'
            )


def is_array_like(checker: TypeChecker, instance: Any):
    """Check if an instance is an array-like object."""
    return Draft202012Validator.TYPE_CHECKER.is_type(
        instance, 'array'
    ) or isinstance(instance, tuple)


XarrayValidator: Validator = validators.extend(
    validator=Draft202012Validator,
    validators={'requiredPatternProperties': required_pattern_properties},
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
