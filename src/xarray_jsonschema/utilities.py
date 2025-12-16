from typing import Mapping

from xarray_jsonschema import XarraySchema
from xarray_jsonschema.serializers import ObjectSerializer


def mapping_to_object_serializer(
    data: Mapping[str, XarraySchema], *, strict: bool = False
) -> ObjectSerializer:
    """Convert a mapping of schema components to an ``ObjectSerializer``
    instance.

    Parameters
    ----------
    data : Mapping[str, XarraySchema]
        A mapping of schema components. If the schema component has a regex attribute,
        the key will be treated as a regex pattern.
    strict : bool, default False
        A flag indicating if additional properties should be allowed.
    """
    properties = {}
    pattern_properties = {}
    required = set()
    required_pattern_properties = set()
    for key, value in data.items():
        is_regex = getattr(value, 'regex', False)
        is_required = getattr(value, 'required', False)
        if is_regex:
            pattern_properties[key] = value.serializer
            if is_required:
                required_pattern_properties.add(key)
        else:
            properties[key] = value.serializer
            if getattr(value, 'required', False):
                required.add(key)
    return ObjectSerializer(
        properties=properties or None,
        pattern_properties=pattern_properties or None,
        required=required or None,
        required_pattern_properties=required_pattern_properties or None,
        additional_properties=not strict,
    )
