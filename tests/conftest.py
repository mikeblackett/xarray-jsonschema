import pytest as pt
from xarray_jsonschema.validators import XarrayModelValidator


@pt.fixture(scope='package')
def validator():
    return XarrayModelValidator
