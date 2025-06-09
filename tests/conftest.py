import pytest as pt
from xarray_model.validators import XarrayModelValidator


@pt.fixture(scope='package')
def validator():
    return XarrayModelValidator
