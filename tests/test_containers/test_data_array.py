from collections.abc import Sequence

import hypothesis as hp
import numpy as np
import pytest as pt
import xarray as xr
import xarray.testing.strategies as xrst
from hypothesis import strategies as st
from jsonschema import ValidationError, Validator
from numpy.typing import DTypeLike

import xarray_model as xrm
import xarray_model.testing as xmst

from ._strategies import coords_models, data_array_models


class TestCoordsModel:
    @hp.given(model=coords_models())
    def test_schema_is_valid(self, validator: Validator, model: xrm.Coords):
        """Should always produce a valid JSON Schema with any combination of parameters."""
        schema = model.schema
        validator.check_schema(schema)


class TestDataArrayModel:
    @hp.given(model=data_array_models())
    def test_schema_is_valid(self, model: xrm.DataArrayModel):
        """Should always produce a valid JSON Schema with any combination of parameters."""
        model.check_schema()

    @hp.given(da=xmst.data_arrays())
    def test_validation_with_default_parameters_passes(self, da: xr.DataArray):
        """Should always pass with default values."""
        xrm.DataArrayModel().validate(da)


class TestAttrs:
    """Test integration with the attributes component."""

    @hp.given(da=xmst.data_arrays())
    def test_validation_with_allow_extra_items_passes(self, da: xr.DataArray):
        """Should always pass if ``Attrs.allow_extra_items`` is true."""
        expected = xrm.Attrs(allow_extra_items=True)
        xrm.DataArrayModel(attrs=expected).validate(da)

    @hp.given(da=xmst.data_arrays(attrs=xmst.attrs(min_items=1)))
    def test_validation_with_allow_extra_items_fails(self, da: xr.DataArray):
        """Should fail if the instance has unspecified attributes."""
        expected = xrm.Attrs(allow_extra_items=False)
        with pt.raises(ValidationError):
            xrm.DataArrayModel(attrs=expected).validate(da)

    @hp.given(da=xmst.data_arrays(attrs=xmst.attrs(min_items=1)))
    def test_validation_with_required_keys_passes(self, da: xr.DataArray):
        """Should pass if the instance contains the required attribute keys."""
        key = next(iter(da.attrs.keys()))
        expected = xrm.Attrs([xrm.Attr(key, required=True)])
        xrm.DataArrayModel(attrs=expected).validate(da)

    @hp.given(da=xmst.data_arrays(attrs=xmst.attrs(min_items=1)))
    def test_validation_with_optional_keys_passes(self, da: xr.DataArray):
        """Should pass if the instance contains the optional attribute keys."""
        key = next(iter(da.attrs.keys()))
        missing_key = 'random'
        hp.assume(missing_key != key)
        expected = xrm.Attrs(
            [
                xrm.Attr(key, required=False),
                xrm.Attr(missing_key, required=False),
            ]
        )
        xrm.DataArrayModel(attrs=expected).validate(da)

    @hp.given(da=xmst.data_arrays())
    def test_validation_with_required_keys_fails(self, da: xr.DataArray):
        """Should fail if the instance does not contain a required attribute key."""
        key = 'random'
        hp.assume(key not in da.attrs)
        expected = xrm.Attrs([xrm.Attr(key, required=True)])
        with pt.raises(ValidationError):
            xrm.DataArrayModel(attrs=expected).validate(da)

    @hp.given(da=xmst.data_arrays(), pattern=xmst.patterns(), data=st.data())
    def test_validation_with_key_regex_match_passes(
        self, da: xr.DataArray, pattern: str, data: st.DataObject
    ):
        """Should pass if the instance contains an attribute key that matches the specified pattern."""
        expected = xrm.Attrs([xrm.Attr(key=pattern, regex=True)])
        key = data.draw(st.from_regex(pattern))
        da = da.assign_attrs(**{key: 'value'})
        xrm.DataArrayModel(attrs=expected).validate(da)

    @hp.given(da=xmst.data_arrays(attrs=xmst.attrs(min_items=1)))
    def test_validation_with_key_regex_fails(self, da: xr.DataArray):
        """Should fail if the instance does not contain any keys that match the specified pattern."""
        # The ``required`` attribute doesn't apply to pattern properties,
        # so we need to apply `allow_extra_items=False`
        expected = xrm.Attrs(
            [xrm.Attr(key=r'^expected$', regex=True)], allow_extra_items=False
        )
        with pt.raises(ValidationError):
            xrm.DataArrayModel(attrs=expected).validate(da)

    @hp.given(
        da=xmst.data_arrays(attrs=xmst.attrs(min_items=1)), data=st.data()
    )
    def test_validation_with_key_value_pairs_passes(
        self, da: xr.DataArray, data: st.DataObject
    ):
        """Should pass if the instance contains the specified attribute key-value pair."""
        key, value = next(iter(da.attrs.items()))
        expected = xrm.Attrs([xrm.Attr(key, value=value)])
        xrm.DataArrayModel(attrs=expected).validate(da)

    @hp.given(
        da=xmst.data_arrays(attrs=xmst.attrs(min_items=1)), data=st.data()
    )
    def test_validation_with_key_value_pairs_fails(
        self, da: xr.DataArray, data: st.DataObject
    ):
        """Should fail if the instance does not contain the specified attribute key-value pair."""
        key, value = next(iter(da.attrs.items()))
        expected_value = data.draw(
            st.one_of(
                st.booleans(),
                st.integers(min_value=1),
                st.floats(allow_infinity=False),
                st.text(),
            )
        )
        hp.assume(
            not isinstance(value, np.ndarray) and value != expected_value
        )
        expected = xrm.Attrs([xrm.Attr(key, value=expected_value)])
        with pt.raises(ValidationError):
            xrm.DataArrayModel(attrs=expected).validate(da)

    @hp.given(
        da=xmst.data_arrays(attrs=xmst.attrs(min_items=1)), data=st.data()
    )
    def test_validation_with_key_type_pairs_passes(
        self, da: xr.DataArray, data: st.DataObject
    ):
        """Should pass if the instance contains the specified attribute key-type pair."""
        key, value = next(iter(da.attrs.items()))
        expected = xrm.Attrs([xrm.Attr(key, value=type(value))])
        xrm.DataArrayModel(attrs=expected).validate(da)

    @hp.given(
        da=xmst.data_arrays(attrs=xmst.attrs(min_items=1)), data=st.data()
    )
    def test_validation_with_key_type_pairs_fails(
        self, da: xr.DataArray, data: st.DataObject
    ):
        """Should fail if the instance does not contain the specified attribute key-type pair."""
        key, value = next(iter(da.attrs.items()))
        expected_type = data.draw(st.sampled_from([float, int, str, bool]))
        hp.assume(type(value) is not expected_type)
        if isinstance(value, (int, float)):
            # JSON Schema considers numbers like 1 and 1.0 both integers...
            try:
                # If float is NaN type conversion will fail...
                hp.assume(value != expected_type(value))
            except ValueError:
                return
        expected = xrm.Attrs([xrm.Attr(key, value=expected_type)])
        with pt.raises(ValidationError):
            xrm.DataArrayModel(attrs=expected).validate(da)

    @hp.given(da=xmst.data_arrays(attrs=xmst.attrs(min_items=1)))
    def test_validation_with_duplicate_keys_passes(self, da: xr.DataArray):
        """Should handle Attrs with duplicate keys"""
        key, value = next(iter(da.attrs.items()))
        expected = xrm.Attrs([xrm.Attr(key), xrm.Attr(key, value=value)])
        xrm.DataArrayModel(attrs=expected).validate(da)


class TestDataArrayChunks:
    @hp.given(data=st.data())
    def test_validation_with_boolean_passes(self, data: st.DataObject):
        """Should pass when the chunked/unchunked state matches a boolean."""
        expected = xrm.Chunks(data.draw(st.booleans()))
        da = data.draw(xmst.data_arrays(dims=xrst.dimension_sizes(min_dims=1)))
        if expected.chunks:
            da = da.chunk('auto')
        xrm.DataArrayModel(chunks=expected).validate(da)

    @hp.given(data=st.data())
    def test_validation_with_boolean_fails(self, data: st.DataObject):
        """Should fail when the chunked state does not match a boolean."""
        expected = xrm.Chunks(data.draw(st.booleans()))
        da = data.draw(xmst.data_arrays(dims=xrst.dimension_sizes(min_dims=1)))
        if not expected.chunks:
            da = da.chunk('auto')
        with pt.raises(ValidationError):
            xrm.DataArrayModel(chunks=expected).validate(da)

    @hp.given(data=st.data())
    def test_validation_with_integer_passes(self, data: st.DataObject):
        """Should pass if the block sizes of all dimensions match an integer."""
        expected = data.draw(st.integers(min_value=1, max_value=10))
        da = data.draw(
            xmst.data_arrays(
                dims=xrst.dimension_sizes(min_dims=1, min_side=expected)
            )
        )
        da = da.chunk(expected)
        xrm.DataArrayModel(chunks=xrm.Chunks(expected)).validate(da)

    @hp.given(data=st.data())
    def test_validation_with_integer_fails(self, data: st.DataObject):
        """Should fail if the block sizes of all dimensions do not match an integer."""
        expected = data.draw(st.integers(min_value=2, max_value=10))
        da = data.draw(xmst.data_arrays())
        da = da.chunk(1)
        with pt.raises(ValidationError):
            xrm.DataArrayModel(chunks=xrm.Chunks(expected)).validate(da)

    @hp.given(data=st.data())
    def test_validation_with_sequence_of_integers_passes(
        self, data: st.DataObject
    ):
        """Should pass if the block sizes of each dimension match a sequence of integers."""
        da = data.draw(xmst.data_arrays(dims=xrst.dimension_sizes(min_dims=1)))
        da = da.chunk('auto')
        assert da.chunks is not None
        expected = [c[0] for c in da.chunks]
        xrm.DataArrayModel(chunks=xrm.Chunks(expected)).validate(da)

    @hp.given(data=st.data())
    def test_validation_with_sequence_of_integers_fails(
        self, data: st.DataObject
    ):
        """Should fail if the block sizes of each dimension do not match a sequence of integers."""
        da = data.draw(xmst.data_arrays(dims=xrst.dimension_sizes(min_dims=1)))
        da = da.chunk('auto')
        assert da.chunks is not None
        expected = expected = [c[0] + 1 for c in da.chunks]
        with pt.raises(ValidationError):
            xrm.DataArrayModel(chunks=xrm.Chunks(expected)).validate(da)

    @hp.given(data=st.data())
    def test_validation_with_sequence_of_sequence_of_integers_passes(
        self, data: st.DataObject
    ):
        """
        Test exact block sizes per dimension.

        It is equivalent to testing the output of:
        `DataArray(dims=('x', 'y'), ...).chunk(x=(1, 2, 3), y=(1, 2, 3)).chunks`
        """
        da = data.draw(xmst.data_arrays(dims=xrst.dimension_sizes(min_dims=1)))
        da = da.chunk('auto')
        assert da.chunks is not None
        expected = da.chunks
        xrm.DataArrayModel(chunks=xrm.Chunks(expected)).validate(da)

    @hp.given(data=st.data())
    def test_validation_with_sequence_of_sequence_of_integers_fails(
        self, data: st.DataObject
    ):
        """Should fail if the block sizes of each dimension do not match a sequence of integers."""
        da = data.draw(xmst.data_arrays(dims=xrst.dimension_sizes(min_dims=1)))
        da = da.chunk('auto')
        assert da.chunks is not None
        expected = np.array(da.chunks) + 1
        with pt.raises(ValidationError):
            xrm.DataArrayModel(chunks=xrm.Chunks(expected.tolist())).validate(
                da
            )

    @hp.given(data=st.data())
    def test_validation_with_wildcard_passes(self, data: st.DataObject):
        da = data.draw(xmst.data_arrays(dims=xrst.dimension_sizes(min_dims=1)))
        da = da.chunk(-1)
        xrm.DataArrayModel(chunks=xrm.Chunks(-1)).validate(da)

    @hp.given(data=st.data())
    def test_validation_with_wildcard_fails(self, data: st.DataObject):
        da = data.draw(
            xmst.data_arrays(dims=xrst.dimension_sizes(min_dims=1, min_side=2))
        )
        da = da.chunk(1)
        with pt.raises(ValidationError):
            xrm.DataArrayModel(chunks=xrm.Chunks(-1)).validate(da)


class TestDims:
    @hp.given(data=st.data())
    def test_validation_with_sequence_passes(self, data: st.DataObject):
        """Should pass if the instance matches a sequence of names."""
        da = data.draw(xmst.data_arrays())
        expected = [str(name) for name in da.dims]
        xrm.DataArrayModel(dims=xrm.Dims(expected)).validate(da)

    @hp.given(
        da=xmst.data_arrays(),
        expected=xrst.dimension_names(min_dims=1),
    )
    def test_validation_with_sequence_fails(
        self, da: xr.DataArray, expected: Sequence[str]
    ):
        """Should fail if the instance does not match a sequence of names."""
        hp.assume(tuple(da.dims) != tuple(expected))
        with pt.raises(ValidationError):
            xrm.DataArrayModel(dims=xrm.Dims(expected)).validate(da)

    @hp.given(da=xmst.data_arrays(dims=xrst.dimension_sizes(min_dims=1)))
    def test_validation_with_contains_passes(self, da: xr.DataArray):
        """Should pass if the instance contains a name."""
        contains = str(next(iter(da.dims)))
        xrm.DataArrayModel(dims=xrm.Dims(contains=contains)).validate(da)

    @hp.given(
        da=xmst.data_arrays(dims=xrst.dimension_sizes(min_dims=1)),
        expected=xrst.names(),
    )
    def test_validation_with_contains_fails(
        self, da: xr.DataArray, expected: str
    ):
        """Should fail if the instance does not contain a name."""
        hp.assume(expected not in da.dims)
        with pt.raises(ValidationError):
            xrm.DataArrayModel(dims=xrm.Dims(contains=expected)).validate(da)

    @hp.given(data=st.data())
    def test_validation_with_size_constraints_passes(
        self, data: st.DataObject
    ):
        """Should pass if the instance is within the size constraints."""
        min_dims = data.draw(st.integers(min_value=0, max_value=1))
        max_dims = data.draw(st.integers(min_value=min_dims, max_value=3))
        hp.assume(min_dims <= max_dims)
        da = data.draw(
            xmst.data_arrays(
                dims=xrst.dimension_sizes(min_dims=min_dims, max_dims=max_dims)
            )
        )
        xrm.DataArrayModel(
            dims=xrm.Dims(min_dims=min_dims, max_dims=max_dims)
        ).validate(da)

    @hp.given(data=st.data())
    def test_validation_with_size_constraints_fails(self, data: st.DataObject):
        """Should fail if the instance is outside the size constraints."""
        da = data.draw(xmst.data_arrays())
        with pt.raises(ValidationError):
            xrm.DataArrayModel(dims=xrm.Dims(min_dims=4)).validate(da)


class TestDType:
    @hp.given(dtype_like=xmst.supported_dtype_likes(), data=st.data())
    def test_validation_passes(
        self, dtype_like: DTypeLike, data: st.DataObject
    ) -> None:
        """Should pass if the instance matches a dtype-like."""
        da = data.draw(xmst.data_arrays(dtype=st.just(np.dtype(dtype_like))))
        xrm.DataArrayModel(dtype=xrm.DType(dtype_like)).validate(da)

    @hp.given(data=st.data())
    def test_validation_fails(self, data: st.DataObject) -> None:
        """Should fail if the instance does not match a dtype-like."""
        expected = data.draw(xmst.supported_dtype_likes())
        da = data.draw(xmst.data_arrays())
        # np.dtype(None) returns 'float64', but xrm.Dims(None) is a wildcard...
        hp.assume(expected is not None and np.dtype(expected) != da.dtype)
        with pt.raises(ValidationError):
            xrm.DataArrayModel(dtype=xrm.DType(expected)).validate(da)


class TestName:
    @hp.given(da=xmst.data_arrays())
    def test_validation_with_string_passes(self, da: xr.DataArray):
        """Should pass if the name matches a string."""
        assert isinstance(da.name, str)
        xrm.DataArrayModel(name=xrm.Name(da.name)).validate(da)

    @hp.given(da=xmst.data_arrays(), expected=xrst.names())
    def test_validation_with_string_fails(
        self, da: xr.DataArray, expected: str
    ):
        """Should fail if the name does not match a string."""
        hp.assume(expected != da.name)
        with pt.raises(ValidationError):
            xrm.DataArrayModel(name=xrm.Name(expected)).validate(da)

    @hp.given(data=st.data())
    def test_validation_with_sequence_passes(self, data: st.DataObject):
        """Should pass if the name is in a sequence."""
        names = data.draw(xmst.dimension_names(min_dims=1))
        name = data.draw(st.sampled_from(names))
        da = data.draw(xmst.data_arrays(name=st.sampled_from(names)))
        xrm.DataArrayModel(name=xrm.Name(names)).validate(da)

    @hp.given(data=st.data())
    def test_validation_with_sequence_fails(self, data: st.DataObject):
        """Should fail if the name is not in a sequence."""
        names = data.draw(xrst.dimension_names(min_dims=1))
        name = data.draw(xrst.names())
        hp.assume(name not in names)
        da = data.draw(xmst.data_arrays(name=st.sampled_from(names)))
        with pt.raises(ValidationError):
            xrm.DataArrayModel(name=xrm.Name(name)).validate(da)

    @hp.given(pattern=xmst.patterns(), data=st.data())
    def test_validation_with_regex_passes(
        self, pattern: str, data: st.DataObject
    ):
        """Should pass if the name matches a regex pattern"""
        da = data.draw(xmst.data_arrays(name=st.from_regex(pattern)))
        xrm.DataArrayModel(name=xrm.Name(pattern, regex=True)).validate(da)

    @hp.given(data=st.data())
    def test_validation_with_regex_fails(self, data: st.DataObject):
        """Should pass if the name matches a regex pattern"""
        da = data.draw(xmst.data_arrays())
        expected = r'^expected$'
        with pt.raises(ValidationError):
            xrm.DataArrayModel(name=xrm.Name(expected, regex=True)).validate(
                da
            )

    @hp.given(data=st.data())
    def test_validation_with_length_constraints(self, data: st.DataObject):
        """Should pass if the name satisfies the length constraints"""
        min_length = data.draw(st.integers(min_value=0, max_value=3))
        max_length = data.draw(st.integers(min_value=min_length, max_value=6))
        da = data.draw(
            xmst.data_arrays(
                name=st.text(min_size=min_length, max_size=max_length)
            )
        )
        xrm.DataArrayModel(
            name=xrm.Name(min_length=min_length, max_length=max_length)
        ).validate(da)

    @hp.given(data=st.data())
    def test_validation_with_size_constraints(self, data: st.DataObject):
        """Should fail if the name does not satisfy the length constraints"""
        min_length = data.draw(st.integers(min_value=1, max_value=10))
        max_length = data.draw(st.integers(min_value=min_length, max_value=20))
        da = data.draw(
            xmst.data_arrays(
                name=st.one_of(
                    st.text(min_size=max_length + 1),
                    st.text(max_size=min_length - 1),
                )
            )
        )
        with pt.raises(ValidationError):
            xrm.DataArrayModel(
                name=xrm.Name(min_length=min_length, max_length=max_length)
            ).validate(da)


class TestShape:
    @hp.given(da=xmst.data_arrays())
    def test_validation_with_sequence_passes(self, da: xr.DataArray):
        """Should pass when the shape matches a sequence of integers."""
        expected = da.shape
        xrm.DataArrayModel(shape=xrm.Shape(expected)).validate(da)

    @hp.given(da=xmst.data_arrays(), shape=xmst.dimension_shapes())
    def test_validation_with_sequence_fails(
        self, da: xr.DataArray, shape: Sequence[int]
    ):
        """Should fail when the shape does not match a sequence of integers."""
        hp.assume(da.shape != shape)
        with pt.raises(ValidationError):
            xrm.DataArrayModel(shape=xrm.Shape(shape)).validate(da)

    @hp.given(data=st.data())
    def test_validation_with_item_constraints_passes(
        self, data: st.DataObject
    ):
        """Should pass if the instance matches the item constraints."""
        min_dims = data.draw(st.integers(min_value=1, max_value=3))
        max_dims = data.draw(st.integers(min_value=min_dims, max_value=5))
        da = data.draw(
            xmst.data_arrays(
                dims=xrst.dimension_sizes(min_dims=min_dims, max_dims=max_dims)
            )
        )
        xrm.DataArrayModel(
            shape=xrm.Shape(max_dims=max_dims, min_dims=min_dims)
        ).validate(da)

    #
    @hp.given(data=st.data())
    def test_validation_with_item_constraints_fails(self, data: st.DataObject):
        """Should fail if the instance does not match the item constraints."""
        min_dims = data.draw(st.integers(min_value=1, max_value=3))
        max_dims = data.draw(st.integers(min_value=min_dims, max_value=5))
        da = data.draw(
            xmst.data_arrays(
                dims=xrst.dimension_sizes(min_dims=min_dims, max_dims=max_dims)
            )
        )
        with pt.raises(ValidationError):
            xrm.DataArrayModel(
                shape=xrm.Shape(max_dims=min_dims - 1)
            ).validate(da)
            xrm.DataArrayModel(
                shape=xrm.Shape(min_dims=max_dims + 1)
            ).validate(da)


class TestCoords:
    @hp.given(da=xmst.data_arrays())
    def test_validation_with_allow_extra_passes(self, da: xr.DataArray):
        """Should always pass if ``Attrs.allow_extra_items`` is true."""
        expected = xrm.Coords(allow_extra=True)
        xrm.DataArrayModel(coords=expected).validate(da)

    @hp.given(da=xmst.data_arrays(dims=xrst.dimension_sizes(min_dims=1)))
    def test_validation_with_allow_extra_items_fails(self, da: xr.DataArray):
        """Should fail if the instance has unspecified attributes."""
        expected = xrm.Coords(allow_extra=False)
        with pt.raises(ValidationError):
            xrm.DataArrayModel(coords=expected).validate(da)

    @hp.given(da=xmst.data_arrays(dims=xrst.dimension_sizes(min_dims=1)))
    def test_validation_with_require_all_passes(self, da: xr.DataArray):
        """Should pass if the instance has all required coordinates."""
        name = next(iter(da.coords.keys()))
        expected = xrm.Coords(
            coords={name: xrm.DataArrayModel(name=xrm.Name(name))},
            require_all=True,
        )
        xrm.DataArrayModel(coords=expected).validate(da)

    @hp.given(da=xmst.data_arrays(dims=xrst.dimension_sizes(min_dims=1)))
    def test_validation_with_allow_require_all__fails(self, da: xr.DataArray):
        """Should fail if the instance does not have all required coordinates."""
        name = 'required'
        expected = xrm.Coords(
            coords={name: xrm.DataArrayModel(name=xrm.Name(name))},
            require_all=True,
        )
        with pt.raises(ValidationError):
            xrm.DataArrayModel(coords=expected).validate(da)
