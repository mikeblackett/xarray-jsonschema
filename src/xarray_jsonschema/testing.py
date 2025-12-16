import re
from collections.abc import Hashable, Mapping, Sequence
from typing import TypeVar

import hypothesis as hp
import numpy as np
import xarray as xr
import xarray.testing.strategies as xrst
from hypothesis import strategies as st

REGEX_PATTERNS = [
    r'[A-Za-z]{3,10}',  # Letters only, 3-10 chars
    r'\d{2,4}',  # 2-4 digits
    r'[A-Z][a-z]+\d{2}',  # Capitalized word with 2 digits
    r'[a-z]+[-_][a-z]+',  # Two lowercase words with dash/underscore
    r'[A-Z0-9]{8}',  # 8 chars of uppercase letters and numbers
    r'^TEST_[A-Z]+$',  # Uppercase word with TEST_ prefix
    r'^[a-z]+_\d{3}$',  # Lowercase word with 3 digits suffix
    r'^\d{2}-[A-Z]{3}$',  # 2 digits followed by 3 uppercase letters
    r'^[A-Z][a-z]+$',  # Single capitalized word
    r'^user_\d+$',  # Username with number
    r'^\d{4}-[A-Z]{2}$',  # 4 digits followed by 2 uppercase letters
    r'^[a-z]{2}\d{2}$',  # 2 lowercase letters followed by 2 digits
    r'^v\d+\.\d+$',  # Version number format
    r'^[A-Z]{2}_\d{4}$',  # 2 uppercase letters followed by 4 digits
    r'^[a-z]+@test$',  # Lowercase word with @test suffix
]


_readable_characters = st.characters(
    categories=['L', 'N'], max_codepoint=0x017F
)


def readable_text(min_size: int = 0):
    return st.text(_readable_characters, min_size=min_size, max_size=5)


_attr_values = st.one_of(
    st.none(),
    st.booleans(),
    st.floats(),
    st.integers(),
    readable_text(),
)
_attr_keys = readable_text()


@st.composite
def positive_integers(draw: st.DrawFn):
    return draw(st.integers(min_value=1))


@st.composite
def non_negative_integers(draw: st.DrawFn):
    return draw(st.integers(min_value=0))


T = TypeVar('T', bound=Hashable)


def dimension_name(
    *,
    name_strategy: st.SearchStrategy[T] = xrst.names(),
    min_dims: int = 0,
    max_dims: int = 3,
) -> st.SearchStrategy[list[T]]:
    """
    Generates an arbitrary list of valid dimension names.

    Requires the hypothesis package to be installed.

    Parameters
    ----------
    name_strategy
        Strategy for making names. Useful if we need to share this.
    min_dims
        Minimum number of dimensions in generated list.
    max_dims
        Maximum number of dimensions in generated list.
    """
    return st.lists(
        elements=name_strategy,
        min_size=min_dims,
        max_size=max_dims,
        unique=True,
    )


def nested_attrs(
    min_items: int = 0, max_leaves: int = 3
) -> st.SearchStrategy[dict[str, None | bool | float | int | str]]:
    return st.recursive(
        base=st.dictionaries(
            keys=_attr_keys, values=_attr_values, min_size=min_items
        ),
        extend=lambda children: st.dictionaries(
            keys=_attr_keys, values=children, min_size=min_items
        ),
        max_leaves=max_leaves,
    )


@st.composite
def attrs(
    draw: st.DrawFn, min_items: int = 0
) -> dict[str, None | bool | float | int | str]:
    return draw(
        st.dictionaries(
            keys=_attr_keys, values=_attr_values, min_size=min_items
        )
    )


@st.composite
def supported_dtype_likes(
    draw: st.DrawFn,
    dtype: np.dtype | None = None,
) -> np.dtype | str | type | None:
    """Generate only those numpy DTypeLike that xarray can handle.

    See @https://numpy.org/doc/stable/reference/arrays.dtypes.html to know what
    can be converted to a data-type object.

    If a dtype is provided, then only those values that are compatible with
    the dtype will be returned.
    """
    if dtype is None:
        _dtype_strategy = st.one_of(
            st.none(),
            xrst.supported_dtypes(),
            st.sampled_from([int, float, bool, str, complex]),
        )
        dtype = np.dtype(draw(_dtype_strategy))

    return draw(
        st.sampled_from(
            [
                # dtype
                dtype,
                # string
                dtype.name,
                # array-protocol typestring
                dtype.str,
                # One-character strings
                dtype.char,
            ],
        )
    )


@st.composite
def patterns(draw: st.DrawFn) -> re.Pattern:
    """Generate regular expression patterns."""
    return re.compile(draw(st.sampled_from(REGEX_PATTERNS)))


@st.composite
def multiples_of(
    draw, base: int, min_value: int = 0, max_value: int | None = None
) -> int:
    multiplier = draw(st.integers(min_value=1))
    result = base * multiplier
    hp.assume(result >= min_value)
    if max_value is not None:
        hp.assume(result <= max_value)
    return result


@st.composite
def dimension_shapes(
    draw,
    min_dims: int = 1,
    max_dims: int = 5,
    min_side: int = 0,
    max_side: int | None = None,
) -> tuple[int, ...]:
    """Generate an arbitrary tuple of dimension sizes."""
    return tuple(
        draw(
            xrst.dimension_sizes(
                min_dims=min_dims,
                max_dims=max_dims,
                min_side=min_side,
                max_side=max_side,
            )
        ).values()
    )


@st.composite
def uniform_chunks(
    draw: st.DrawFn,
    min_block_size: int = 1,
    max_block_size: int | None = None,
    min_dims: int = 1,
    max_dims: int = 10,
):
    if max_block_size is None:
        max_block_size = draw(st.integers(min_value=min_block_size + 1))
    block_size = draw(
        st.integers(min_value=min_block_size, max_value=max_block_size)
    )
    multiplier = draw(st.integers(min_value=min_dims, max_value=max_dims))
    size = draw(
        st.integers(
            min_value=block_size * min_dims, max_value=block_size * multiplier
        )
    )
    n = size // block_size
    last_block = size % block_size
    blocks = [block_size] * n
    if last_block:
        blocks.append(last_block)
    return blocks


@st.composite
def _data_arrays(
    draw: st.DrawFn,
    *,
    dims: st.SearchStrategy[
        Sequence[Hashable] | Mapping[Hashable, int]
    ] = xrst.dimension_sizes(),
    dtype: st.SearchStrategy[np.dtype] = xrst.supported_dtypes(),
    attrs: st.SearchStrategy[Mapping] = attrs(),
    name: st.SearchStrategy[Hashable] = xrst.names(),
) -> xr.DataArray:
    return xr.DataArray(
        data=draw(xrst.variables(dims=dims, dtype=dtype, attrs=attrs)),
        name=draw(name),
    )


@st.composite
def data_arrays(
    draw: st.DrawFn,
    *,
    dims: st.SearchStrategy[
        Sequence[Hashable] | Mapping[Hashable, int]
    ] = xrst.dimension_sizes(),
    dtype: st.SearchStrategy[np.dtype] = xrst.supported_dtypes(),
    attrs: st.SearchStrategy[Mapping] = attrs(),
    name: st.SearchStrategy[Hashable] = xrst.names(),
) -> xr.DataArray:
    """Generate an arbitrary DataArray.

    Each dimension of the array will be a dimension-coordinate.
    """
    da = draw(_data_arrays(dims=dims, dtype=dtype, attrs=attrs, name=name))
    dim_coords = draw(coordinates(dims=st.just(da.sizes)))
    return da.assign_coords(dim_coords)


@st.composite
def coordinates(
    draw: st.DrawFn,
    *,
    dims: st.SearchStrategy[Mapping[Hashable, int]] = xrst.dimension_sizes(),
) -> dict[Hashable, xr.DataArray]:
    """Generate a mapping of coordindate names to DataArrays."""
    return {
        name: draw(
            _data_arrays(dims=st.just({name: size}), name=st.just(name))
        )
        for name, size in draw(dims).items()
    }


@st.composite
def data_variables(
    draw: st.DrawFn,
    *,
    dims: st.SearchStrategy[Mapping[Hashable, int]] = xrst.dimension_sizes(),
) -> dict[Hashable, xr.DataArray]:
    """Generate a mapping of data variable names to DataArrays."""
    return {
        name: draw(
            _data_arrays(dims=st.just({name: size}), name=st.just(name))
        )
        for name, size in draw(dims).items()
    }


@st.composite
def datasets(
    draw: st.DrawFn,
    *,
    data_vars: st.SearchStrategy[Mapping[Hashable, xr.DataArray]]
    | None = None,
    coords: st.SearchStrategy[Mapping[Hashable, xr.DataArray]] | None = None,
    attrs: st.SearchStrategy[Mapping] = attrs(),
) -> xr.Dataset:
    """Generate an arbitrary Dataset."""
    if data_vars is None and coords is None:
        return xr.Dataset(data_vars=draw(data_variables()), attrs=draw(attrs))
    return xr.Dataset(
        data_vars=draw(data_variables),
        coords=draw(coordinates),
        attrs=draw(attrs),
    )
