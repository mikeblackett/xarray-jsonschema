import hypothesis.strategies as st
import xarray.testing.strategies as xrst

import xarray_jsonschema as xrm
import xarray_jsonschema.testing as xmst
from xarray_jsonschema.containers import DataVarsSchema


@st.composite
def attr_models(draw: st.DrawFn):
    key = draw(xmst.readable_text())
    regex = draw(st.booleans())
    value = draw(
        st.one_of(
            st.none(),
            st.booleans(),
            st.integers(),
            st.floats(),
            xmst.readable_text(),
        )
    )
    required = draw(st.booleans())
    return xrm.AttrSchema(key, regex=regex, value=value, required=required)


@st.composite
def attrs_models(draw: st.DrawFn):
    allow_extra_items = draw(st.booleans())
    return xrm.AttrsSchema(
        draw(st.lists(elements=attr_models())),
        allow_extra_items=allow_extra_items,
    )


@st.composite
def chunks_models(draw: st.DrawFn):
    chunks = draw(
        st.one_of(
            st.booleans(),
            xmst.positive_integers(),
            st.recursive(
                base=st.lists(
                    elements=st.integers(min_value=1), min_size=1, max_size=3
                ),
                extend=st.lists,
                max_leaves=3,
            ),
        )
    )
    return xrm.ChunksSchema(chunks)


@st.composite
def dims_models(draw: st.DrawFn):
    dims = draw(st.one_of(st.none(), xrst.dimension_names()))
    contains = draw(st.one_of(st.none(), xrst.names()))
    max_dims = draw(st.one_of(st.none(), xmst.non_negative_integers()))
    min_dims = draw(st.one_of(st.none(), xmst.non_negative_integers()))
    return xrm.DimsSchema(
        dims, contains=contains, max_dims=max_dims, min_dims=min_dims
    )


@st.composite
def dtype_models(draw: st.DrawFn):
    dtype = draw(st.one_of(st.none(), xmst.supported_dtype_likes()))
    return xrm.DTypeSchema(dtype)


@st.composite
def name_models(draw: st.DrawFn):
    name = draw(st.one_of(st.none(), xrst.names()))
    regex = draw(st.booleans())
    min_length = draw(st.one_of(st.none(), xmst.non_negative_integers()))
    max_length = draw(st.one_of(st.none(), xmst.non_negative_integers()))
    return xrm.NameSchema(
        name, regex=regex, min_length=min_length, max_length=max_length
    )


@st.composite
def shape_models(draw: st.DrawFn):
    shape = draw(st.one_of(st.none(), xmst.dimension_shapes()))
    min_items = draw(st.one_of(st.none(), xmst.non_negative_integers()))
    max_items = draw(st.one_of(st.none(), xmst.non_negative_integers()))
    return xrm.ShapeSchema(shape, min_dims=min_items, max_dims=max_items)


@st.composite
def data_array_models(draw: st.DrawFn):
    """Return a DataArraySchema object with randomized component attributes."""
    attrs = draw(attrs_models())
    chunks = draw(chunks_models())
    dims = draw(dims_models())
    dtype = draw(dtype_models())
    shape = draw(shape_models())

    return xrm.DataArraySchema(
        attrs=attrs, chunks=chunks, dims=dims, dtype=dtype, shape=shape
    )


@st.composite
def coords_models(draw: st.DrawFn) -> xrm.CoordsSchema:
    coords = draw(
        st.dictionaries(keys=xrst.names(), values=data_array_models())
    )
    require_all = draw(st.booleans())
    allow_extra = draw(st.booleans())
    return xrm.CoordsSchema(
        coords,
        require_all=require_all,
        allow_extra=allow_extra,
    )


@st.composite
def dataset_models(draw: st.DrawFn):
    attrs = draw(attrs_models())
    coords = draw(coords_models())
    data_vars = draw(data_vars_models())

    return xrm.DatasetSchema(
        attrs=attrs,
        coords=coords,
        data_vars=data_vars,
    )


@st.composite
def data_vars_models(draw: st.DrawFn) -> DataVarsSchema:
    data_vars = draw(
        st.dictionaries(keys=xrst.names(), values=data_array_models())
    )
    require_all = draw(st.booleans())
    allow_extra = draw(st.booleans())
    return xrm.DataVarsSchema(
        data_vars,
        require_all=require_all,
        allow_extra=allow_extra,
    )
