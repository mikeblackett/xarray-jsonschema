import hypothesis.strategies as st
from xarray.testing import strategies as xrst
import xarray_jsonschema as xrm
import xarray_jsonschema.testing as xmst


@st.composite
def attr_schemas(draw: st.DrawFn):
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
    regex = draw(st.booleans())
    return xrm.AttrSchema(value, regex=regex, required=required)


@st.composite
def attrs_schemas(draw: st.DrawFn):
    strict = draw(st.booleans())
    return xrm.AttrsSchema(
        draw(
            st.dictionaries(
                keys=xmst.readable_text(),
                values=attr_schemas(),
            )
        ),
        strict=strict,
    )


# @st.composite
# def chunks_schemas(draw: st.DrawFn):
#     chunks = draw(
#         st.one_of(
#             st.booleans(),
#             xmst.positive_integers(),
#             st.recursive(
#                 base=st.lists(
#                     elements=st.integers(min_value=1), min_size=1, max_size=3
#                 ),
#                 extend=st.lists,
#                 max_leaves=3,
#             ),
#         )
#     )
#     return xrm.ChunksSchema(chunks)


@st.composite
def dims_schemas(draw: st.DrawFn):
    dims = draw(st.one_of(st.none(), xrst.dimension_names()))
    contains = draw(st.one_of(st.none(), xrst.names()))
    max_dims = draw(st.one_of(st.none(), xmst.non_negative_integers()))
    min_dims = draw(st.one_of(st.none(), xmst.non_negative_integers()))
    return xrm.DimsSchema(
        dims, contains=contains, max_dims=max_dims, min_dims=min_dims
    )


@st.composite
def dtype_schemas(draw: st.DrawFn):
    dtype = draw(st.one_of(st.none(), xmst.supported_dtype_likes()))
    return xrm.DTypeSchema(dtype)


@st.composite
def name_schemas(draw: st.DrawFn):
    name = draw(st.one_of(st.none(), xrst.names()))
    regex = draw(st.booleans())
    min_length = draw(st.one_of(st.none(), xmst.non_negative_integers()))
    max_length = draw(st.one_of(st.none(), xmst.non_negative_integers()))
    return xrm.NameSchema(
        name, regex=regex, min_length=min_length, max_length=max_length
    )


@st.composite
def shape_schemas(draw: st.DrawFn):
    shape = draw(st.one_of(st.none(), xmst.dimension_shapes()))
    min_items = draw(st.one_of(st.none(), xmst.non_negative_integers()))
    max_items = draw(st.one_of(st.none(), xmst.non_negative_integers()))
    return xrm.ShapeSchema(shape, min_dims=min_items, max_dims=max_items)


@st.composite
def data_array_schemas(draw: st.DrawFn):
    """Return a DataArraySchema object with randomized component attributes."""
    attrs = draw(attrs_schemas())
    coords = draw(coords_schemas())
    # chunks = draw(chunks_schemas())
    dims = draw(dims_schemas())
    dtype = draw(dtype_schemas())
    shape = draw(shape_schemas())

    return xrm.DataArraySchema(
        attrs=attrs, coords=coords, dims=dims, dtype=dtype, shape=shape
    )


@st.composite
def coords_schemas(draw: st.DrawFn) -> xrm.CoordsSchema:
    coords = draw(
        st.dictionaries(keys=xrst.names(), values=data_array_schemas())
    )
    strict = draw(st.booleans())
    return xrm.CoordsSchema(
        coords,
        strict=strict,
    )


#
#
# @st.composite
# def dataset_schemas(draw: st.DrawFn):
#     attrs = draw(attrs_schemas())
#     coords = draw(coords_schemas())
#     data_vars = draw(data_vars_schemas())
#
#     return xrm.DatasetSchema(
#         attrs=attrs,
#         coords=coords,
#         data_vars=data_vars,
#     )
#
#
# @st.composite
# def data_vars_schemas(draw: st.DrawFn) -> DataVarsSchema:
#     data_vars = draw(
#         st.dictionaries(keys=xrst.names(), values=data_array_schemas())
#     )
#     require_all_items = draw(st.booleans())
#     allow_extra_items = draw(st.booleans())
#     return xrm.DataVarsSchema(
#         data_vars,
#         require_all_items=require_all_items,
#         allow_extra_items=allow_extra_items,
#     )
