import re
from collections.abc import Mapping
from typing import Any

import numpy as np
import xarray.testing.strategies as xrst
from hypothesis import strategies as st

REGEX_PATTERNS = [
    r'[A-Za-z]{3,10}',  # Letters only, 3-10 chars
    r'\d{2,4}',  # 2-4 digits
    r'[A-Z][a-z]+\d{2}',  # Capitalized word with 2 digits
    r'[a-z]+[-_][a-z]+',  # Two lowercase words with dash/underscore
    r'[A-Z0-9]{8}',  # 8 chars of uppercase letters and numbers
    r'^[a-z]+_\d{3}$',  # Lowercase word with 3 digits suffix
    r'^\d{2}-[A-Z]{3}$',  # 2 digits followed by 3 uppercase letters
    r'^[A-Z][a-z]+$',  # Single capitalized word
    r'^\d{4}-[A-Z]{2}$',  # 4 digits followed by 2 uppercase letters
    r'^v\d+\.\d+$',  # Version number format
]


@st.composite
def patterns(draw: st.DrawFn) -> re.Pattern:
    """Generate regular expression patterns."""
    return re.compile(draw(st.sampled_from(REGEX_PATTERNS)))


_readable_characters = st.characters(
    categories=['L', 'N'], max_codepoint=0x017F
)


def readable_text(min_length=0, max_length=5):
    return st.text(_readable_characters, min_size=1, max_size=5)


@st.composite
def supported_dtype_likes(
    draw: st.DrawFn,
    dtype: np.dtype | None = None,
) -> np.dtype | str | type | None:
    """Generate supported dtype-like objects."""
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


_attr_values = st.one_of(
    st.none(),
    st.booleans(),
    st.floats(),
    st.integers(),
    readable_text(),
)


def attrs(
    min_items: int = 0, max_leaves: int = 3
) -> st.SearchStrategy[Mapping[str, Any]]:
    """Generate nested attribute mappings"""
    return st.recursive(
        base=st.dictionaries(
            keys=readable_text(), values=_attr_values, min_size=min_items
        ),
        extend=lambda children: st.dictionaries(
            keys=readable_text(), values=children, min_size=min_items
        ),
        max_leaves=max_leaves,
    )
