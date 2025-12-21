from typing import Sequence

import hypothesis as hp
import hypothesis.extra.numpy as hn
import pytest as pt

from xarray_jsonschema import ShapeModel, ValidationError


class TestShape:
    @hp.given(expected=hn.array_shapes(max_side=5))
    def test_generates_valid_schema(self, expected: Sequence) -> None:
        """Should produce valid JSON Model."""
        model = ShapeModel(expected)
        assert ShapeModel.check_schema(model.to_schema()) is None

    def test_default_value(self) -> None:
        """Should produce a default schema if no attrs are provided."""
        model = ShapeModel()
        assert model.to_schema() == {
            '$schema': 'https://json-schema.org/draft/2020-12/schema',
            'type': 'array',
        }

    @hp.given(expected=hn.array_shapes(max_side=5))
    def test_argument_is_not_kw_only(self, expected: Sequence) -> None:
        assert ShapeModel(expected) == ShapeModel(shape=expected)

    @hp.given(expected=hn.array_shapes(max_side=5))
    def test_validation(self, expected: Sequence) -> None:
        """Should pass if the instance shape matches the expected sequence."""
        ShapeModel(expected).validate(expected)

    @hp.given(
        expected=hn.array_shapes(max_side=5),
        actual=hn.array_shapes(max_side=5),
    )
    def test_invalidation(self, expected: Sequence, actual: Sequence) -> None:
        """Should fail if the instance shape does not match the expected sequence."""
        hp.assume(actual != expected)
        with pt.raises(ValidationError):
            ShapeModel(expected).validate(actual)
