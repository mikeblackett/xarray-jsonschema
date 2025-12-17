# API reference

```{eval-rst}
..  currentmodule:: xarray_jsonschema
```

## Base schema

```{eval-rst}
..  autosummary::
    :toctree: generated/

    XarraySchema
```

## Component schema

```{eval-rst}
..  autosummary::
    :toctree: generated/

    AttrSchema
    AttrsSchema
    CoordsSchema
    DataVarsSchema
    DimsSchema
    DTypeSchema
    NameSchema
    ShapeSchema
    SizeSchema
```

## DataArraySchema

```{eval-rst}
..  autosummary::
    :toctree: generated/

    DataArraySchema
```

### Attributes

```{eval-rst}
..  autosummary::
    :toctree: generated/

    DataArraySchema.json
    DataArraySchema.validator
```

### Methods

```{eval-rst}
..  autosummary::
    :toctree: generated/

    DataArraySchema.check_schema
    DataArraySchema.dumps
    DataArraySchema.validate
```

## DatasetSchema

```{eval-rst}
..  autosummary::
    :toctree: generated/

    DatasetSchema
```

### Attributes

```{eval-rst}
..  autosummary::
    :toctree: generated/

    DatasetSchema.json
    DatasetSchema.validator
```

### Methods

```{eval-rst}
..  autosummary::
    :toctree: generated/

    DatasetSchema.check_schema
    DatasetSchema.dumps
    DatasetSchema.validate
```