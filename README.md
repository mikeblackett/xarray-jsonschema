# xarray-jsonschema

JSON Schema validation for Xarray objects.

> [!WARNING]  
> This package is in an early stage of development. Frequent and breaking changes are expected.

`xarray-jsonschema` is a re-implementation of [xarray-schema](https://xarray-schema.readthedocs.io/en/latest) that uses [JSON Schema](https://json-schema.org/) for validation.

## Motivation

I needed to validate xarray objects produced in an ETL pipeline, but none of the existing tools had all the features that I required. I took inspiration from the [xarray-schema](https://github.com/xarray-contrib/xarray-schema) and [xarray-validate](https://github.com/leroyvn/xarray-validate) packages and created `xarray-jsonschema`.

## Installation

`xarray-jsonschema` is still in development.

You can install it from source:

```bash
pip install git+https://github.com/mikeblackett/xarray-jsonschema
```

Or for development:

```bash
mkdir xarray-jsonschema
cd xarray-jsonschema
git clone https://github.com/mikeblackett/xarray-jsonschema
pip install -e .[dev]
```

## Contributing

Contributions are encouraged! Please feel free to submit a Pull Request.

## Next steps

Check out the [Docs](https://xarray-jsonschema.readthedocs.io/en/latest/)
