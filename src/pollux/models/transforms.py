"""
TODO: fix the docstrings and typing?
"""

__all__ = [
    "AbstractSingleTransform",
    "AbstractTransform",
    "AffineTransform",
    "EquinoxNNTransform",
    "FunctionTransform",
    "LinearTransform",
    "NoOpTransform",
    "OffsetTransform",
    "PolyFeatureTransform",
    "QuadraticTransform",
    "TransformSequence",
]

import abc
import inspect
from dataclasses import dataclass
from itertools import combinations_with_replacement
from math import comb
from typing import Any, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from xmmutablemap import ImmutableMap

from ..exceptions import ModelValidationError
from ..typing import (
    BatchedLatentsT,
    BatchedOutputT,
    LatentsT,
    LinearT,
    OutputT,
    QuadT,
    TransformFuncT,
)


@dataclass(frozen=True)
class ShapeSpec:
    """Represents a shape that may contain named dimensions."""

    dims: tuple[str | int, ...]

    def resolve(self, dim_sizes: dict[str, int | None]) -> tuple[int, ...]:
        """Convert named dimensions to concrete sizes.

        Uses the provided dimension size mappings to convert any string dimension
        names to their integer sizes.
        """
        dim_sizes = dim_sizes.copy()
        dim_sizes["one"] = 1
        return tuple(
            dim_sizes[d] if isinstance(d, str) and dim_sizes[d] is not None else d  # type: ignore[misc]
            for d in self.dims
        )


ParamPriorsT: TypeAlias = ImmutableMap[str, dist.Distribution]
ParamShapesT: TypeAlias = ImmutableMap[str, ShapeSpec | tuple[int, ...]]

# Tuples of parameters for TransformSequence
ParamPriorsTupleT: TypeAlias = tuple[ParamPriorsT, ...]
ParamShapesTupleT: TypeAlias = tuple[ParamShapesT, ...]


class AbstractTransform(eqx.Module):
    """Base class defining the transform interface.

    Transforms convert latent vectors to observable quantities through parameterized
    functions. They define the mapping between latent space and output spaces.
    """

    output_size: int

    # TODO: param_priors and param_shapes must be defined on any abstract subclass, but
    # there is no way to define an abstract class property
    # param_priors: ParamPriorsT | ParamPriorsTupleT
    # param_shapes: ParamShapesT | ParamShapesTupleT

    @abc.abstractmethod
    def apply(self, latents: BatchedLatentsT, **pars: Any) -> BatchedOutputT:
        """Apply the transform to input latent vectors.

        Takes a batch of latent vectors and transforms them using the provided
        parameters to produce output values.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_expanded_priors(
        self, latent_size: int, data_size: int | None = None
    ) -> ParamPriorsT:
        """Get expanded parameter priors.

        Expands the parameter prior distributions to the concrete shapes needed
        for the transform, based on latent size and optional data size.
        """
        raise NotImplementedError


class AbstractSingleTransform(AbstractTransform):
    """Base class providing common functionality for atomic transforms.

    "Single" transforms apply a single operation to convert latent vectors to outputs.
    """

    param_priors: ParamPriorsT = eqx.field(converter=ImmutableMap)
    param_shapes: ParamShapesT = eqx.field(converter=ImmutableMap)
    transform: TransformFuncT

    _param_names: tuple[str, ...] = eqx.field(init=False, repr=False)
    _transform: TransformFuncT = eqx.field(init=False, repr=False)
    vmap: bool = True

    def __post_init__(self) -> None:
        """Initialize transform parameters after object creation.

        Extracts parameter names from the transform function signature and sets up
        vectorized application if requested.
        """
        sig = inspect.signature(self.transform)
        self._param_names = tuple(sig.parameters.keys())[1:]  # skip first (latents)

        # Validate that parameter names don't contain colons (reserved for internal use)
        for param_name in self._param_names:
            if ":" in param_name:
                msg = (
                    f"Transform parameter name '{param_name}' contains ':' which is "
                    "reserved for internal parameter naming. Please rename this parameter."
                )
                raise ValueError(msg)

        # Also validate param_priors and param_shapes keys
        for param_name in self.param_priors:
            if ":" in param_name:
                msg = (
                    f"Parameter prior name '{param_name}' contains ':' which is "
                    "reserved for internal parameter naming. Please rename this parameter."
                )
                raise ValueError(msg)
        for param_name in self.param_shapes:
            if ":" in param_name:
                msg = (
                    f"Parameter shape name '{param_name}' contains ':' which is "
                    "reserved for internal parameter naming. Please rename this parameter."
                )
                raise ValueError(msg)

        # Set up vmap'd transform
        self._transform = (
            jax.vmap(self.transform, in_axes=(0, *([None] * len(self._param_names))))
            if self.vmap
            else self.transform
        )

    def apply(self, latents: BatchedLatentsT, **pars: Any) -> BatchedOutputT:
        """Apply the transform to input latent vectors.

        Extracts the required parameters from the kwargs and applies the transform
        function to the latents, handling vectorization automatically.
        """
        try:
            arg_pars = tuple(pars[p] for p in self._param_names)
        except KeyError as e:
            msg = f"Missing parameters: {self._param_names}"
            raise RuntimeError(msg) from e
        return self._transform(latents, *arg_pars)

    def get_expanded_priors(
        self, latent_size: int, data_size: int | None = None
    ) -> ParamPriorsT:
        """Get expanded parameter priors.

        Expands the parameter prior distributions to the concrete shapes needed
        for the transform, based on latent size and optional data size.
        """
        priors = {}
        for name, prior in self.param_priors.items():
            if name in self.param_shapes:
                shapespec = self.param_shapes[name]
                shape = (
                    shapespec.resolve(
                        {
                            "output_size": self.output_size,
                            "latent_size": latent_size,
                            "data_size": data_size,
                        }
                    )
                    if isinstance(shapespec, ShapeSpec)
                    else shapespec
                )
                priors[name] = prior.expand(shape)
            else:
                priors[name] = prior
        return ImmutableMap(**priors)

    def unpack_pars(
        self, flat_pars: dict[str, Any], ignore_missing: bool = False
    ) -> dict[str, Any]:  # TODO: fix Any types
        """For compatibility with TransformSequence."""
        for param_name in self._param_names:
            if param_name not in flat_pars and not ignore_missing:
                msg = f"Missing value in transform: {param_name}"
                raise ValueError(msg)
        return flat_pars

    def pack_pars(
        self, nested_pars: dict[str, Any], ignore_missing: bool = False
    ) -> dict[str, Any]:  # TODO: fix Any types
        """For compatibility with TransformSequence."""
        for param_name in self._param_names:
            if param_name not in nested_pars and not ignore_missing:
                msg = f"Missing value in transform: {param_name}"
                raise ValueError(msg)
        return nested_pars


class TransformSequence(AbstractTransform):
    """A sequence of transforms applied in order.

    Composes multiple transforms together, where the output of each transform
    becomes the input to the next transform in the sequence.

    Parameters are stored as tuples of dictionaries, one element per transform.
    """

    transforms: tuple[AbstractSingleTransform, ...]

    def __init__(self, transforms: tuple[AbstractSingleTransform, ...]):
        """Initialize a sequence of transforms."""
        if not transforms:
            msg = "At least one transform required"
            raise ModelValidationError(msg)

        # Set output size to the final transform's output size
        self.output_size = transforms[-1].output_size
        self.transforms = transforms

    @property
    def param_priors(self) -> ParamPriorsTupleT:
        """Collect parameter priors from all transforms in the sequence."""
        return tuple(
            getattr(transform, "param_priors", ImmutableMap())
            for transform in self.transforms
        )

    @property
    def param_shapes(self) -> ParamShapesTupleT:
        """Collect parameter shapes from all transforms in the sequence."""
        return tuple(
            getattr(transform, "param_shapes", ImmutableMap())
            for transform in self.transforms
        )

    def apply(
        self, latents: BatchedLatentsT, *args: dict[str, Any], **kwargs: Any
    ) -> BatchedOutputT:
        """Apply the sequence of transforms to input latent vectors.

        Parameters can be provided in two ways:
        1. As positional arguments: One dictionary per transform in sequence order
        2. As keyword arguments: Using "{transform_index}:{param}" naming scheme, so a
           parameter named "A" in transform 0 of the sequence would be "0:A".

        Parameters
        ----------
        latents
            Input latent vectors
        *args
            Parameter dictionaries, one per transform in the sequence
        **kwargs
            Flat parameters using the new naming scheme "{transform_index}:{param_name}"
        """
        # Check if using positional parameter dictionaries
        if args:
            if len(args) != len(self.transforms):
                msg = (
                    f"Expected {len(self.transforms)} parameter dictionaries, "
                    f"got {len(args)}"
                )
                raise ValueError(msg)

            if kwargs:
                msg = "Cannot mix positional parameter dicts with keyword parameters"
                raise ValueError(msg)

            output = latents
            for transform, transform_pars in zip(self.transforms, args):
                output = transform.apply(output, **transform_pars)
            return output

        # Handle flat format with "{index}:{param}" naming
        transform_pars_list: list[dict[str, Any]] = [{} for _ in self.transforms]

        for param_name, param_value in kwargs.items():
            if ":" in param_name:
                # New format: "0:A", "1:p1", etc.
                idx_str, actual_param_name = param_name.split(":", 1)
                transform_idx = int(idx_str)
                if not 0 <= transform_idx < len(self.transforms):
                    msg = f"Invalid transform index: {transform_idx}"
                    raise ValueError(msg)
                transform_pars_list[transform_idx][actual_param_name] = param_value

            else:
                # Handle any other parameter format as needed
                msg = f"Unsupported parameter name format: {param_name}"
                raise ValueError(msg)

        output = latents
        for transform, transform_pars in zip(self.transforms, transform_pars_list):
            output = transform.apply(output, **transform_pars)
        return output

    def unpack_pars(
        self, flat_pars: dict[str, Any], ignore_missing: bool = False
    ) -> tuple[dict[str, Any], ...]:
        """Convert flat parameter names to nested tuple structure.

        Takes parameters with names like "0:A", "1:p1" and converts them to
        a list of parameter dictionaries: [{"A": value}, {"p1": value}]

        Parameters
        ----------
        flat_pars
            Dictionary with parameter names in format "{transform_index}:{param_name}"

        Returns
        -------
        list
            List of parameter dictionaries, one per transform in the sequence
        """
        nested_pars: list[dict[str, Any]] = [{} for _ in self.transforms]

        for param_name in self.names_flat:
            param_value = flat_pars.get(param_name)

            if param_value is None:
                if not ignore_missing:
                    msg = f"Missing value in transform: {param_name}"
                    raise ValueError(msg)
                # Skip missing parameters when ignore_missing=True
                continue

            if ":" in param_name:
                idx_str, actual_param_name = param_name.split(":", 1)
                transform_idx = int(idx_str)
                if 0 <= transform_idx < len(self.transforms):
                    nested_pars[transform_idx][actual_param_name] = param_value

        return tuple(nested_pars)

    def pack_pars(
        self, nested_pars: list[dict[str, Any]], ignore_missing: bool = False
    ) -> dict[str, Any]:
        """Convert nested parameter structure to flat naming scheme.

        Takes a list of parameter dictionaries and converts them to flat
        parameter names like "0:A", "1:p1".

        Parameters
        ----------
        nested_pars
            List of parameter dictionaries, one per transform in the sequence
        ignore_missing
            If True, skip missing parameters instead of raising an error.
            Currently unused but kept for API consistency.

        Returns
        -------
        dict
            Dictionary with parameter names in format "{transform_index}:{param_name}"
        """
        # TODO: ignore_missing is not used here...?
        _ = ignore_missing

        flat_pars = {}

        for i, transform_pars in enumerate(nested_pars):
            for param_name, param_value in transform_pars.items():
                flat_name = f"{i}:{param_name}"
                flat_pars[flat_name] = param_value

        return flat_pars

    @property
    def names_nested(self) -> tuple[tuple[str, ...], ...]:
        return tuple(t._param_names for t in self.transforms)

    @property
    def names_flat(self) -> tuple[str, ...]:
        return tuple(
            f"{i}:{name}" for i, names in enumerate(self.names_nested) for name in names
        )

    def get_expanded_priors(
        self, latent_size: int, data_size: int | None = None
    ) -> ParamPriorsT:
        """Get expanded parameter priors using flat naming scheme.

        Returns flattened parameter priors with index-based naming for
        compatibility with the AbstractTransform interface.
        Parameter names will be in the format: "{transform_index}:{param_name}"

        Note: For transform sequences, each transform's "latent_size" is the
        output size of the previous transform (or the model's latent_size for
        the first transform).
        """
        priors = {}
        current_size = latent_size

        for i, transform in enumerate(self.transforms):
            transform_priors = transform.get_expanded_priors(
                latent_size=current_size, data_size=data_size
            )
            for param_name, prior in transform_priors.items():
                flat_name = f"{i}:{param_name}"
                priors[flat_name] = prior

            # Update current_size for the next transform
            # Check if transform has get_output_size method (like PolyFeatureTransform)
            get_output_size = getattr(transform, "get_output_size", None)
            if callable(get_output_size):
                current_size = get_output_size(current_size)
            else:
                current_size = transform.output_size

        return ImmutableMap(**priors)


class FunctionTransform(AbstractSingleTransform):
    """Function transformation using a user-defined function.

    This transform allows for arbitrary transformations defined by the user.

    Examples
    --------
    TODO: add in quadrature
    """


# ----


def _noop_transform(z: LatentsT) -> OutputT:
    """No-op transformation."""
    return z


class NoOpTransform(AbstractSingleTransform):
    """No-op transformation."""

    output_size: int = 0
    transform: TransformFuncT = _noop_transform
    param_priors: ParamPriorsT = ImmutableMap()
    param_shapes: ParamShapesT = ImmutableMap()


# ----


def _compute_n_poly_features(n_inputs: int, degree: int, include_bias: bool) -> int:
    """Compute the number of polynomial features.

    For n inputs with degree d, the number of features is C(n+d, d) - 1 (if no bias)
    or C(n+d, d) (if bias included).

    Parameters
    ----------
    n_inputs
        Number of input features.
    degree
        Maximum polynomial degree.
    include_bias
        Whether to include a bias term (constant 1).

    Returns
    -------
    int
        Number of polynomial features.
    """
    # Total monomials of degree <= d with n variables is C(n+d, d)
    n_features = comb(n_inputs + degree, degree)
    if not include_bias:
        n_features -= 1  # Remove the constant term
    return n_features


def polynomial_features(
    x: BatchedLatentsT, degree: int = 2, include_bias: bool = True
) -> BatchedOutputT:
    """Expand input into polynomial features.

    Generates all polynomial combinations of features up to the specified degree.
    For inputs [x1, x2] with degree=2 and include_bias=True, produces:
    [1, x1, x2, x1^2, x1*x2, x2^2]

    Parameters
    ----------
    x
        Input array of shape (n_samples, n_features).
    degree
        Maximum polynomial degree. Default is 2.
    include_bias
        Whether to include a bias column of ones. Default is True.

    Returns
    -------
    array
        Polynomial features of shape (n_samples, n_poly_features).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    >>> polynomial_features(x, degree=2, include_bias=True)
    Array([[ 1.,  1.,  2.,  1.,  2.,  4.],
           [ 1.,  3.,  4.,  9., 12., 16.]], dtype=float32)
    """
    n_samples, n_features = x.shape

    # Generate all monomials: for each degree d, generate all combinations
    # of d indices (with replacement) from 0 to n_features-1
    columns = []

    for d in range(degree + 1):
        if d == 0:
            if include_bias:
                columns.append(jnp.ones((n_samples, 1), dtype=x.dtype))
        else:
            for indices in combinations_with_replacement(range(n_features), d):
                # Compute the product of features at these indices
                col = jnp.ones(n_samples, dtype=x.dtype)
                for idx in indices:
                    col = col * x[:, idx]
                columns.append(col[:, None])

    return jnp.concatenate(columns, axis=1)


def _poly_feature_transform(z: LatentsT, degree: int, include_bias: bool) -> OutputT:
    """Transform function for polynomial features (single sample)."""
    # This will be called via vmap, so z is shape (n_features,)
    # We need to add a batch dimension, apply, and remove it
    return polynomial_features(z[None, :], degree, include_bias)[0]


class PolyFeatureTransform(AbstractTransform):
    """Polynomial feature expansion transform.

    Expands input features into polynomial combinations up to the specified degree.
    This transform has NO learnable parameters - it's a deterministic feature expansion.

    This is useful for implementing The Cannon model, where labels are expanded into
    polynomial features before a linear transformation to predict spectra.

    Parameters
    ----------
    degree
        Maximum polynomial degree. Default is 2.
    include_bias
        Whether to include a bias term (constant 1). Default is True.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from pollux.models.transforms import PolyFeatureTransform, LinearTransform
    >>> from pollux.models.transforms import TransformSequence

    Create a Cannon-style transform (polynomial features -> linear):

    >>> cannon = TransformSequence((
    ...     PolyFeatureTransform(degree=2),
    ...     LinearTransform(output_size=128),
    ... ))

    The polynomial transform expands 3 labels into 10 features (with bias):
    - degree 0: 1 (bias)
    - degree 1: x1, x2, x3
    - degree 2: x1^2, x1*x2, x1*x3, x2^2, x2*x3, x3^2
    """

    degree: int = 2
    include_bias: bool = True

    # No learnable parameters
    param_priors: ParamPriorsT = ImmutableMap()
    param_shapes: ParamShapesT = ImmutableMap()

    # output_size is computed dynamically based on input size
    # We set a placeholder that will be overridden in apply()
    output_size: int = eqx.field(default=0)

    def apply(self, latents: BatchedLatentsT, **_pars: Any) -> BatchedOutputT:
        """Apply polynomial feature expansion.

        Parameters
        ----------
        latents
            Input array of shape (n_samples, n_features).
        **_pars
            Ignored (no learnable parameters).

        Returns
        -------
        array
            Polynomial features of shape (n_samples, n_poly_features).
        """
        return polynomial_features(latents, self.degree, self.include_bias)

    def get_expanded_priors(
        self, latent_size: int, data_size: int | None = None
    ) -> ParamPriorsT:
        """Return empty priors (no learnable parameters)."""
        del latent_size, data_size  # Unused - no learnable parameters
        return ImmutableMap()

    def get_output_size(self, input_size: int) -> int:
        """Compute output size given input size.

        Parameters
        ----------
        input_size
            Number of input features.

        Returns
        -------
        int
            Number of polynomial features.
        """
        return _compute_n_poly_features(input_size, self.degree, self.include_bias)

    def unpack_pars(
        self, _flat_pars: dict[str, Any], _ignore_missing: bool = False
    ) -> dict[str, Any]:
        """For compatibility with TransformSequence (returns empty dict)."""
        return {}

    def pack_pars(
        self, _nested_pars: dict[str, Any], _ignore_missing: bool = False
    ) -> dict[str, Any]:
        """For compatibility with TransformSequence (returns empty dict)."""
        return {}


# ----


def _linear_transform(z: LatentsT, A: LinearT) -> OutputT:
    """Apply a linear transformation.

    Computes the matrix product A @ z.
    """
    return A @ z


class LinearTransform(AbstractSingleTransform):
    """Linear transformation from latent to output space.

    Implements the transformation: y = A @ z, where A is a matrix and z is a latent
    vector.
    """

    transform: TransformFuncT = _linear_transform
    param_priors: ParamPriorsT = eqx.field(
        default=ImmutableMap({"A": dist.Normal(0, 1)}),
        converter=ImmutableMap,
    )
    param_shapes: ParamShapesT = ImmutableMap(
        {"A": ShapeSpec(("output_size", "latent_size"))}
    )


# ----


def _offset_transform(z: LatentsT, b: OutputT) -> OutputT:
    """Apply an offset transformation.

    Adds a bias vector b to the input: z + b.
    """
    return z + b


class OffsetTransform(AbstractSingleTransform):
    """Offset transformation that adds a bias vector to inputs.

    Implements the transformation: y = z + b, where b is a bias vector.
    """

    transform: TransformFuncT = _offset_transform
    param_priors: ParamPriorsT = eqx.field(
        default=ImmutableMap({"b": dist.Normal(0, 1)}),
        converter=ImmutableMap,
    )
    param_shapes: ParamShapesT = ImmutableMap({"b": ShapeSpec(("output_size", "one"))})


# ----


def _affine_transform(z: LatentsT, A: LinearT, b: OutputT) -> OutputT:
    """Apply an affine transformation.

    Computes a linear transformation followed by an offset: A @ z + b.
    """
    return A @ z + b


class AffineTransform(AbstractSingleTransform):
    """Affine transformation combining linear transform and offset.

    Implements the transformation: y = A @ z + b, where A is a matrix,
    z is a latent vector, and b is a bias vector.
    """

    transform: TransformFuncT = _affine_transform
    param_priors: ParamPriorsT = eqx.field(
        default=ImmutableMap({"A": dist.Normal(0, 1), "b": dist.Normal(0, 1)}),
        converter=ImmutableMap,
    )
    param_shapes: ParamShapesT = ImmutableMap(
        {
            "A": ShapeSpec(("output_size", "latent_size")),
            "b": ShapeSpec(("output_size", "one")),
        }
    )


# ----


def _quadratic_transform(z: LatentsT, Q: QuadT, A: LinearT, b: OutputT) -> OutputT:
    """Apply a quadratic transformation.

    Computes a quadratic form plus a linear term and an offset: z^T Q z + A @ z + b.
    """
    return jnp.einsum("i,oij,j->o", z, Q, z) + A @ z + b


class QuadraticTransform(AbstractSingleTransform):
    """Quadratic transformation of latent vectors.

    Implements the transformation: y = z^T Q z + A @ z + b, where Q is a tensor,
    A is a matrix, z is a latent vector, and b is a bias vector.
    """

    transform: TransformFuncT = _quadratic_transform
    param_priors: ParamPriorsT = eqx.field(
        default=ImmutableMap(
            {"Q": dist.Normal(0, 1), "A": dist.Normal(0, 1), "b": dist.Normal(0, 1)}
        ),
        converter=ImmutableMap,
    )
    param_shapes: ParamShapesT = ImmutableMap(
        {
            "Q": ShapeSpec(("output_size", "latent_size", "latent_size")),
            "A": ShapeSpec(("output_size", "latent_size")),
            "b": ShapeSpec(("output_size", "one")),
        }
    )


# ----


def _get_param_paths(
    module: eqx.Module, prefix: str = "", separator: str = "."
) -> tuple[str, ...]:
    """Generate unique string paths for all array parameters in an Equinox module.

    Traverses the PyTree structure of an Equinox module and generates a unique
    path string for each array leaf. This is used to create flat parameter names
    for numpyro sampling.

    Parameters
    ----------
    module
        An Equinox module to extract parameter paths from.
    prefix
        Prefix to prepend to all paths (used for recursion).
    separator
        Separator to use between path components. Default is ".".

    Returns
    -------
    tuple[str, ...]
        A tuple of parameter path strings, one for each array in the module.

    Examples
    --------
    >>> import equinox as eqx
    >>> import jax
    >>> key = jax.random.PRNGKey(0)
    >>> mlp = eqx.nn.MLP(in_size=4, out_size=8, width_size=16, depth=2, key=key)
    >>> paths = _get_param_paths(mlp)
    >>> print(paths[:4])  # First few paths
    ('layers.0.weight', 'layers.0.bias', 'layers.1.weight', 'layers.1.bias')
    """
    paths = []

    # Get the fields of the module
    items = module.__dict__.items() if hasattr(module, "__dict__") else []

    for name, value in items:
        # Skip private attributes
        if name.startswith("_"):
            continue

        current_path = f"{prefix}{separator}{name}" if prefix else name

        if eqx.is_array(value):
            paths.append(current_path)
        elif isinstance(value, eqx.Module):
            # Recursively process nested modules
            paths.extend(_get_param_paths(value, current_path, separator))
        elif isinstance(value, (list, tuple)):
            # Handle lists/tuples of modules (like layers in MLP)
            for i, item in enumerate(value):
                item_path = f"{current_path}{separator}{i}"
                if eqx.is_array(item):
                    paths.append(item_path)
                elif isinstance(item, eqx.Module):
                    paths.extend(_get_param_paths(item, item_path, separator))

    return tuple(paths)


def _get_flat_params(module: eqx.Module) -> list[jax.Array]:
    """Extract all array parameters from an Equinox module in a flat list.

    Parameters
    ----------
    module
        An Equinox module to extract parameters from.

    Returns
    -------
    list[jax.Array]
        A list of all array parameters in the module, in the same order as
        _get_param_paths returns their paths.
    """
    params = []

    items = module.__dict__.items() if hasattr(module, "__dict__") else []

    for name, value in items:
        if name.startswith("_"):
            continue

        if eqx.is_array(value):
            params.append(value)
        elif isinstance(value, eqx.Module):
            params.extend(_get_flat_params(value))
        elif isinstance(value, (list, tuple)):
            for item in value:
                if eqx.is_array(item):
                    params.append(item)
                elif isinstance(item, eqx.Module):
                    params.extend(_get_flat_params(item))

    return params


def _set_param_by_path(
    module: eqx.Module, path: str, value: jax.Array, separator: str = "."
) -> eqx.Module:
    """Set a parameter in an Equinox module by its path string.

    Parameters
    ----------
    module
        The Equinox module to modify.
    path
        The path to the parameter (e.g., "layers.0.weight").
    value
        The new value for the parameter.
    separator
        The separator used in the path. Default is ".".

    Returns
    -------
    eqx.Module
        A new module with the parameter updated.
    """
    parts = path.split(separator)

    def update_nested(obj: Any, parts: list[str], val: jax.Array) -> Any:
        if not parts:
            return val

        key = parts[0]
        remaining = parts[1:]

        if isinstance(obj, eqx.Module):
            # Get current value
            if hasattr(obj, key):
                current = getattr(obj, key)
                new_val = update_nested(current, remaining, val)
                return eqx.tree_at(lambda m: getattr(m, key), obj, new_val)
            msg = f"Module has no attribute '{key}'"
            raise AttributeError(msg)
        if isinstance(obj, (list, tuple)):
            idx = int(key)
            current = obj[idx]
            new_val = update_nested(current, remaining, val)
            if isinstance(obj, tuple):
                return (*obj[:idx], new_val, *obj[idx + 1 :])
            new_list = list(obj)
            new_list[idx] = new_val
            return new_list
        msg = f"Cannot traverse into type {type(obj)}"
        raise TypeError(msg)

    return update_nested(module, parts, value)


class EquinoxNNTransform(AbstractTransform):
    """Neural network transform using an Equinox module.

    This transform wraps an Equinox neural network module and exposes its parameters
    for Bayesian inference via numpyro. The network structure is defined by a factory
    function that creates the network given input size, output size, and a random key.

    Parameters
    ----------
    output_size
        The output dimension of the transform.
    nn_factory
        A callable that creates an Equinox module. It should have the signature:
        ``nn_factory(in_size: int, out_size: int, key: jax.Array) -> eqx.Module``
    weight_prior
        Prior distribution for weight parameters. Default is Normal(0, 1).
    bias_prior
        Prior distribution for bias parameters. Default is Normal(0, 1).

    Examples
    --------
    >>> import jax
    >>> import equinox as eqx
    >>> import numpyro.distributions as dist
    >>> from pollux.models.transforms import EquinoxNNTransform

    Create a simple MLP transform:

    >>> nn_trans = EquinoxNNTransform(
    ...     output_size=128,
    ...     nn_factory=lambda in_size, out_size, key: eqx.nn.MLP(
    ...         in_size=in_size,
    ...         out_size=out_size,
    ...         width_size=64,
    ...         depth=2,
    ...         key=key,
    ...     ),
    ...     weight_prior=dist.Normal(0, 0.1),
    ...     bias_prior=dist.Normal(0, 0.01),
    ... )

    Use with LuxModel:

    >>> import pollux as plx
    >>> model = plx.LuxModel(latent_size=8)
    >>> model.register_output("flux", nn_trans)
    """

    output_size: int
    nn_factory: Any  # Callable[[int, int, jax.Array], eqx.Module]
    weight_prior: dist.Distribution = eqx.field(
        default_factory=lambda: dist.Normal(0.0, 1.0)
    )
    bias_prior: dist.Distribution = eqx.field(
        default_factory=lambda: dist.Normal(0.0, 1.0)
    )

    # No static param_priors or param_shapes - these are computed dynamically
    param_priors: ParamPriorsT = eqx.field(default_factory=lambda: ImmutableMap())
    param_shapes: ParamShapesT = eqx.field(default_factory=lambda: ImmutableMap())

    # Internal state - computed in get_expanded_priors
    _param_paths: tuple[str, ...] = eqx.field(default=(), repr=False)
    _template_nn: Any = eqx.field(default=None, repr=False)  # eqx.Module

    def get_expanded_priors(
        self, latent_size: int, data_size: int | None = None
    ) -> ParamPriorsT:
        """Create one prior per neural network parameter.

        Parameters
        ----------
        latent_size
            The input size to the neural network.
        data_size
            Not used for NN transforms (included for interface compatibility).

        Returns
        -------
        ParamPriorsT
            Dictionary mapping parameter paths to expanded prior distributions.
        """
        del data_size  # Unused

        # Create a template NN to get the PyTree structure
        key = jax.random.PRNGKey(0)  # Just for structure, not actual init
        template_nn = self.nn_factory(latent_size, self.output_size, key)

        # Store template and paths for use in apply()
        # Note: We use object.__setattr__ because eqx.Module is frozen
        object.__setattr__(self, "_template_nn", template_nn)
        object.__setattr__(self, "_param_paths", _get_param_paths(template_nn))

        # Get flat parameters to determine shapes
        flat_params = _get_flat_params(template_nn)

        # Create expanded priors for each parameter
        priors = {}
        for path, param in zip(self._param_paths, flat_params):
            # Validate that path doesn't contain ":"
            if ":" in path:
                msg = (
                    f"Neural network parameter path '{path}' contains ':' which is "
                    "reserved for internal parameter naming. This may cause issues."
                )
                raise ValueError(msg)

            # Choose prior based on parameter name
            if "weight" in path.lower():
                prior = self.weight_prior
            elif "bias" in path.lower():
                prior = self.bias_prior
            else:
                # Default to weight prior for unknown parameters
                prior = self.weight_prior

            priors[path] = prior.expand(param.shape)

        return ImmutableMap(**priors)

    def apply(self, latents: BatchedLatentsT, **params: Any) -> BatchedOutputT:
        """Apply the neural network transform.

        Parameters
        ----------
        latents
            Input latent vectors of shape (n_samples, latent_size).
        **params
            Neural network parameters, keyed by their path names.

        Returns
        -------
        array
            Output of shape (n_samples, output_size).
        """
        if self._template_nn is None:
            msg = (
                "EquinoxNNTransform.get_expanded_priors() must be called before apply()"
            )
            raise RuntimeError(msg)

        # Reconstruct the NN with the provided parameters
        nn: eqx.Module = self._template_nn
        for path in self._param_paths:
            if path in params:
                nn = _set_param_by_path(nn, path, params[path])

        # Apply NN to each latent vector using vmap
        # The nn is an eqx.Module which is callable via __call__
        def forward(x: jax.Array) -> jax.Array:
            return nn(x)  # type: ignore[operator]

        return jax.vmap(forward)(latents)

    def unpack_pars(
        self, flat_pars: dict[str, Any], ignore_missing: bool = False
    ) -> dict[str, Any]:
        """Unpack flat parameters (for compatibility with TransformSequence)."""
        result = {}
        for path in self._param_paths:
            if path in flat_pars:
                result[path] = flat_pars[path]
            elif not ignore_missing:
                msg = f"Missing NN parameter: {path}"
                raise ValueError(msg)
        return result

    def pack_pars(
        self, nested_pars: dict[str, Any], ignore_missing: bool = False
    ) -> dict[str, Any]:
        """Pack parameters to flat format (for compatibility with TransformSequence)."""
        result = {}
        for path in self._param_paths:
            if path in nested_pars:
                result[path] = nested_pars[path]
            elif not ignore_missing:
                msg = f"Missing NN parameter: {path}"
                raise ValueError(msg)
        return result


# ----

# TODO: implement a Gaussian Process transform using the tinygp library. The user should specify the kernel, and parameter priors for the kernel.
# class GaussianProcessTransform(SingleTransformMixin, AbstractTransform):
#     transform: TransformFuncT = ...
