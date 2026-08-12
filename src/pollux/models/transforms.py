"""Transforms for mapping latent vectors to output quantities."""

__all__ = [
    "AbstractMultiTransform",
    "AbstractSingleTransform",
    "AbstractTransform",
    "AdditiveOffsetTransform",
    "AffineTransform",
    "ConcatenateTransform",
    "EquinoxNNTransform",
    "FunctionTransform",
    "LinearTransform",
    "NoOpTransform",
    "OffsetTransform",
    "ParamPriorsT",
    "ParamShapesT",
    "PolyFeatureTransform",
    "QuadraticTransform",
    "ShapeT",
    "TransformSequence",
]

import abc
import inspect
from itertools import combinations_with_replacement
from math import comb
from typing import Any, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
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

#: Named dimensions usable in a parameter shape, resolved when the model is built:
#: ``"output_size"`` (the transform's output dimension), ``"latent_size"`` (the
#: latent space dimension, set when registering with a model), ``"data_size"``
#: (the number of objects in the dataset) and ``"one"`` (always 1, for bias terms).
ShapeT: TypeAlias = tuple[str | int, ...]


def _resolve_shape(shape: ShapeT, **dim_sizes: int | None) -> tuple[int, ...]:
    """Convert any named dimensions in a shape to concrete sizes.

    Examples
    --------
    >>> from pollux.models.transforms import _resolve_shape
    >>> _resolve_shape(("output_size", "latent_size"), output_size=128, latent_size=8)
    (128, 8)
    >>> _resolve_shape(("output_size", "one"), output_size=128)
    (128, 1)
    """
    sizes: dict[str, int | None] = {"one": 1, **dim_sizes}
    return tuple(
        sizes[d] if isinstance(d, str) and sizes[d] is not None else d  # type: ignore[misc]
        for d in shape
    )


#: Type alias for parameter priors: maps parameter names to distributions.
#: Used with :class:`FunctionTransform` to specify priors for learnable parameters.
ParamPriorsT: TypeAlias = ImmutableMap[str, dist.Distribution]

#: Type alias for parameter shapes: maps parameter names to shapes. Each shape is
#: a tuple of concrete sizes and/or named dimensions (see :data:`ShapeT`).
ParamShapesT: TypeAlias = ImmutableMap[str, ShapeT]

# Internal: Tuples of parameters for TransformSequence
ParamPriorsTupleT: TypeAlias = tuple[ParamPriorsT, ...]
ParamShapesTupleT: TypeAlias = tuple[ParamShapesT, ...]


class AbstractTransform(eqx.Module):
    """Base class defining the transform interface.

    Transforms convert latent vectors to observable quantities through parameterized
    functions. They define the mapping between latent space and output spaces.
    """

    output_size: int

    # TODO: priors and shapes must be defined on any abstract subclass, but
    # there is no way to define an abstract class property
    # priors: ParamPriorsT | ParamPriorsTupleT
    # shapes: ParamShapesT | ParamShapesTupleT

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

    Parameters
    ----------
    output_size
        Size of the output vector.
    priors
        Prior distributions for transform parameters.
    shapes
        Shape specifications for transform parameters.
    transform
        The transform function. Should take latents as the first argument,
        followed by any parameters.
    vmap
        Whether to automatically vectorize the transform over the batch dimension.
        If True (default), the transform function should be written for a single
        sample (latents shape ``(latent_size,)``), and JAX's ``vmap`` will be applied
        to handle batches. Parameters are shared across all samples.
        If False, the transform function must handle batching itself. This is
        useful when parameters are per-sample (e.g., per-star nuisance parameters)
        or when the function has custom batching requirements.
    """

    transform: TransformFuncT
    priors: ParamPriorsT = eqx.field(default=ImmutableMap(), converter=ImmutableMap)
    shapes: ParamShapesT = eqx.field(default=ImmutableMap(), converter=ImmutableMap)

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
        for param_name in (*self._param_names, *self.priors, *self.shapes):
            if ":" in param_name:
                msg = (
                    f"Transform parameter name '{param_name}' contains ':' which is "
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
        expanded_priors = {}
        for name, prior in self.priors.items():
            if name in self.shapes:
                shape = _resolve_shape(
                    self.shapes[name],
                    output_size=self.output_size,
                    latent_size=latent_size,
                    data_size=data_size,
                )
                expanded_priors[name] = prior.expand(shape)
            else:
                expanded_priors[name] = prior
        return ImmutableMap(**expanded_priors)

    def unpack_pars(
        self, flat_pars: dict[str, Any], ignore_missing: bool = False
    ) -> dict[str, Any]:
        """Pack/unpack parameters (identity for single transforms)."""
        for param_name in self._param_names:
            if param_name not in flat_pars and not ignore_missing:
                msg = f"Missing value in transform: {param_name}"
                raise ValueError(msg)
        return flat_pars

    pack_pars = unpack_pars


class AbstractMultiTransform(AbstractTransform):
    """Base class for transforms that delegate to a tuple of child transforms.

    Child transform parameters are named with a flat ``"{index}:{param}"`` scheme,
    so a parameter ``"A"`` belonging to child 0 is called ``"0:A"``. Subclasses
    define how the children are wired together (in sequence, in parallel, ...) by
    implementing :meth:`apply` and :meth:`get_expanded_priors`.
    """

    transforms: tuple[AbstractTransform, ...]

    @property
    def priors(self) -> ParamPriorsTupleT:
        """Collect parameter priors from all child transforms."""
        return tuple(
            getattr(transform, "priors", ImmutableMap())
            for transform in self.transforms
        )

    @property
    def shapes(self) -> ParamShapesTupleT:
        """Collect parameter shapes from all child transforms."""
        return tuple(
            getattr(transform, "shapes", ImmutableMap())
            for transform in self.transforms
        )

    @property
    def names_nested(self) -> tuple[tuple[str, ...], ...]:
        """Parameter names grouped by child transform."""
        return tuple(t._param_names for t in self.transforms)  # type: ignore[attr-defined]

    @property
    def names_flat(self) -> tuple[str, ...]:
        """Flat parameter names using the ``{index}:{param}`` convention."""
        return tuple(
            f"{i}:{name}" for i, names in enumerate(self.names_nested) for name in names
        )

    @property
    def _param_names(self) -> tuple[str, ...]:
        """Flat parameter names, for compatibility when nested in another transform."""
        return self.names_flat

    def _child_pars(
        self, args: tuple[dict[str, Any], ...], kwargs: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Resolve ``apply()`` arguments into one parameter dict per child transform.

        Parameters can be passed either as one positional dictionary per child, or
        as flat keyword arguments using the ``"{index}:{param}"`` naming scheme.
        """
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
            return list(args)

        child_pars: list[dict[str, Any]] = [{} for _ in self.transforms]
        for param_name, param_value in kwargs.items():
            idx_str, sep, actual_param_name = param_name.partition(":")
            if not sep:
                msg = f"Unsupported parameter name format: {param_name}"
                raise ValueError(msg)
            transform_idx = int(idx_str)
            if not 0 <= transform_idx < len(self.transforms):
                msg = f"Invalid transform index: {transform_idx}"
                raise ValueError(msg)
            child_pars[transform_idx][actual_param_name] = param_value
        return child_pars

    def unpack_pars(
        self, flat_pars: dict[str, Any], ignore_missing: bool = False
    ) -> tuple[dict[str, Any], ...]:
        """Convert flat parameter names to nested tuple structure.

        Takes parameters with names like ``"0:A"``, ``"1:p1"`` and converts them to
        a tuple of parameter dictionaries, one per child transform.
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

            idx_str, _, actual_param_name = param_name.partition(":")
            nested_pars[int(idx_str)][actual_param_name] = param_value

        return tuple(nested_pars)

    def pack_pars(
        self, nested_pars: list[dict[str, Any]], ignore_missing: bool = False
    ) -> dict[str, Any]:
        """Convert nested parameter structure to flat naming scheme.

        Takes a list of parameter dictionaries, one per child transform, and
        converts them to flat parameter names like ``"0:A"``, ``"1:p1"``.
        """
        _ = ignore_missing  # unused: every provided parameter is packed
        return {
            f"{i}:{param_name}": param_value
            for i, transform_pars in enumerate(nested_pars)
            for param_name, param_value in transform_pars.items()
        }


class TransformSequence(AbstractMultiTransform):
    """A sequence of transforms applied in order.

    Composes multiple transforms together, where the output of each transform
    becomes the input to the next transform in the sequence.

    Parameters are stored as tuples of dictionaries, one element per transform.
    """

    def __init__(self, transforms: tuple[AbstractTransform, ...]):
        """Initialize a sequence of transforms."""
        if not transforms:
            msg = "At least one transform required"
            raise ModelValidationError(msg)

        # Set output size to the final transform's output size
        self.output_size = transforms[-1].output_size
        self.transforms = transforms

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
        output = latents
        for transform, transform_pars in zip(
            self.transforms, self._child_pars(args, kwargs)
        ):
            output = transform.apply(output, **transform_pars)
        return output

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


class ConcatenateTransform(AbstractMultiTransform):
    """Transform that splits input latents and passes slices to child transforms.

    Splits the input latent vector by ``input_sizes``, passes each slice to a
    corresponding child transform, and concatenates the outputs. This is useful
    for models where different subsets of latent variables control different
    output components (e.g., separate absorption and continuum models).

    Parameters
    ----------
    transforms
        Child transforms, one per input slice.
    input_sizes
        Number of latent dimensions to send to each child transform.
        Must satisfy ``len(input_sizes) == len(transforms)`` and
        ``sum(input_sizes) == latent_size``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from pollux.models.transforms import (
    ...     ConcatenateTransform, LinearTransform, PolyFeatureTransform,
    ...     TransformSequence,
    ... )

    Combine a polynomial feature transform with a linear transform:

    >>> concat = ConcatenateTransform(
    ...     transforms=(
    ...         TransformSequence((
    ...             PolyFeatureTransform(degree=2),
    ...             LinearTransform(output_size=10),
    ...         )),
    ...         LinearTransform(output_size=4),
    ...     ),
    ...     input_sizes=(3, 4),
    ... )

    With 7 total latent dimensions (3 + 4), the first 3 go through the
    polynomial + linear path producing 10 outputs, and the last 4 go through
    the linear path producing 4 outputs, for a total output size of 14.
    """

    input_sizes: tuple[int, ...]

    def __init__(
        self,
        transforms: tuple[AbstractTransform, ...],
        input_sizes: tuple[int, ...],
    ):
        """Initialize a concatenation of transforms with input size allocation."""
        if not transforms:
            msg = "At least one transform required"
            raise ModelValidationError(msg)

        if len(transforms) != len(input_sizes):
            msg = (
                f"Number of transforms ({len(transforms)}) must match "
                f"number of input_sizes ({len(input_sizes)})"
            )
            raise ModelValidationError(msg)

        self.transforms = transforms
        self.input_sizes = tuple(input_sizes)
        self.output_size = sum(t.output_size for t in transforms)

    def apply(
        self, latents: BatchedLatentsT, *args: dict[str, Any], **kwargs: Any
    ) -> BatchedOutputT:
        """Apply child transforms to slices of the input and concatenate outputs.

        Parameters can be provided in two ways:

        1. As positional arguments: One dictionary per transform.
        2. As keyword arguments: Using ``"{transform_index}:{param}"`` naming.

        Parameters
        ----------
        latents
            Input latent vectors of shape ``(n_samples, sum(input_sizes))``.
        *args
            Parameter dictionaries, one per child transform.
        **kwargs
            Flat parameters using ``"{transform_index}:{param_name}"`` naming.

        Returns
        -------
        array
            Concatenated outputs of shape ``(n_samples, sum(output_sizes))``.
        """
        # Split latents by input_sizes along the last axis
        split_indices = tuple(
            sum(self.input_sizes[: i + 1]) for i in range(len(self.input_sizes) - 1)
        )
        latent_slices = jnp.split(latents, split_indices, axis=-1)

        outputs = [
            transform.apply(slice_, **pars)
            for transform, slice_, pars in zip(
                self.transforms, latent_slices, self._child_pars(args, kwargs)
            )
        ]
        return jnp.concatenate(outputs, axis=-1)

    def get_expanded_priors(
        self, latent_size: int, data_size: int | None = None
    ) -> ParamPriorsT:
        """Get expanded parameter priors using flat naming scheme.

        Each child transform receives its corresponding ``input_sizes[i]`` as
        its ``latent_size``.

        Parameters
        ----------
        latent_size
            Total latent size (must equal ``sum(input_sizes)``).
        data_size
            Number of objects in the dataset, passed through to child transforms.

        Raises
        ------
        ModelValidationError
            If ``latent_size`` does not match ``sum(input_sizes)``.
        """
        expected_total = sum(self.input_sizes)
        if latent_size != expected_total:
            msg = (
                f"latent_size ({latent_size}) does not match "
                f"sum(input_sizes) ({expected_total})"
            )
            raise ModelValidationError(msg)

        priors = {}
        for i, (transform, input_size) in enumerate(
            zip(self.transforms, self.input_sizes)
        ):
            transform_priors = transform.get_expanded_priors(
                latent_size=input_size, data_size=data_size
            )
            for param_name, prior in transform_priors.items():
                flat_name = f"{i}:{param_name}"
                priors[flat_name] = prior

        return ImmutableMap(**priors)

    def get_output_size(self, input_size: int) -> int:
        """Compute total output size.

        Validates that ``input_size`` matches ``sum(input_sizes)`` and returns
        the total output size.

        Parameters
        ----------
        input_size
            Number of input features (must equal ``sum(input_sizes)``).

        Returns
        -------
        int
            Total output size (sum of child output sizes).
        """
        expected_total = sum(self.input_sizes)
        if input_size != expected_total:
            msg = (
                f"input_size ({input_size}) does not match "
                f"sum(input_sizes) ({expected_total})"
            )
            raise ModelValidationError(msg)
        return self.output_size


class FunctionTransform(AbstractSingleTransform):
    """Custom transformation using a user-defined function.

    This transform allows for arbitrary transformations defined by the user.
    It is particularly useful for modeling complex relationships or per-sample
    nuisance parameters.

    Parameters
    ----------
    output_size
        Size of the output vector.
    transform
        The transform function. Should take latents as the first argument,
        followed by any parameters defined in ``priors``.
    priors
        Prior distributions for transform parameters. Use :data:`ParamPriorsT`
        (an ``ImmutableMap[str, dist.Distribution]``).
    shapes
        Shape specifications for transform parameters. Use :data:`ParamShapesT`
        (an ``ImmutableMap[str, ShapeT]``). Shapes may name dimensions such as
        ``"latent_size"`` or ``"data_size"`` (see :data:`ShapeT`).
    vmap
        Whether to automatically vectorize the transform over the batch dimension.
        Set to False when parameters are per-sample (e.g., per-star continuum
        corrections) and the function handles batching internally.

    Examples
    --------
    Define a custom linear transform with learnable weights:

    >>> import jax.numpy as jnp
    >>> import numpyro.distributions as dist
    >>> from xmmutablemap import ImmutableMap
    >>> from pollux.models.transforms import FunctionTransform
    >>>
    >>> def my_transform(z, A):
    ...     return jnp.dot(A, z)
    >>>
    >>> custom = FunctionTransform(
    ...     output_size=128,
    ...     transform=my_transform,
    ...     priors=ImmutableMap({"A": dist.Normal(0, 1)}),
    ...     shapes=ImmutableMap({"A": ("output_size", "latent_size")}),
    ... )

    The parameter ``A`` will have shape ``(128, latent_size)`` where ``latent_size``
    is determined when the transform is registered with a model.

    See also the "Inferring Continuum Model Parameters" tutorial for an example of
    using FunctionTransform with per-star parameters and ``vmap=False``.
    """


# ----


def _noop_transform(z: LatentsT) -> OutputT:
    """No-op transformation."""
    return z


class NoOpTransform(AbstractSingleTransform):
    """No-op transformation."""

    output_size: int = 0
    transform: TransformFuncT = _noop_transform
    priors: ParamPriorsT = ImmutableMap()
    shapes: ParamShapesT = ImmutableMap()


# ----


def _compute_n_poly_features(n_inputs: int, degree: int, include_bias: bool) -> int:
    """Number of monomials of degree <= ``degree`` in ``n_inputs`` variables.

    That is C(n+d, d), minus the constant term when ``include_bias`` is False.
    """
    return comb(n_inputs + degree, degree) - (0 if include_bias else 1)


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
    >>> polynomial_features(x, degree=2, include_bias=True)  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    Array([[ 1.,  1.,  2.,  1.,  2.,  4.],
           [ 1.,  3.,  4.,  9., 12., 16.]], dtype=float...)
    """
    # Each monomial is a combination (with replacement) of column indices. The
    # degree-0 combination is the empty tuple, whose empty product is the bias.
    monomials = [
        idx
        for d in range(0 if include_bias else 1, degree + 1)
        for idx in combinations_with_replacement(range(x.shape[1]), d)
    ]
    return jnp.stack([jnp.prod(x[:, list(idx)], axis=1) for idx in monomials], axis=1)


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
    priors: ParamPriorsT = ImmutableMap()
    shapes: ParamShapesT = ImmutableMap()

    # output_size is computed dynamically based on input size
    # We set a placeholder that will be overridden in apply()
    output_size: int = eqx.field(default=0)

    @property
    def _param_names(self) -> tuple[str, ...]:
        """Return empty tuple (no learnable parameters)."""
        return ()

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
    priors: ParamPriorsT = eqx.field(
        default=ImmutableMap({"A": dist.Normal(0, 1)}),
        converter=ImmutableMap,
    )
    shapes: ParamShapesT = ImmutableMap({"A": ("output_size", "latent_size")})


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
    priors: ParamPriorsT = eqx.field(
        default=ImmutableMap({"b": dist.Normal(0, 1)}),
        converter=ImmutableMap,
    )
    shapes: ParamShapesT = ImmutableMap({"b": ("output_size", "one")})


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
    priors: ParamPriorsT = eqx.field(
        default=ImmutableMap({"A": dist.Normal(0, 1), "b": dist.Normal(0, 1)}),
        converter=ImmutableMap,
    )
    shapes: ParamShapesT = ImmutableMap(
        {
            "A": ("output_size", "latent_size"),
            "b": ("output_size", "one"),
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
    priors: ParamPriorsT = eqx.field(
        default=ImmutableMap(
            {"Q": dist.Normal(0, 1), "A": dist.Normal(0, 1), "b": dist.Normal(0, 1)}
        ),
        converter=ImmutableMap,
    )
    shapes: ParamShapesT = ImmutableMap(
        {
            "Q": ("output_size", "latent_size", "latent_size"),
            "A": ("output_size", "latent_size"),
            "b": ("output_size", "one"),
        }
    )


# ----


def _param_path_str(path: tuple[Any, ...]) -> str:
    """Format a JAX key path as a dotted string, e.g. ``"layers.0.weight"``.

    Only attribute and sequence keys are handled, which are the only kinds that
    Equinox modules produce.
    """
    return ".".join(
        str(key.idx) if isinstance(key, jtu.SequenceKey) else str(key.name)
        for key in path
    )


def _get_flat_params(module: eqx.Module) -> tuple[tuple[str, jax.Array], ...]:
    """Path/array pairs for every array parameter in an Equinox module.

    Traverses the PyTree structure of an Equinox module to generate a unique
    path string for each array leaf, in flattening order. This is used to create
    flat parameter names for numpyro sampling.

    Examples
    --------
    >>> import equinox as eqx
    >>> import jax
    >>> key = jax.random.PRNGKey(0)
    >>> mlp = eqx.nn.MLP(in_size=4, out_size=8, width_size=16, depth=2, key=key)
    >>> tuple(path for path, _ in _get_flat_params(mlp))[:4]  # First few paths
    ('layers.0.weight', 'layers.0.bias', 'layers.1.weight', 'layers.1.bias')
    """
    arrays = eqx.filter(module, eqx.is_array)
    return tuple(
        (_param_path_str(path), leaf)
        for path, leaf in jtu.tree_leaves_with_path(arrays)
    )


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

    # No static priors or shapes - these are computed dynamically
    priors: ParamPriorsT = eqx.field(default_factory=lambda: ImmutableMap())
    shapes: ParamShapesT = eqx.field(default_factory=lambda: ImmutableMap())

    # Internal state - computed in get_expanded_priors
    _param_paths: tuple[str, ...] = eqx.field(default=(), repr=False)
    _template_nn: Any = eqx.field(default=None, repr=False)  # eqx.Module

    @property
    def _param_names(self) -> tuple[str, ...]:
        """Parameter names (delegated to _param_paths)."""
        return self._param_paths

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

        flat_params = _get_flat_params(template_nn)

        # Store template and paths for use in apply()
        # Note: We use object.__setattr__ because eqx.Module is frozen
        object.__setattr__(self, "_template_nn", template_nn)
        object.__setattr__(self, "_param_paths", tuple(p for p, _ in flat_params))

        # Create expanded priors for each parameter
        priors = {}
        for path, param in flat_params:
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

        # Reconstruct the NN, swapping in whichever parameters were provided
        arrays, static = eqx.partition(self._template_nn, eqx.is_array)
        leaves, treedef = jtu.tree_flatten(arrays)
        nn: eqx.Module = eqx.combine(
            jtu.tree_unflatten(
                treedef,
                [
                    params.get(path, leaf)
                    for path, leaf in zip(self._param_paths, leaves)
                ],
            ),
            static,
        )

        # Apply NN to each latent vector using vmap
        # The nn is an eqx.Module which is callable via __call__
        def forward(x: jax.Array) -> jax.Array:
            result: jax.Array = nn(x)  # type: ignore[operator]
            return result

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


class AdditiveOffsetTransform(eqx.Module):
    """Transform that wraps a base transform and adds a per-star scalar offset.

    This transform is useful for modeling per-object nuisance parameters like
    distance modulus, where each object has its own offset that applies uniformly
    to all output dimensions. This is a generalization of the :class:`AffineTransform`
    and :class:`OffsetTransform` class, because here the offset can vary per object
    instead of per output.

    In other words, unlike :class:`OffsetTransform` which has a fixed offset vector of
    shape ``(output_size,)``, this transform samples a separate scalar offset for each
    object in the dataset, with shape ``(data_size,)``. The offset is then broadcast to
    all output dimensions.

    Parameters
    ----------
    base_transform
        The underlying transform to wrap (e.g., :class:`LinearTransform`).
    offset_prior
        Prior distribution for the per-object offset. This will be expanded
        to shape ``(data_size,)`` during inference.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import numpyro.distributions as dist
    >>> from pollux.models.transforms import AdditiveOffsetTransform, LinearTransform

    Model apparent magnitudes as absolute magnitudes plus distance modulus:

    >>> phot_trans = AdditiveOffsetTransform(
    ...     base_transform=LinearTransform(output_size=3),  # 3 photometric bands
    ...     offset_prior=dist.Normal(11.0, 3.0),  # Distance modulus prior
    ... )

    The offset adapts to the data size automatically:

    >>> import pollux as plx
    >>> model = plx.Lux(latent_size=8)
    >>> model.register_output("phot", phot_trans)
    >>> # During training with 1000 stars, offset has shape (1000,)
    >>> # During testing with 500 stars, offset has shape (500,)

    Notes
    -----
    The per-star offset is broadcast to all output dimensions, meaning the same
    offset value is added to every element of the output for a given object.
    This is appropriate for distance modulus (which shifts all magnitudes equally)
    but may not be appropriate for other use cases.
    """

    base_transform: AbstractSingleTransform | TransformSequence
    offset_prior: dist.Distribution = eqx.field(
        default_factory=lambda: dist.Normal(0.0, 1.0)
    )

    @property
    def output_size(self) -> int:
        """Output size, inherited from the base transform."""
        return self.base_transform.output_size

    def get_expanded_priors(
        self, latent_size: int, data_size: int | None = None
    ) -> ParamPriorsT:
        """Get expanded parameter priors including the per-star offset.

        Parameters
        ----------
        latent_size
            Size of the latent vector.
        data_size
            Number of objects in the dataset. Required for this transform.

        Returns
        -------
        ParamPriorsT
            Dictionary of priors including base transform priors (prefixed with
            "base:") and the offset prior with shape ``(data_size,)``.

        Raises
        ------
        ValueError
            If ``data_size`` is None.
        """
        if data_size is None:
            msg = (
                "AdditiveOffsetTransform requires data_size to be specified. "
                "This should be set automatically during model.optimize()."
            )
            raise ValueError(msg)

        # Get base transform priors with "base:" prefix
        base_priors = self.base_transform.get_expanded_priors(
            latent_size=latent_size, data_size=data_size
        )

        priors = {}
        for name, prior in base_priors.items():
            priors[f"base:{name}"] = prior

        # Add per-star offset prior
        priors["offset"] = self.offset_prior.expand((data_size,))

        return ImmutableMap(**priors)

    def apply(self, latents: BatchedLatentsT, **params: Any) -> BatchedOutputT:
        """Apply the base transform and add the per-star offset.

        Parameters
        ----------
        latents
            Input latent vectors of shape ``(n_samples, latent_size)``.
        **params
            Parameters including base transform parameters (prefixed with "base:")
            and the "offset" parameter of shape ``(n_samples,)``.

        Returns
        -------
        array
            Output of shape ``(n_samples, output_size)``.
        """
        # Extract base transform parameters (strip "base:" prefix)
        base_params = {}
        for name, value in params.items():
            if name.startswith("base:"):
                base_params[name[5:]] = value

        # Apply base transform
        base_output = self.base_transform.apply(latents, **base_params)

        # Add per-star offset (broadcast to all output dimensions)
        offset = params.get("offset")
        if offset is not None:
            # offset shape: (n_samples,) -> (n_samples, 1) for broadcasting
            base_output = base_output + offset[:, None]

        return base_output

    def unpack_pars(
        self, flat_pars: dict[str, Any], ignore_missing: bool = False
    ) -> dict[str, Any]:
        """Unpack flat parameters into nested structure.

        Parameters
        ----------
        flat_pars
            Flat parameter dictionary with "base:..." prefixed keys and "offset".
        ignore_missing
            If True, skip missing parameters.

        Returns
        -------
        dict
            Nested parameter dictionary with "base" and "offset" keys.
        """
        # Extract base parameters
        base_flat = {}
        offset = None
        for name, value in flat_pars.items():
            if name.startswith("base:"):
                base_flat[name[5:]] = value
            elif name == "offset":
                offset = value

        # Unpack base transform parameters
        base_nested = self.base_transform.unpack_pars(
            base_flat, ignore_missing=ignore_missing
        )

        result: dict[str, Any] = {"base": base_nested}
        if offset is not None:
            result["offset"] = offset
        elif not ignore_missing:
            msg = "Missing parameter: offset"
            raise ValueError(msg)

        return result

    def pack_pars(
        self, nested_pars: dict[str, Any], ignore_missing: bool = False
    ) -> dict[str, Any]:
        """Pack nested parameters into flat structure.

        Parameters
        ----------
        nested_pars
            Nested parameter dictionary with "base" and "offset" keys.
        ignore_missing
            If True, skip missing parameters.

        Returns
        -------
        dict
            Flat parameter dictionary with "base:..." prefixed keys and "offset".
        """
        result = {}

        # Pack base transform parameters
        base_nested = nested_pars.get("base", {})
        base_flat = self.base_transform.pack_pars(
            base_nested, ignore_missing=ignore_missing
        )
        for name, value in base_flat.items():
            result[f"base:{name}"] = value

        # Add offset
        if "offset" in nested_pars:
            result["offset"] = nested_pars["offset"]
        elif not ignore_missing:
            msg = "Missing parameter: offset"
            raise ValueError(msg)

        return result


# ----
