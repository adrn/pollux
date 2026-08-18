"""Transforms for mapping latent vectors to output quantities."""

__all__ = [
    "AbstractMultiTransform",
    "AbstractSingleTransform",
    "AbstractTransform",
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
    "ScatterTransform",
    "ShapeT",
    "TransformSequence",
]

import abc
import inspect
from collections.abc import Callable
from itertools import accumulate, combinations_with_replacement
from math import comb
from typing import Any

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
    TransformFuncT,
)

#: Named dimensions usable in a parameter shape, resolved when the model is built:
#: ``"output_size"`` (the transform's output dimension), ``"latent_size"`` (the
#: latent space dimension, set when registering with a model), ``"data_size"``
#: (the number of objects in the dataset) and ``"one"`` (always 1, for bias terms).
type ShapeT = tuple[str | int, ...]


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

    resolved = []
    for dim in shape:
        if not isinstance(dim, str):
            resolved.append(dim)
            continue

        size = sizes.get(dim)
        if size is None:
            known = sorted(name for name, value in sizes.items() if value is not None)
            msg = (
                f"Cannot resolve shape dimension '{dim}': no size is known for it "
                f"here. Known dimensions: {known}. Note that 'data_size' is only "
                "known once the transform is used with data, e.g. during "
                "model.optimize()."
            )
            raise ValueError(msg)
        resolved.append(size)
    return tuple(resolved)


def _expand_prior(
    prior: dist.Distribution, shape: tuple[int, ...]
) -> dist.Distribution:
    """Expand a prior so that a draw from it has shape ``shape``.

    :meth:`~numpyro.distributions.Distribution.expand` takes a *batch* shape, so a
    prior carrying event dimensions of its own -- a ``MultivariateNormal`` correlating
    one axis of a parameter -- must only be expanded over the leading axes, with its
    event shape accounting for the rest. Expanding it over the full shape instead
    would silently give draws an extra trailing axis.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import numpyro.distributions as dist
    >>> from pollux.models.transforms import _expand_prior
    >>> mvn = dist.MultivariateNormal(jnp.zeros(3), covariance_matrix=jnp.eye(3))
    >>> expanded = _expand_prior(mvn, (8, 3))
    >>> expanded.batch_shape, expanded.event_shape
    ((8,), (3,))
    >>> _expand_prior(dist.Normal(), (8, 3)).batch_shape
    (8, 3)
    """
    event = tuple(prior.event_shape)
    if not event:
        return prior.expand(shape)

    if len(event) > len(shape) or tuple(shape[-len(event) :]) != event:
        msg = (
            f"A prior with event shape {event} cannot describe a parameter of shape "
            f"{shape}: its event shape has to match the trailing axes. A "
            "MultivariateNormal correlating an axis of length n needs event shape "
            "(n,), and that axis has to be the last one."
        )
        raise ValueError(msg)

    return prior.expand(shape[: -len(event)])


#: Type alias for parameter priors: maps parameter names to distributions.
#: Used with :class:`FunctionTransform` to specify priors for learnable parameters.
type ParamPriorsT = ImmutableMap[str, dist.Distribution]

#: Type alias for parameter shapes: maps parameter names to shapes. Each shape is
#: a tuple of concrete sizes and/or named dimensions (see :data:`ShapeT`).
type ParamShapesT = ImmutableMap[str, ShapeT]

# Internal: Tuples of parameters for TransformSequence
type ParamPriorsTupleT = tuple[ParamPriorsT, ...]
type ParamShapesTupleT = tuple[ParamShapesT, ...]


class AbstractTransform(eqx.Module):
    """Base class defining the transform interface.

    Transforms convert latent vectors to observable quantities through parameterized
    functions. They define the mapping between latent space and output spaces.
    """

    output_size: int

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

    def get_output_size(self, input_size: int) -> int:
        """Output size given an input size.

        Fixed for most transforms; overridden by those (like
        :class:`PolyFeatureTransform`) whose output size depends on their input.
        """
        del input_size
        return self.output_size


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
                expanded_priors[name] = _expand_prior(prior, shape)
            else:
                expanded_priors[name] = prior
        return ImmutableMap(**expanded_priors)

    def unpack_pars(
        self, flat_pars: dict[str, Any], ignore_missing: bool = False
    ) -> dict[str, Any]:
        """Unpack parameters (identity for single transforms)."""
        for param_name in self._param_names:
            if param_name not in flat_pars and not ignore_missing:
                msg = f"Missing value in transform: {param_name}"
                raise ValueError(msg)
        return flat_pars

    def pack_pars(
        self, nested_pars: dict[str, Any], ignore_missing: bool = False
    ) -> dict[str, Any]:
        """Pack parameters (identity for single transforms)."""
        return self.unpack_pars(nested_pars, ignore_missing=ignore_missing)


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
        # Every concrete transform defines _param_names, as a field or a
        # property, but it cannot be declared on AbstractTransform: eqx.AbstractVar
        # reads as a required dataclass field, which then breaks every constructor.
        return tuple(t._param_names for t in self.transforms)  # ty: ignore[unresolved-attribute]

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

            # The next transform's "latent_size" is this one's output size
            current_size = transform.get_output_size(current_size)

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
        latent_slices = jnp.split(
            latents, tuple(accumulate(self.input_sizes[:-1])), axis=-1
        )

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
        self.get_output_size(latent_size)  # validates latent_size

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

    Per-object nuisance parameters use the ``"data_size"`` named dimension and
    ``vmap=False``, so the function sees the whole batch. For example, a distance
    modulus — one scalar offset per object, added to every output dimension —
    composes with any base transform:

    >>> from pollux.models.transforms import LinearTransform, TransformSequence
    >>> distance_modulus = FunctionTransform(
    ...     output_size=3,
    ...     transform=lambda mags, offset: mags + offset[:, None],
    ...     priors=ImmutableMap({"offset": dist.Normal(11.0, 3.0)}),
    ...     shapes=ImmutableMap({"offset": ("data_size",)}),
    ...     vmap=False,
    ... )
    >>> apparent_mags = TransformSequence(
    ...     (LinearTransform(output_size=3), distance_modulus)
    ... )

    The offset is sampled with shape ``(data_size,)``, so it adapts to however many
    objects are in the dataset being fit.

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
            "b": ("output_size",),
        }
    )


# ----


def _scatter_transform(err: OutputT, s: OutputT) -> OutputT:
    """Add an intrinsic scatter in quadrature to the reported errors.

    Computes sqrt(err^2 + s^2).
    """
    return jnp.sqrt(err**2 + s**2)


class ScatterTransform(AbstractSingleTransform):
    """Intrinsic scatter added in quadrature to the reported errors.

    Implements the transformation: y = sqrt(err^2 + s^2), where ``s`` is a fitted
    per-element scatter. This is meant to be used as the ``err_transform`` of an
    output, where it absorbs variance the reported errors do not account for.

    The prior on ``s`` is an ordinary field, so the scale that suits your data --
    which depends on how the output was preprocessed -- can be set at construction.

    Examples
    --------
    >>> import numpyro.distributions as dist
    >>> from pollux.models.transforms import ScatterTransform
    >>> trans = ScatterTransform(output_size=8)
    >>> wider = ScatterTransform(output_size=8, priors={"s": dist.HalfNormal(5.0)})
    """

    transform: TransformFuncT = _scatter_transform
    priors: ParamPriorsT = eqx.field(
        default=ImmutableMap({"s": dist.HalfNormal(1.0)}),
        converter=ImmutableMap,
    )
    shapes: ParamShapesT = ImmutableMap({"s": ("output_size",)})


def scatter_at_scale(output_size: int, scale: float | None) -> ScatterTransform:
    """A :class:`ScatterTransform` with its ``HalfNormal`` prior at ``scale``.

    ``scale=None`` keeps the transform's own default prior, so that default is
    written down in exactly one place. Architectures use this to turn a
    per-output scatter selector into transforms.
    """
    if scale is None:
        return ScatterTransform(output_size=output_size)
    return ScatterTransform(
        output_size=output_size, priors={"s": dist.HalfNormal(scale)}
    )


# ----


def _param_path_key_str(key: Any) -> str:
    """Format one JAX key path entry, e.g. ``"layers"``, ``"0"``, or a dict key."""
    if isinstance(key, jtu.GetAttrKey):
        return str(key.name)
    if isinstance(key, jtu.SequenceKey):
        return str(key.idx)
    if isinstance(key, (jtu.DictKey, jtu.FlattenedIndexKey)):
        return str(key.key)
    # Custom pytree nodes may register their own key type
    return str(key).lstrip(".")


def _param_path_str(path: tuple[Any, ...]) -> str:
    """Format a JAX key path as a dotted string, e.g. ``"layers.0.weight"``.

    Handles the standard JAX key types, so modules holding their submodules in a
    dict (``DictKey``) or in a custom pytree node (``FlattenedIndexKey``) name
    their parameters just as well as attributes and sequences do.
    """
    return ".".join(_param_path_key_str(key) for key in path)


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

    Use with LVM:

    >>> import pollux as plx
    >>> model = plx.LVM(latent_size=8)
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

        # Reconstruct the NN, swapping in whichever parameters were provided.
        # Typed as a callable because that is what this transform requires of it.
        arrays, static = eqx.partition(self._template_nn, eqx.is_array)
        leaves, treedef = jtu.tree_flatten(arrays)
        nn: Callable[[jax.Array], jax.Array] = eqx.combine(
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
        return jax.vmap(nn)(latents)

    def unpack_pars(
        self, flat_pars: dict[str, Any], ignore_missing: bool = False
    ) -> dict[str, Any]:
        """Unpack parameters (identity, keyed by NN parameter path)."""
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
        """Pack parameters (identity, keyed by NN parameter path)."""
        return self.unpack_pars(nested_pars, ignore_missing=ignore_missing)
