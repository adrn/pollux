# Linearized, Closed-form solves

After constructing a model, you can optimize it either with stochastic variational
inference (SVI) via the {py:func}`~pollux.models.lux.Lux.optimize` method or with
`optimize_iterative`. `optimize_iterative` fits a model by block coordinate descent: it
cycles through blocks of parameters, optimizing each block while holding the others
fixed. The reason this is worth doing is that some of the sub-problems are linear in the
parameters and thus have a closed-form (least squares) solution.

Given the flexibility of Pollux's model construction and transform system, it is not
always obvious which sub-problems are linear for the iterative optimizer to use the
closed-form solution. This page describes how Pollux handles this, because it makes use
of some nice features in JAX.

## "Is this transform linear?"

Internally, Pollux needs to know whether a block of parameters can be solved in closed
form (i.e., whether the sub-problem is linear in the parameters). One obvious approach
could be to check the type of the transform: If an output is related to the latents via
a {py:class}`~pollux.models.transforms.LinearTransform`, then it is clearly linear. But
this is not sufficient or fully flexible, because a transform can be linear even if it
is not a `LinearTransform`. For example, a transform could be a composition (or
sequence) of linear transforms, or it could be a linear transform preceded by a
parameter-free function (like slicing the latent vector). So developing a set of type
checks to determine linearity is brittle and incomplete.

Here is an explicit example of a model that is linear in the latents but whose
`data_transform` is not a `LinearTransform`:

```python
import pollux as plx
import pollux.models.transforms as trans


def latent_slice(lo, hi):
    """Route latents[lo:hi] into a branch."""
    return trans.FunctionTransform(output_size=hi - lo, transform=lambda z: z[lo:hi])


model = plx.Lux(latent_size=4)
model.register_output(
    "spec",
    trans.TransformSequence(
        (latent_slice(0, 2), trans.LinearTransform(output_size=128))
    ),
)
model.register_output(
    "labels",
    trans.TransformSequence((latent_slice(2, 4), trans.LinearTransform(output_size=3))),
)
```

Every output/branch of the model is linear, and in fact the whole thing is linear, but
no individual data transform (mapping from latents to outputs) is a
{py:class}`~pollux.models.transforms.LinearTransform`. So this would fail a simple type
check, even though the sub-problems are linear and can be solved in closed form.

We could keep widening the check and the logic for identifying linear branches (e.g.,
allow sequences whose members are all linear, allow
{py:class}`~pollux.models.transforms.ConcatenateTransform` of linear transforms, allow a
linear map followed by a fixed offset, etc.). But that would be a very restrictive
approach. Worse, it can only ever recognize transforms that ship with Pollux, so
transforms that are linear but defined by a user via a
{py:class}`~pollux.models.transforms.FunctionTransform` would not be recognized as
linear.

Luckily, we can use JAX to automatically determine whether a transform is linear,
without having to enumerate all the cases. The next section describes how this works.

## Linearization

For the closed-form, least-squares solve, we need to know the design matrix $A$ and
offset $b$ for each linear output:

$$ y = A \, z + b $$

where $z$ is the latent vector and $A$ and $b$ depend only on the parameters being held
fixed. If the transform is a `LinearTransform`, then $A$ is its matrix and $b$ is zero,
but for more general compositions of linear transforms, $A$ and $b$ are not as obvious.
Fortunately, JAX can compute them automatically given any arbitrarily constructed
transform sequence for an output in Pollux: the Jacobian of the full output transform
$f$ at a point $z_0$ is the design matrix $A$!

{py:func}`jax.linearize` produces exactly this decomposition for any function:

```python
import jax.numpy as jnp
import jax

n_data, latent_size = 512, 4
z0 = jnp.zeros((n_data, latent_size))


def predict(latents):
    return model.predict_outputs(latents, params, names=["spec"])["spec"]


b, jvp = jax.linearize(predict, z0)
```

`b` is `predict(z0)`, and `jvp` is a function that applies $A$ to a vector.

This is not an approximation. For a composition of linear primitives, the JVP performs
the same sequence of multiplications and additions on the same values as the primal
computation, so it returns the design matrix itself.

## But: don't construct the design matrix explicitly

Recovering $A$ explicitly is fine for a few output variables, but it's not a great idea
for a stellar spectrum: at 8000 pixels and 8 latent dimensions, the columns are eight
full copies of the data.

The normal equations don't actually need $A$, only $A^\top W A$ and $A^\top W (y - b)$.
The transpose is available directly from {py:func}`jax.linear_transpose`, which turns
the JVP closure into a function applying $J^\top$:

```python
vjpT = jax.linear_transpose(jvp, z0)

AtWy = vjpT(w * (y - b))[0]  # (n_data, latent_size)
AtWA = jnp.stack(
    [vjpT(w * jvp(jnp.broadcast_to(e, z0.shape)))[0] for e in jnp.eye(latent_size)],
    axis=-1,
)  # (n_data, latent_size, latent_size)
```

Each column of $A^\top W A$ costs one JVP and one VJP, and the largest array that ever
exists is one output-sized temporary. The accumulated system is
`(n_data, latent_size, latent_size)`. Summing these contributions over outputs and
adding the prior's regularization gives one small linear system per object, solved with
a single `jax.vmap(jnp.linalg.solve)`.

## But: linearize will work on any function!

Unfortunately, there is one catch. The JVP is defined for any function, not just linear
ones. It is the tangent plane at `z0`, so even nonlinear transforms will produce a
design matrix and offset.

So the decomposition alone doesn't prove that a given output transform is linear -- we
have to detect that another way. We test this numerically within Pollux, which isn't
perfect but seems to work ok in practice.

In a bit more detail, a function is affine if and only if it equals its own
linearization for all $z$ (i.e. everywhere, not just at the point of linearization):

$$ f(z) = f(0) + J z \quad \text{for all } z $$

where $J$ is the Jacobian of $f$. We cannot check this for all $z$, but we can check it
at a few points. The `is_affine` function in Pollux does this: it evaluates both sides
at concrete points and compares, refusing the closed-form solve when they disagree by
more than a relative tolerance of $10^{-6}$.

This means:

- A genuinely affine transform reproduces its linearization exactly, so it can never be
  wrongly refused. No linear model gets demoted to Adam by accident.
- The residual risk runs the other way -- a nonlinear map might slip through by
  happening to touch its tangent plane exactly where we checked.

The only way to make this a proof would be to walk the jaxpr and verify that every
primitive reachable from the input is linear, and no one wants to have to do that!
