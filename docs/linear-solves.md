# Closed-form solves, found by linearization

`optimize_iterative` fits a model by block coordinate descent: it cycles through groups
of parameters, optimizing each while holding the rest fixed. The reason this is worth
doing is that some of those sub-problems are not just easier than the full problem —
they are _quadratic_, and a quadratic has an exact answer that you can write down. No
learning rate, no step count, no convergence check.

The awkward part is knowing which sub-problems those are. This page is about how Pollux
answers that question, because the answer turns out to be a rather nice use of JAX.

## The problem with asking "is this transform linear?"

The obvious approach is to check the type of the transform. Pollux used to do exactly
that: if an output's transform was a
{py:class}`~pollux.models.transforms.LinearTransform`, solve it in closed form;
otherwise, run SVI.

That check is wrong in a specific and annoying way. Consider a model whose latent vector
is partitioned, with one half driving a spectrum and the other half driving a set of
labels:

```python
import pollux as plx
from pollux.models.transforms import (
    FunctionTransform,
    LinearTransform,
    TransformSequence,
)


def latent_slice(lo, hi):
    """Route latents[lo:hi] into a branch."""
    return FunctionTransform(output_size=hi - lo, transform=lambda z: z[lo:hi])


model = plx.Lux(latent_size=4)
model.register_output(
    "spec",
    TransformSequence((latent_slice(0, 2), LinearTransform(output_size=128))),
)
model.register_output(
    "labels",
    TransformSequence((latent_slice(2, 4), LinearTransform(output_size=3))),
)
```

Every branch of this model is linear. The whole thing is linear. But no individual
`data_transform` is a `LinearTransform` — each is a `TransformSequence` containing a
`FunctionTransform` — so a type check sees nothing it recognises and quietly falls back
to Adam, on a problem that has an exact answer.

You could keep widening the check: allow sequences whose members are all linear, allow
{py:class}`~pollux.models.transforms.ConcatenateTransform` of linear children, allow a
linear map followed by a fixed offset. That list never ends, and every entry on it is a
chance to be wrong. Worse, it can only ever recognise transforms that ship with Pollux —
a user's own affine `FunctionTransform` is invisible to it.

The better question is not _what type is this transform_ but _what shape is this
sub-problem_. And JAX will answer that for you.

## Linearization gives you the design matrix

Weighted least squares needs a design matrix. For one output, we want to write the
prediction as

$$ y \approx J z + c $$

where $z$ is the latent vector and $J$ and $c$ depend only on the parameters being held
fixed. If the transform is a `LinearTransform`, then $J$ is its matrix `A` and $c$ is
zero — but we would rather not have to know that.

{py:func}`jax.linearize` produces exactly this decomposition for any function:

```python
import jax.numpy as jnp
import jax

n_data, latent_size = 512, 4
z0 = jnp.zeros((n_data, latent_size))


def predict(latents):
    return model.predict_outputs(latents, params, names=["spec"])["spec"]


c, jvp = jax.linearize(predict, z0)
```

`c` is `predict(z0)`, and `jvp` is a function that applies $J$. Push the basis vectors
through it and you recover the design matrix column by column:

```python
columns = jnp.stack(
    [jvp(jnp.broadcast_to(e, z0.shape)) for e in jnp.eye(latent_size)], axis=-1
)
```

For the partitioned model above, `columns` comes back as the `(128, 4)` matrix whose
first two columns are the branch's `A` and whose last two are zero — precisely the
effective design matrix of that output, assembled without anyone having to know that a
slice was involved.

This is not an approximation. For a composition of linear primitives, the JVP performs
the same sequence of multiplications and additions on the same values as the primal
computation, so it returns the design matrix _itself_. Pollux's test suite asserts this
in the strongest available form: for a bare `LinearTransform`, the recovered matrix is
**bitwise** equal to `A`, and the recovered offset is bitwise zero.

## Never building the matrix

Recovering $J$ explicitly is fine for a 3-label output. It is not fine for a spectrum:
at 8000 pixels and 8 latent dimensions, the columns are eight full copies of the data.

The normal equations don't actually need $J$, only $J^\top W J$ and $J^\top W (y - c)$.
The transpose is available directly from {py:func}`jax.linear_transpose`, which turns
the JVP closure into a function applying $J^\top$:

```python
vjpT = jax.linear_transpose(jvp, z0)

AtWy = vjpT(w * (y - c))[0]  # (n_data, latent_size)
AtWA = jnp.stack(
    [vjpT(w * jvp(jnp.broadcast_to(e, z0.shape)))[0] for e in jnp.eye(latent_size)],
    axis=-1,
)  # (n_data, latent_size, latent_size)
```

Each column of $J^\top W J$ costs one JVP and one VJP, and the largest array that ever
exists is one output-sized temporary. The accumulated system is
`(n_data, latent_size, latent_size)` — tiny. Summing these contributions over outputs
and adding the prior's regularization gives one small linear system per object, solved
with a single `jax.vmap(jnp.linalg.solve)`.

So the general version is not just more general than the hand-written one that indexed
`params["A"]`. It also has the same memory footprint, and it picks up the offset $c$ —
which the hand-written version silently dropped, quietly biasing any model with a bias
term.

## Knowing when _not_ to do this

Here is the catch, and it is worth being precise about: **{py:func}`jax.linearize` never
fails**. Hand it a genuinely nonlinear function and it cheerfully returns the tangent
plane at `z0`. Using that as a design matrix would produce a confident, wrong answer.

So the decomposition alone proves nothing. What Pollux does is test the defining
property directly. A function is affine if and only if it equals its own linearization
_everywhere_:

$$ f(z) = f(0) + J z \quad \text{for all } z $$

and so it evaluates both sides at concrete probe points and compares, refusing the
closed-form solve when they disagree by more than a relative tolerance of $10^{-6}$.

This gives a **one-sided guarantee**, which is the important thing to understand about
it:

- A genuinely affine transform reproduces its linearization exactly, so it can never be
  wrongly refused. No linear model gets demoted to Adam by accident.
- The residual risk runs the other way — a nonlinear map might slip through by happening
  to touch its tangent plane exactly where we probed.

Two things make that second case very unlikely. First, a probe array has shape
`(n_data, latent_size)`, and the transform is applied per object, so a single probe is
already `n_data` independent points; the residual is maxed over every output element of
every one of them. Second, the probes come at two amplitudes, one at the scale of the
current latents and one ten times larger. A smooth nonlinearity's deviation from its
tangent plane grows quadratically with amplitude, so the far probe is roughly a hundred
times more sensitive to a map that is only slightly non-affine. The measured separation
is wide:

| probe scale | linear | slice → linear | nonlinear at $10^{-4}$ | degree-2 polynomial |
| ----------- | ------ | -------------- | ---------------------- | ------------------- |
| 0.1         | 0      | 0              | 2.6e-05                | 1.3e-01             |
| 1.0         | 0      | 0              | 2.6e-04                | 9.5e-01             |
| 100.0       | 0      | 0              | 2.6e-02                | 9.9e-01             |

Exactly zero on the left, many orders of magnitude above the tolerance on the right.

The only way to make this a _proof_ would be to walk the jaxpr and verify that every
primitive reachable from the input is linear. That means maintaining a whitelist of
dozens of JAX primitives — which is the same kind of enumeration this design exists to
avoid, one level further down.

## What you get

Solving the output parameters uses the same idea from the other side. The per-pixel
structure is what makes that solve fast, so it is kept: what changes is only that the
design matrix is the features _arriving at_ the linear layer, obtained by running the
parameter-free part of the transform forward. Whether those features came from a latent
slice, a polynomial expansion, or something you wrote yourself makes no difference.

The practical effect is that these all get exact solves now:

| Model                                                | Why it works                                               |
| ---------------------------------------------------- | ---------------------------------------------------------- |
| Partitioned latents feeding separate linear branches | affine in the latents; each branch ends in a linear layer  |
| Polynomial features → linear (i.e. the Cannon)       | the prefix is parameter-free, so it just produces features |
| `ConcatenateTransform` of linear children            | affine in the latents                                      |
| A linear map plus a fixed per-object offset          | the offset lands in $c$                                    |
| `AffineTransform`, `LinearTransform`                 | as before, bit for bit                                     |

## Checking what actually happened

Since the decision is made by measurement rather than declaration, the result reports
it. `IterativeOptimizationResult` carries the blocks as they were actually run:

```python
result = model.optimize_iterative(data, max_cycles=10)

for block in result.blocks:
    print(block.name, block.optimizer)
# latents      least_squares
# spec:data    least_squares
# labels:data  least_squares
```

If a block asks for a closed-form solve and cannot have one, it is downgraded to SVI and
a `PolluxLinearizationWarning` says which blocks and why:

```
PolluxLinearizationWarning: optimize_iterative could not use closed-form
solves for 2 of 3 blocks, falling back to SVI/Adam:
  latents      - output 'flux' is not affine in the latents
  flux:data    - output 'flux' does not end in a linear layer holding all of its parameters
```

The fallback is per block, not all-or-nothing. A model with one neural-network output
and one linear output cannot solve the latents exactly — the latents couple every output
— but the linear output's own parameters are still solved in closed form.

The warning is an ordinary warning category, so the standard library turns it off:

```python
import warnings
from pollux.exceptions import PolluxLinearizationWarning

warnings.filterwarnings("ignore", category=PolluxLinearizationWarning)
```

## When it still won't apply

Two structures are declined on purpose:

- **Parameters spread across more than one layer of a sequence.** Solving these exactly
  means running coordinate descent _inside_ the block. Not implemented; these fall back
  to SVI.
- **Error-transform parameters.** These enter the likelihood through the variance rather
  than the mean, so the sub-problem is not least squares at all.

And of course a genuinely nonlinear transform — a neural network, polynomial features of
the latents themselves — has no closed form to find. That is the case the probe exists
to detect.
