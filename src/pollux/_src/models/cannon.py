"""NOTE: this is just a placeholder and sketch of what the Cannon model might look like.

TODO:
- Figure out data preprocessor/feature-izer
- Regularization schemes (numpyro factor?)
- Docstrings and examples
"""

from typing import Any

import jax
import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike

from ..shared import AbstractModel


class Cannon(AbstractModel):
    """The Cannon is a model for stellar spectra and other data given stellar labels.

    Parameters
    ----------
    feature_size
        The size of the feature vectors for the model.

    Examples
    --------

    """

    feature_size: int

    def predict_outputs(
        self,
        features: jax.Array,
        params: dict[str, Any],
        which: list[str] | str | None = None,
    ) -> jax.Array | dict[str, jax.Array]:
        """Predict outputs for given latent vectors and parameters.

        Parameters
        ----------
        TODO
            The latent vectors for your sample of stars.
        params
            A dictionary of parameters for each output in the model.
        which
            A list of output names to predict. If None, predict all outputs.

        Returns
        -------
        dict
            A dictionary of predicted outputs, where the keys are the output names.
        """

        if which is None:
            which = list(self.outputs.keys())
        elif isinstance(which, str):
            which = [which]

        results = {}
        for name in which:
            output = self.outputs[name]

            # Note: needs to be a list or iterable because of how vmap works
            # output_params = {k: params[name][k] for k in output.transform_params}
            output_params = [params[name][k] for k in output.transform_params]
            results[name] = output.transform(features, *output_params)

        return results

    def setup_numpyro(
        self,
        features: jax.Array,
        data: dict[str, ArrayLike],
        errs: dict[str, ArrayLike],
        which: list[str] | None = None,
    ) -> dict[str, Any]:
        """Sample parameters and set up basic numpyro model.

        Parameters
        ----------
        TODO
            The latent vectors for your sample of stars.
        data
            A dictionary of observed data.
        errs
            A dictionary of errors for the observed data.
        which
            A list of output names to include in the model. If None, use all outputs.

        Returns
        -------
        dict
            A dictionary of sampled parameters for each output.
        """
        which = which or list(self.outputs.keys())

        params: dict[str, dict[str, Any]] = {}
        for name in which:
            output = self.outputs[name]
            params[name] = {}
            for param_name, prior in output.transform_params.items():
                params[name][param_name] = numpyro.sample(f"{name}:{param_name}", prior)

        outputs = self.predict_outputs(features, params, which=which)
        for name in which:
            pred = outputs[name]
            numpyro.sample(f"obs:{name}", dist.Normal(pred, errs[name]), obs=data[name])

        return params

    def default_numpyro_model(
        self,
        data: dict[str, ArrayLike],
        errs: dict[str, ArrayLike],
        fixed_params: dict[str, jax.Array] | None = None,
    ) -> None:
        """Create the default numpyro model for the Cannon.

        TODO

        Parameters
        ----------
        data
            A dictionary of observed data.
        errs
            A dictionary of errors for the observed data.
        fixed_params
            A dictionary of fixed parameters to condition on. If None, all parameters
            will be sampled.

        """
        data_, errs_ = self._validate_data(data, errs)
        # n_data = len(data_[next(iter(data_.keys()))])

        # Use condition handler to fix parameters if specified
        with numpyro.handlers.condition(data=fixed_params or {}):
            self.setup_numpyro(data["features"], data_, errs_)  # type: ignore[arg-type]
