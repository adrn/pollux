"""Custom exceptions and warnings."""


class ModelValidationError(Exception):
    pass


class PolluxLinearizationWarning(UserWarning):
    """A block fell back to SVI because it has no closed-form solve.

    Raised by :func:`pollux.models.optimize_iterative`. Silence it with::

        import warnings
        from pollux.exceptions import PolluxLinearizationWarning

        warnings.filterwarnings("ignore", category=PolluxLinearizationWarning)
    """
