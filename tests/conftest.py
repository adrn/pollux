import importlib


def pytest_report_header(config):
    msgs = []

    pkgs = ["numpy", "jax", "numpyro", "equinox"]
    versions = "\n\t".join(
        f"{pkg}: {importlib.import_module(pkg).__version__}" for pkg in pkgs
    )
    msgs.append(f"\nproject deps:\n\t{versions}\n")

    if config.get_verbosity() > 0:
        # TODO: in case we need to add more verbose info
        pass
        # msgs.append()

    return msgs
