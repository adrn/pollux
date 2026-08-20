import importlib


def pytest_report_header():
    pkgs = ["numpy", "jax", "numpyro", "equinox"]
    versions = "\n\t".join(
        f"{pkg}: {importlib.import_module(pkg).__version__}" for pkg in pkgs
    )
    return [f"\nproject deps:\n\t{versions}\n"]
