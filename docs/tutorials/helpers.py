import pathlib
import sys

# Mirror the integration test helpers here:
tutorials_path = pathlib.Path(__file__).parent
path = tutorials_path / "../../tests/integration"

if str(path) not in sys.path:
    sys.path.append(str(path))

from integration_test_helpers import make_simulated_linear_data  # noqa: F401, E402
