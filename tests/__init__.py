import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="mir_eval")
# Removed problematic imports:
# from .test_core import hierarchies
# from .test_formatting import test_data
