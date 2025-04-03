import numpy as np

import sys

sys.path.append("scripts")
from utils.parsers import ParseSturdrLogs

if __name__ == "__main__":
    nav, var, ch = ParseSturdrLogs("results/THESIS_SIM_ARRAY", True)
    print()
