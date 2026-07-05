import numpy as np

# Monkey-patch numpy.load to allow pickle loading for onnx2tf verification data
original_load = np.load
def patched_load(*args, **kwargs):
    kwargs['allow_pickle'] = True
    return original_load(*args, **kwargs)
np.load = patched_load

import sys
from onnx2tf import main

if __name__ == "__main__":
    # Remove run_onnx2tf.py itself from args and run onnx2tf main
    sys.exit(main())
