import os
import sys

# Hardcode the built module path for now
ROOT = os.path.abspath(os.path.dirname(__file__))
PYMOD_DIR = os.path.join(ROOT, "build", "python")
if PYMOD_DIR not in sys.path:
    sys.path.insert(0, PYMOD_DIR)

print("Set up path")

import mdio_cpp as mdio  # noqa: E402
import numpy as np  # noqa: E402
print("Imported mdio_cpp and numpy")

# Fill in your dataset path and open mode
DATASET_PATH = "/home/ubuntu/source/mdio-cpp/teapot_example.mdio"
print("Dataset path: ", DATASET_PATH)
def main():
    # Open existing dataset
    print("Opening dataset")
    ds = mdio.Dataset.open(DATASET_PATH, open_mode="open")
    print("Dataset opened")
    # Slice inline, crossline, time to [0, 128)
    sliced = ds.isel([
        ("inline", 0, 128),
        ("crossline", 0, 128),
        ("time", 0, 128),
    ])
    print("Sliced")
    # Get the amplitude variable
    amp_var = sliced.get_variable("amplitude")
    print("Amplitude variable obtained")
    # Read to NumPy
    amp_data = amp_var.read()
    arr = amp_data["data"]  # numpy array view
    print("Original dtype/shape:", arr.dtype, arr.shape)
    print("Mutating in-place")
    # Mutate in-place
    arr[...] = 4.20

    # TODO: write back (binding not exposed yet)
    # For example, once Write is bound:
    # amp_var.write(amp_data)  # where write accepts the same VariableData dict
    # ds.commit_metadata()     # if metadata changed

if __name__ == "__main__":
    main()