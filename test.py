from preprocess.preprocess_traning_data import preprocess_trainset
import os

# Define parameters
inp_root = os.path.join("data")

if os.path.exists(inp_root):
    print("File exists")
else:
    print("File not found:", inp_root)

sr = 22050  # Sample rate
n_p = 4  # Number of parallel processes
exp_dir = os.path.join("output")
per = 3.7  # Some parameter for segment duration
noparallel = False

# Ensuring the multiprocessing part runs within the main script context
if __name__ == "__main__":
    # Call the function
    preprocess_trainset(inp_root, sr, n_p, exp_dir, per, noparallel)
