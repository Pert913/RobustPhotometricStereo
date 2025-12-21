import numpy as np
import os

# --- Define File Paths ---
txt_filename = './data/buddha/buddhaPNG/light_directions.txt'  # The name of the file from the DiLiGenT dataset
npy_filename = './data/buddha/buddhaPNG/light_directions.npy'  # The name your program expects

# 1. Load the data from the text file using numpy.loadtxt()
try:
    # np.loadtxt is ideal for simple text files containing only numbers.
    # It reads the data and returns a 2D NumPy array (N_lights x 3)
    # The delimiter is usually whitespace (default), but you can specify ',' if it's a CSV.
    light_directions = np.loadtxt(txt_filename)
    
    # You can verify the shape of the resulting array
    print(f"Successfully loaded {txt_filename} with shape: {light_directions.shape}")

except FileNotFoundError:
    print(f"Error: {txt_filename} not found. Check the file path.")
    exit()

# 2. Save the array to the binary .npy file format
np.save(npy_filename, light_directions)

print(f"Successfully converted and saved array to {npy_filename}")

# --- Optional: Verify the saved file ---
loaded_npy = np.load(npy_filename)
print(f"Loaded .npy file shape check: {loaded_npy.shape}")

# Now you can use 'lights.npy' in your `load_lightnpy()` function!