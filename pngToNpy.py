import numpy as np
from PIL import Image
import os
import glob # For easily listing files with a pattern
import re # Added for advanced string operations (extracting numbers)

# --- Configuration ---
INPUT_FOLDER = './data/buddha/buddhaPNG/'
OUTPUT_FOLDER = './data/buddha/buddhaPNG_npy/' # Create a new folder for the .npy files

# Define the new prefix for the .npy files
NEW_PREFIX = 'image' # <-- Define the new prefix here

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print(f"Processing images in: {INPUT_FOLDER}")

# Find all PNG files in the input folder
png_files = glob.glob(os.path.join(INPUT_FOLDER, '*.png'))
png_files.sort() # Ensure they are processed in order (e.g., 001.png, 002.png, ...)

for input_path in png_files:
    try:
        # Get the original filename (e.g., '001.png')
        original_filename = os.path.basename(input_path)
        
        # --- NEW CODE SECTION FOR RENAMING ---
        
        # 1. Extract the numeric part (e.g., '001')
        # This regex looks for digits at the start of the filename (handles 001.png, 1234.png, etc.)
        match = re.search(r'^(\d+)', original_filename)
        
        if not match:
            print(f"Skipping {original_filename}: Filename does not start with a number. (Expected: 001.png)")
            continue
            
        numeric_part = match.group(1) # e.g., '001'
        
        # 2. Create the new output filename (e.g., 'image001.npy')
        output_filename = f"{NEW_PREFIX}{numeric_part}.npy"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        # --- END NEW CODE SECTION ---

        # 1. Load, convert to Grayscale, and transform to array
        img = Image.open(input_path).convert('L') # 'L' converts to single-channel 8-bit grayscale
        img_array = np.array(img, dtype=np.uint8) # Ensure data type is uint8 (0-255)
        
        # 2. Save the array to .npy
        np.save(output_path, img_array)
        
        print(f"Converted and Renamed: {original_filename} -> {output_filename}")

    except Exception as e:
        print(f"Failed to process {input_path}: {e}")

print("\nConversion complete.")