import shutil
import os

def duplicate_images(folder_path, total_duplicates_needed=72):
    # The starting point for the new files
    start_index = 25
    end_index = 96
    
    # Identify the original 24 converted images (001.png through 024.png)
    # We will loop through these repeatedly to create the duplicates
    original_files = [f"{i:03}.png" for i in range(1, 25)]
    
    # Verify the first 24 images exist before starting
    missing_files = [f for f in original_files if not os.path.exists(os.path.join(folder_path, f))]
    if missing_files:
        print(f"Error: The following base files are missing: {missing_files}")
        return

    print(f"Duplicating images from 025.png to 096.png...")

    current_new_index = start_index
    while current_new_index <= end_index:
        for original in original_files:
            if current_new_index > end_index:
                break
                
            # Define the source and the new destination name
            source_path = os.path.join(folder_path, original)
            new_filename = f"{current_new_index:03}.png"
            destination_path = os.path.join(folder_path, new_filename)
            
            # Perform the copy
            shutil.copy2(source_path, destination_path)
            print(f"Created: {new_filename} (copy of {original})")
            
            current_new_index += 1

    print("Duplication complete.")

# Use '.' if the script is in the same folder as your 001.png-024.png files
duplicate_images('.')