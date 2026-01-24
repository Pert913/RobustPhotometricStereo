import cv2
import numpy as np
import os

def generate_mask(image_path, output_path, threshold_value=20):
    """
    Generates a binary mask for a foreground object in an image.
    
    Args:
        image_path (str): Path to the source image.
        output_path (str): Path to save the generated mask.
        threshold_value (int): The brightness value to separate foreground from background. 
                               A low value works best for bright objects on dark backgrounds.
    """
    # 1. Load the image
    if not os.path.exists(image_path):
        print(f"Error: Input image '{image_path}' not found.")
        return
        
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image '{image_path}'.")
        return

    # 2. Convert to grayscale
    # Color information is not needed for generating a silhouette mask.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Apply Binary Thresholding
    # Any pixel with a value greater than threshold_value becomes white (255),
    # and all others become black (0).
    # For your image, the background is near black, so a low threshold works well.
    _, binary_thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    # 4. Refine the mask to keep only the main object
    # Find contours (boundaries) of all white shapes in the thresholded image.
    contours, _ = cv2.findContours(binary_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a completely black image of the same size for our final mask.
    final_mask = np.zeros_like(gray)

    if contours:
        # Assuming the main object is the largest thing in the image,
        # find the largest contour by area.
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Draw the largest contour onto the black image and fill it with white.
        # This removes any small noise and ensures a solid shape.
        cv2.drawContours(final_mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

    # 5. Save the final mask image
    cv2.imwrite(output_path, final_mask)
    print(f"Successfully generated mask: {output_path}")

if __name__ == "__main__":
    # --- Configuration ---
    input_image = "001.png"  # Replace with your image's filename
    output_mask = "mask.png"
    # ---------------------

    generate_mask(input_image, output_mask)