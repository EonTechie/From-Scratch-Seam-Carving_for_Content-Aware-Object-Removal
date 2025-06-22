#!/usr/bin/env python3
"""
ManuelObjectVanish Demo Script
==============================

This script demonstrates the object removal capabilities of the project.
It processes sample images and shows before/after comparisons.
"""

import cv2
import numpy as np
import os
from segmentation import add_image, add_mask, compute_energy, reweight_energy, iterative_object_removal, restore_image_to_original_size

def display_comparison(original, processed, title="Before vs After"):
    """Display original and processed images side by side."""
    # Resize images to same height for comparison
    height = min(original.shape[0], processed.shape[0])
    width1 = int(original.shape[1] * height / original.shape[0])
    width2 = int(processed.shape[1] * height / processed.shape[0])
    
    original_resized = cv2.resize(original, (width1, height))
    processed_resized = cv2.resize(processed, (width2, height))
    
    # Create side-by-side comparison
    comparison = np.hstack([original_resized, processed_resized])
    
    # Add text labels
    cv2.putText(comparison, "BEFORE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "AFTER", (width1 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Save comparison
    cv2.imwrite(f"output/{title.lower().replace(' ', '_')}_comparison.jpg", comparison)
    print(f"Comparison saved: output/{title.lower().replace(' ', '_')}_comparison.jpg")

def demo_object_removal():
    """Run the main demo with sample images."""
    print("--- ManuelObjectVanish Demo ---")
    print("----------------------------------------")
    
    # Check if input directory exists
    if not os.path.exists('input'):
        print("Error: Input directory not found. Please ensure images are in the 'input' folder.")
        return
    
    # Process each sample image
    samples = [
        ('ball.jpg', 'ball_mask.jpg', 'Ball Removal'),
        ('chick.jpg', 'chick_mask.jpg', 'Chick Removal'),
        ('object.jpg', 'object_mask.jpg', 'Object Removal')
    ]
    
    for image_name, mask_name, title in samples:
        print(f"\nProcessing: {title}")
        
        # Load image and mask
        image_path = f'input/{image_name}'
        mask_path = f'input/{mask_name}'
        
        if not os.path.exists(image_path):
            print(f"Warning: Skipping {title}. Reason: {image_path} not found.")
            continue
            
        if not os.path.exists(mask_path):
            print(f"Warning: Skipping {title}. Reason: {mask_path} not found.")
            continue
        
        # Load and process
        original_image = add_image(image_path)
        mask = add_mask(mask_path)
        
        if original_image is None or mask is None:
            print(f"Error: Failed to load necessary files for {title}.")
            continue
        
        # Process the image
        energy_matrix = compute_energy(original_image)
        reweighted_energy = reweight_energy(energy_matrix, mask)
        processed_image = iterative_object_removal(original_image, mask, reweighted_energy)
        
        # Restore to original size to get the final result
        final_image = restore_image_to_original_size(processed_image, original_image.shape[1])
        
        # Save final result
        output_name = f"output/demo_{image_name.replace('.jpg', '')}_final.jpg"
        cv2.imwrite(output_name, final_image)
        
        # Create comparison with the final, restored image
        display_comparison(original_image, final_image, title)
        
        print(f"Status: {title} completed successfully.")
    
    print("\nDemo finished. Please check the 'output' folder for results.")
    print("\nGenerated files in output directory:")
    for file in os.listdir('output'):
        if file.startswith('demo_') or file.endswith('_comparison.jpg'):
            print(f"   - {file}")

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Run the demo
    demo_object_removal() 