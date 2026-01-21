"""
Batch Detection Script - Process Multiple Images

Tests camera placement by detecting objects in all images from a folder.
ROIs are disabled for this script (detects across entire image).
"""

import sys
import os
import json
from pathlib import Path

# Fix Jupyter kernel args
if any('ipykernel' in arg for arg in sys.argv):
    sys.argv = [sys.argv[0]]

from efficientdet_detector import detect_objects_efficientdet


def batch_detect(input_folder,
                 output_folder='batch_results',
                 threshold=0.4,
                 model_path='efficientdet_lite0.tflite'):
    """
    Process all images in a folder.

    Args:
        input_folder: Folder containing images to process
        output_folder: Where to save detection results
        threshold: Detection confidence threshold
        model_path: Path to TFLite model
    """
    # Supported image formats
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    # Find all images
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"Error: Folder '{input_folder}' not found")
        return

    image_files = [
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    if not image_files:
        print(f"No images found in '{input_folder}'")
        print(f"Looking for: {', '.join(image_extensions)}")
        return

    # Create output folder
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)

    print("="*70)
    print("BATCH DETECTION - Camera Placement Testing")
    print("="*70)
    print(f"\nInput folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Found {len(image_files)} image(s)")
    print(f"Threshold: {threshold}")
    print(f"ROIs: Disabled (full image detection)")
    print("="*70 + "\n")

    # Process each image
    all_results = []

    for idx, img_file in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] Processing: {img_file.name}")
        print("-" * 60)

        try:
            # Run detection (no ROIs)
            result_img, detections = detect_objects_efficientdet(
                image_path=str(img_file),
                model_path=model_path,
                threshold=threshold,
                food_rois=None,  # No ROIs for camera placement testing
                save_output=True,
                output_path=str(output_path / f"detected_{img_file.name}")
            )

            # Store results
            result = {
                'filename': img_file.name,
                'detections': detections,
                'cat_detected': any(d['class'] in ['cat', 'dog'] for d in detections),
                'food_detected': any(d['class'] == 'food' for d in detections),
                'total_objects': len(detections)
            }
            all_results.append(result)

            # Print summary for this image
            cats = [d for d in detections if d['class'] in ['cat', 'dog']]
            food = [d for d in detections if d['class'] == 'food']

            print(f"  ✓ Detected {len(detections)} objects:")
            if cats:
                print(f"    - {len(cats)} cat/dog(s)")
            if food:
                print(f"    - {len(food)} food item(s)")
            if not detections:
                print("    - No objects detected")

        except Exception as e:
            print(f"  ✗ Error processing {img_file.name}: {e}")
            all_results.append({
                'filename': img_file.name,
                'error': str(e)
            })

    # Save summary report
    summary_file = output_path / 'detection_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print final summary
    print("\n" + "="*70)
    print("BATCH DETECTION COMPLETE")
    print("="*70)
    print(f"\nProcessed: {len(image_files)} images")
    print(f"Results saved to: {output_folder}/")
    print(f"Summary: {summary_file}")

    # Statistics
    successful = [r for r in all_results if 'error' not in r]
    with_cats = sum(1 for r in successful if r.get('cat_detected'))
    with_food = sum(1 for r in successful if r.get('food_detected'))

    print(f"\nStatistics:")
    print(f"  - Successful: {len(successful)}/{len(image_files)}")
    print(f"  - Images with cats: {with_cats}")
    print(f"  - Images with food: {with_food}")

    # Camera placement recommendations
    print(f"\n" + "="*70)
    print("CAMERA PLACEMENT TIPS:")
    print("="*70)
    if with_cats < len(successful) * 0.5:
        print("⚠ Cat detected in less than 50% of images")
        print("  → Consider adjusting camera angle or position")
    else:
        print("✓ Good cat detection coverage")

    if with_food > 0:
        print(f"✓ Food items detected in {with_food} image(s)")
        print("  → Use these images to set up ROIs with setup_roi.py")
    else:
        print("⚠ No food items detected")
        print("  → Ensure food/bowls are visible in camera view")

    print("\nNext steps:")
    print("  1. Review images in batch_results/ folder")
    print("  2. Choose best camera angle (where cat + food are visible)")
    print("  3. Run setup_roi.py on a sample image to define food zones")
    print("="*70)


def main():
    """Command-line interface."""
    if len(sys.argv) < 2:
        print("Usage: python batch_detect.py <folder_path> [threshold]")
        print("\nExample:")
        print("  python batch_detect.py camera_test_images/")
        print("  python batch_detect.py camera_test_images/ 0.3")
        print("\nProcesses all images in the folder and saves results to 'batch_results/'")
        sys.exit(1)

    folder = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4

    batch_detect(folder, threshold=threshold)


if __name__ == '__main__':
    main()
