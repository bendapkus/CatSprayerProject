"""
Interactive ROI (Region of Interest) Setup Tool

Run this script to visually define the food detection zone on your camera view.
The ROI coordinates will be saved and can be used with the detector.
"""

import cv2
import numpy as np
import sys
import json

# Fix Jupyter kernel args
if any('ipykernel' in arg for arg in sys.argv):
    sys.argv = [sys.argv[0]]

# Global variables for mouse callback
drawing = False
start_point = None
end_point = None
roi_rect = None


def draw_rectangle(event, x, y, flags, param):
    """Mouse callback for drawing ROI rectangle."""
    global drawing, start_point, end_point, roi_rect

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        end_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        # Calculate final rectangle
        x1 = min(start_point[0], end_point[0])
        y1 = min(start_point[1], end_point[1])
        x2 = max(start_point[0], end_point[0])
        y2 = max(start_point[1], end_point[1])
        roi_rect = (x1, y1, x2, y2)


def setup_roi(image_path, output_file='roi_config.json', allow_multiple=True):
    """
    Interactive ROI setup - supports multiple ROI zones.

    Args:
        image_path: Path to sample image from your camera
        output_file: Where to save ROI configuration
        allow_multiple: If True, allows drawing multiple ROI zones

    Returns:
        List of normalized ROI coordinates [(x1, y1, x2, y2), ...]
    """
    global drawing, start_point, end_point, roi_rect

    saved_rois = []  # Store all saved ROIs

    # Load image
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError(f"Could not load image: {image_path}")

    height, width = original_img.shape[:2]

    # Scale image to fit screen if needed
    max_display_size = 1200
    scale_factor = 1.0

    if width > max_display_size or height > max_display_size:
        scale_factor = min(max_display_size / width, max_display_size / height)
        display_width = int(width * scale_factor)
        display_height = int(height * scale_factor)
        display_img = cv2.resize(original_img, (display_width, display_height))
        print(f"Image scaled by {scale_factor:.2f}x for display ({width}x{height} -> {display_width}x{display_height})")
    else:
        display_img = original_img.copy()

    display_height, display_width = display_img.shape[:2]

    # Create window
    window_name = 'ROI Setup - Draw rectangle, SPACE to save, R to reset, ESC to cancel'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, draw_rectangle)

    print("\n" + "="*70)
    print("MULTI-ROI SETUP TOOL")
    print("="*70)
    print(f"\nImage size: {width}x{height}")
    print(f"Display size: {display_width}x{display_height}")
    print(f"Multiple ROIs: {'Enabled' if allow_multiple else 'Disabled'}")
    print("\nInstructions:")
    print("  1. Click and drag to draw a rectangle around a food/table area")
    print("  2. Press SPACE to save this ROI")
    if allow_multiple:
        print("  3. Draw another ROI and press SPACE again (repeat as needed)")
        print("  4. Press ENTER when done with all ROIs")
    else:
        print("  3. ROI will be saved and tool will close")
    print("  R - Reset current ROI and redraw")
    print("  ESC - Cancel and quit")
    print("\nROI zones define high-sensitivity areas for food detection.")
    print("Boxes/bags only count as food when inside an ROI!")
    print("Cats are detected anywhere in the image.")
    print("="*70 + "\n")

    while True:
        # Create display image
        img_display = display_img.copy()

        # Draw all saved ROIs
        for idx, saved_roi in enumerate(saved_rois):
            x1, y1, x2, y2 = saved_roi
            # Scale to display coordinates
            disp_x1 = int(x1 * scale_factor)
            disp_y1 = int(y1 * scale_factor)
            disp_x2 = int(x2 * scale_factor)
            disp_y2 = int(y2 * scale_factor)

            cv2.rectangle(img_display, (disp_x1, disp_y1), (disp_x2, disp_y2), (0, 200, 0), 2)
            cv2.putText(img_display, f'ROI {idx+1}', (disp_x1 + 5, disp_y1 + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        # Draw current rectangle if being drawn or completed
        if start_point and end_point:
            cv2.rectangle(img_display, start_point, end_point, (0, 255, 0), 3)

            # Add label
            label_pos = (start_point[0] + 5, start_point[1] + 25)
            cv2.putText(img_display, 'Current ROI', label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show dimensions
            if roi_rect:
                x1, y1, x2, y2 = roi_rect
                rect_width = x2 - x1
                rect_height = y2 - y1
                dim_text = f"{rect_width}x{rect_height}px"
                cv2.putText(img_display, dim_text, (x1, y2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Show count at top
        if saved_rois:
            cv2.putText(img_display, f'Saved ROIs: {len(saved_rois)}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow(window_name, img_display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            print("Cancelled")
            cv2.destroyAllWindows()
            return None

        elif key == ord('r') or key == ord('R'):  # Reset
            start_point = None
            end_point = None
            roi_rect = None
            drawing = False
            print("Reset - draw new ROI")

        elif key == ord(' '):  # SPACE - save current ROI
            if roi_rect:
                # Scale coordinates back to original image size
                x1, y1, x2, y2 = roi_rect

                # Convert display coordinates to original image coordinates
                orig_x1 = int(x1 / scale_factor)
                orig_y1 = int(y1 / scale_factor)
                orig_x2 = int(x2 / scale_factor)
                orig_y2 = int(y2 / scale_factor)

                roi_pixels = (orig_x1, orig_y1, orig_x2, orig_y2)
                saved_rois.append(roi_pixels)

                print(f"\n✓ ROI #{len(saved_rois)} saved!")
                print(f"  Pixel coords: {roi_pixels}")
                print(f"  Size: {orig_x2-orig_x1}x{orig_y2-orig_y1} pixels")

                # Reset for next ROI
                start_point = None
                end_point = None
                roi_rect = None

                if allow_multiple:
                    print("  Draw another ROI or press ENTER to finish")
                else:
                    # Single ROI mode - save and exit
                    key = 13  # Simulate ENTER press
            else:
                print("⚠ Draw an ROI rectangle first!")

        elif key == 13 or key == 10:  # ENTER - finish and save all ROIs
            if saved_rois:
                # Convert all to normalized coordinates
                norm_rois = []
                for roi_pixels in saved_rois:
                    orig_x1, orig_y1, orig_x2, orig_y2 = roi_pixels
                    norm_roi = (
                        orig_x1 / width,
                        orig_y1 / height,
                        orig_x2 / width,
                        orig_y2 / height
                    )
                    norm_rois.append(norm_roi)

                # Save to file
                config = {
                    'rois_normalized': norm_rois,
                    'rois_pixels': saved_rois,
                    'image_size': (width, height),
                    'count': len(saved_rois),
                    'description': f'{len(saved_rois)} food detection ROI zone(s) - boxes/bags only food when inside'
                }

                with open(output_file, 'w') as f:
                    json.dump(config, f, indent=2)

                print(f"\n✓ All ROIs saved to: {output_file}")
                print(f"  Total ROIs: {len(saved_rois)}")
                for idx, (roi_pix, roi_norm) in enumerate(zip(saved_rois, norm_rois)):
                    print(f"  ROI {idx+1}: {roi_pix} -> normalized {roi_norm}")

                print("\nTo use these ROIs in your detector:")
                print("  from efficientdet_detector import detect_objects_efficientdet")
                print("  import json")
                print(f"  with open('{output_file}') as f:")
                print("      rois = json.load(f)['rois_normalized']")
                print("  result, dets = detect_objects_efficientdet(")
                print("      'image.jpg',")
                print("      food_rois=rois,")
                print("      food_roi_threshold=0.25")
                print("  )")

                cv2.destroyAllWindows()
                return norm_rois
            else:
                print("⚠ No ROIs to save!")

    cv2.destroyAllWindows()
    return None


def load_roi_config(config_file='roi_config.json'):
    """Load ROI from saved config file."""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return tuple(config['roi_normalized'])
    except FileNotFoundError:
        print(f"Config file {config_file} not found. Run setup_roi() first.")
        return None


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python setup_roi.py <image_path>")
        print("Example: python setup_roi.py cat.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    roi = setup_roi(image_path)

    if roi:
        print("\n✓ ROI setup complete!")
    else:
        print("\n✗ ROI setup cancelled")
