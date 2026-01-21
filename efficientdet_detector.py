"""
Improved Cat & Food Detector for Raspberry Pi
Uses EfficientDet Lite0 for better accuracy than SSD MobileNet

Cross-platform compatible: Mac (development) + Raspberry Pi (deployment)
"""

import numpy as np
import sys
import json

# Fix for Jupyter kernel args interfering with cv2
if any('ipykernel' in arg for arg in sys.argv):
    sys.argv = [sys.argv[0]]

import cv2

# Cross-platform TFLite import
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter


def load_labels(path):
    """Load class labels from a text file.

    Args:
        path: Path to labels file (one label per line)

    Returns:
        Dictionary mapping class IDs to label names
    """
    with open(path, 'r') as f:
        labels = {i: line.strip() for i, line in enumerate(f.readlines()) if line.strip()}
    return labels


def detect_objects_efficientdet(image_path,
                                  model_path='efficientdet_lite0.tflite',
                                  labels_path='coco_labels.txt',
                                  threshold=0.4,
                                  filter_classes=None,
                                  food_rois=None,
                                  food_roi_threshold=0.25,
                                  save_output=False,
                                  output_path='output.jpg'):
    """
    Detect objects using EfficientDet Lite model with multi-ROI support.

    Args:
        image_path: Path to input image
        model_path: Path to TFLite model file
        labels_path: Path to label map file
        threshold: Confidence threshold for general detections (0.0 to 1.0)
        filter_classes: List of classes to detect (e.g., ['cat', 'bowl', 'pizza'])
                       If None, detects all classes
        food_rois: Region(s) of Interest for food items. Can be:
                  - Single ROI: (x1, y1, x2, y2) in normalized coords (0-1)
                  - Multiple ROIs: [(x1, y1, x2, y2), (x1, y1, x2, y2), ...]
                  Food detections inside any ROI use food_roi_threshold (more sensitive)
                  Boxes/paper towels ONLY count as food when inside an ROI
                  Cat detections always use standard threshold (full image)
        food_roi_threshold: Lower threshold for food items inside ROI (default 0.25)
        save_output: Whether to save result image
        output_path: Where to save result image

    Returns:
        result_img: Image with drawn detections (RGB format)
        detections: List of detection dictionaries
    """
    # 1. Load model
    print(f"Loading model: {model_path}")
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 2. Load labels
    labels = load_labels(labels_path)
    print(f"Loaded {len(labels)} object classes")

    # 3. Get model input size
    input_shape = input_details[0]['shape']
    height, width = input_shape[1], input_shape[2]
    print(f"Model input size: {width}x{height}")

    # Check if model expects uint8 or float32 input
    input_dtype = input_details[0]['dtype']
    is_quantized = input_dtype == np.uint8
    print(f"Model type: {'Quantized (uint8)' if is_quantized else 'Float32'}")

    # 4. Load and preprocess image
    print(f"Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (width, height))

    # Prepare input tensor based on model type
    if is_quantized:
        # Quantized model expects uint8 [0-255]
        input_data = np.expand_dims(img_resized, axis=0).astype(np.uint8)
    else:
        # Float model expects normalized values
        input_data = np.expand_dims(img_resized, axis=0).astype(np.float32)
        # Some models expect [0,1], others [-1,1] - EfficientDet uses [0,1]
        input_data = input_data / 255.0

    # 5. Run inference
    print("Running object detection...")
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # 6. Get output tensors
    # EfficientDet outputs: [locations, classes, scores, num_detections]
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding boxes
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class IDs
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence scores
    num_detections = int(interpreter.get_tensor(output_details[3]['index'])[0])

    print(f"Model detected {num_detections} objects")

    # 7. Process detections
    img_height, img_width = img_rgb.shape[:2]
    result_img = img_rgb.copy()
    detections = []

    # Normalize food_rois to always be a list of ROIs
    if food_rois is None:
        roi_list = []
    elif isinstance(food_rois, tuple) and len(food_rois) == 4:
        # Single ROI as tuple
        roi_list = [food_rois]
    else:
        # Already a list of ROIs
        roi_list = food_rois

    print(f"Using {len(roi_list)} ROI zone(s) for food detection")

    for i in range(num_detections):
        class_id = int(classes[i])
        label = labels.get(class_id, f'Class_{class_id}')
        confidence = float(scores[i])

        # Convert normalized coordinates to pixel coordinates first
        ymin, xmin, ymax, xmax = boxes[i]
        left = int(xmin * img_width)
        top = int(ymin * img_height)
        right = int(xmax * img_width)
        bottom = int(ymax * img_height)

        # Calculate center of bounding box
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2

        # Check if object is inside any ROI
        in_roi = False
        for roi in roi_list:
            roi_x1 = roi[0] * img_width
            roi_y1 = roi[1] * img_height
            roi_x2 = roi[2] * img_width
            roi_y2 = roi[3] * img_height

            if roi_x1 <= center_x <= roi_x2 and roi_y1 <= center_y <= roi_y2:
                in_roi = True
                break

        # Categorize items
        # Items that are ONLY food when in ROI (cats like these for non-food reasons)
        roi_only_food = ['backpack', 'handbag', 'suitcase', 'vase']

        # Always food items
        always_food = ['bowl', 'cup', 'bottle', 'wine glass', 'fork', 'knife', 'spoon',
                      'dining table', 'kitchen', 'sink', 'refrigerator',
                      'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                      'hot dog', 'pizza', 'donut', 'cake']

        # Determine if this should be treated as food
        is_food_item = False
        if label in always_food:
            is_food_item = True
        elif label in roi_only_food and in_roi:
            # Only counts as food if inside an ROI
            is_food_item = True

        # Dynamic threshold based on object type and location
        current_threshold = threshold
        if is_food_item and in_roi:
            # Food item in ROI - use lower threshold
            current_threshold = food_roi_threshold

        # Apply threshold check
        if confidence >= current_threshold:

            # Skip if filtering and class not in filter list
            if filter_classes and label not in filter_classes:
                continue

            # Remap certain classes to "food" category
            # Includes containers, utensils, and surfaces where food might be
            food_related_items = [
                # Containers
                'bowl', 'cup', 'bottle', 'wine glass', 'vase',
                # Utensils
                'fork', 'knife', 'spoon',
                # Surfaces/furniture where food is placed
                'dining table', 'kitchen', 'sink', 'refrigerator',
                # Food itself
                'banana', 'apple', 'sandwich', 'orange', 'broccoli',
                'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                # Other containers cats might get into
                'backpack', 'handbag', 'suitcase'
            ]
            display_label = label
            if label in food_related_items:
                display_label = 'food'

            # Store detection with original label
            detections.append({
                'class': display_label,  # Show as "food" if remapped
                'original_class': label,  # Keep original for reference
                'class_id': class_id,
                'confidence': confidence,
                'box': (left, top, right, bottom)
            })

            # Choose color based on class
            if 'cat' in label.lower() or 'dog' in label.lower():
                color = (255, 0, 255)  # Magenta for pets
            elif label in food_related_items:
                color = (255, 255, 0)  # Yellow for food and food-related items
            else:
                color = (0, 255, 0)  # Green for other objects

            # Draw bounding box
            cv2.rectangle(result_img, (left, top), (right, bottom), color, 3)

            # Create label text (use display_label for bounding box)
            label_text = f"{display_label}: {confidence:.0%}"

            # Draw label background for better readability
            (label_width, label_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(result_img,
                         (left, top - label_height - baseline - 5),
                         (left + label_width, top),
                         color, -1)

            # Draw label text
            cv2.putText(result_img, label_text, (left, top - baseline - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Print with both display and original label
            if display_label != label:
                print(f"  [{i+1}] {display_label} ({label}): {confidence:.1%} at ({left},{top},{right},{bottom})")
            else:
                print(f"  [{i+1}] {label}: {confidence:.1%} at ({left},{top},{right},{bottom})")

    print(f"\nFiltered detections (>{threshold:.0%}): {len(detections)}")

    # Draw ROI boxes on top of everything for visualization
    for idx, roi in enumerate(roi_list):
        roi_x1 = int(roi[0] * img_width)
        roi_y1 = int(roi[1] * img_height)
        roi_x2 = int(roi[2] * img_width)
        roi_y2 = int(roi[3] * img_height)

        # Draw ROI as bright green rectangle with thicker lines
        cv2.rectangle(result_img, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 4)

        # Add label with background
        label_text = f'Food ROI {idx+1}'
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

        # Dark background for text
        cv2.rectangle(result_img,
                     (roi_x1, roi_y1 - text_height - baseline - 10),
                     (roi_x1 + text_width + 10, roi_y1),
                     (0, 200, 0), -1)

        # White text
        cv2.putText(result_img, label_text, (roi_x1 + 5, roi_y1 - baseline - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 8. Save if requested
    if save_output:
        result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, result_bgr)
        print(f"Saved result to {output_path}")

    return result_img, detections


def main():
    """Command-line interface for the detector."""
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python efficientdet_detector.py <image_path> [threshold] [--save]")
        print("Example: python efficientdet_detector.py cat.jpg 0.4 --save")
        sys.exit(1)

    image_path = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
    save_output = '--save' in sys.argv

    # Load ROIs from config if it exists
    rois = None
    try:
        with open('roi_config.json', 'r') as f:
            config = json.load(f)
            rois = config.get('rois_normalized') or config.get('roi_normalized')
            if rois:
                print(f"Loaded {len(rois) if isinstance(rois, list) else 1} ROI(s) from roi_config.json")
    except FileNotFoundError:
        print("No roi_config.json found - detecting across entire image")

    # Classes to detect for cat feeding monitor
    cat_food_classes = ['cat', 'dog', 'bowl', 'cup', 'banana', 'apple',
                        'sandwich', 'orange', 'pizza', 'cake', 'bottle']

    # Run detection
    result_img, detections = detect_objects_efficientdet(
        image_path=image_path,
        threshold=threshold,
        food_rois=rois,  # Pass ROIs!
        filter_classes=cat_food_classes,
        save_output=save_output
    )

    # Print summary
    print("\n" + "="*60)
    print("DETECTION SUMMARY")
    print("="*60)
    for i, det in enumerate(detections, 1):
        print(f"\n{i}. {det['class']}")
        print(f"   Confidence: {det['confidence']:.1%}")
        print(f"   Box: {det['box']}")


if __name__ == '__main__':
    main()
