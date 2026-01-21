# Cat Food Detector - Multi-ROI System

Detects cats and food items with smart ROI (Region of Interest) zones.

## Quick Start

### 1. Setup Multiple ROI Zones
```bash
python setup_roi.py cat.jpg
```
- Draw rectangles around tables/counters where food appears
- Press **SPACE** to save each ROI
- Press **ENTER** when done
- Saved to `roi_config.json`

### 2. Run Detection
```python
from efficientdet_detector import detect_objects_efficientdet
import json

# Load ROIs
with open('roi_config.json') as f:
    rois = json.load(f)['rois_normalized']

# Detect
result, detections = detect_objects_efficientdet(
    'image.jpg',
    food_rois=rois,
    food_roi_threshold=0.25,  # Sensitive inside ROI
    threshold=0.4,            # Normal elsewhere
    save_output=True
)
```

### 3. Or from command line
```bash
python efficientdet_detector.py cat.jpg 0.4 --save
```
(Loads ROIs from `roi_config.json` automatically in main())

---

## How It Works

### Detection Categories

**Always Food (detected anywhere):**
- bowl, cup, bottle, fork, knife, spoon
- dining table, kitchen, sink, refrigerator
- banana, apple, sandwich, pizza, cake, etc.

**Only Food in ROI (ignored outside ROI):**
- backpack, handbag, suitcase, vase
- **Prevents spraying cat for playing in boxes!**

**Always Detected:**
- cat, dog (magenta boxes, full image)

### Multi-ROI Behavior

- **Inside any ROI**: Food items detected with 25% confidence
- **Outside ROI**: Food needs 40% confidence
- **Boxes/bags**: Only count as food when inside ROI
- **Cats**: Always detected anywhere

---

## Files

- `efficientdet_detector.py` - Main detection script
- `setup_roi.py` - Interactive multi-ROI setup
- `table_to_roi.py` - Auto-detect tables and create ROIs
- `roi_config.json` - Saved ROI zones
- `efficientdet_lite0.tflite` - Fast model (Raspberry Pi)
- `efficientdet_lite2.tflite` - Accurate model (testing)
- `coco_labels.txt` - Object class names

---

## Examples

### Single ROI
```python
rois = [(0.2, 0.3, 0.8, 0.7)]  # One table
```

### Multiple ROIs
```python
rois = [
    (0.1, 0.2, 0.5, 0.6),  # Kitchen counter
    (0.5, 0.4, 0.9, 0.8)   # Dining table
]
```

### Auto-detect tables
```bash
python table_to_roi.py cat.jpg 0.3
```

---

## Performance

**Raspberry Pi 4:**
- Lite0: ~2-5 fps (recommended)
- Lite2: ~0.6-1.2 fps (too slow)

**Mac:**
- Either model works fine for testing

---

## Color Legend

- 🟣 **Magenta**: Cats/Dogs
- 🟡 **Yellow**: Food items
- 🟢 **Green**: ROI zones, other objects
