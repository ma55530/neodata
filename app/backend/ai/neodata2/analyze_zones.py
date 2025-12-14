"""Quick analysis of zone coverage for debugging."""
import cv2
import numpy as np
import os

def analyze_zones(image_path):
    """Show tin coverage per zone."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"  Could not load {image_path}")
        return
    
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Green = tin
    green_mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
    
    zones = {
        'TL': (0, 0, w//3, h//3),
        'TC': (w//3, 0, 2*w//3, h//3),
        'TR': (2*w//3, 0, w, h//3),
        'CL': (0, h//3, w//3, 2*h//3),
        'C': (w//3, h//3, 2*w//3, 2*h//3),
        'CR': (2*w//3, h//3, w, 2*h//3),
        'BL': (0, 2*h//3, w//3, h),
        'BC': (w//3, 2*h//3, 2*w//3, h),
        'BR': (2*w//3, 2*h//3, w, h),
    }
    
    coverages = {}
    for z, (x1, y1, x2, y2) in zones.items():
        zone_area = (x2 - x1) * (y2 - y1)
        zone_mask = green_mask[y1:y2, x1:x2]
        coverage = np.sum(zone_mask > 0) / zone_area * 100
        coverages[z] = coverage
    
    # Find minimum
    min_zone = min(coverages, key=coverages.get)
    min_cov = coverages[min_zone]
    
    print(f"  Coverages: TL={coverages['TL']:.0f}% TC={coverages['TC']:.0f}% TR={coverages['TR']:.0f}%")
    print(f"             CL={coverages['CL']:.0f}% C={coverages['C']:.0f}%   CR={coverages['CR']:.0f}%")
    print(f"             BL={coverages['BL']:.0f}% BC={coverages['BC']:.0f}% BR={coverages['BR']:.0f}%")
    print(f"  Lowest: {min_zone} at {min_cov:.0f}%")
    return coverages


# Analyze key images
import json
base = r"c:\Users\Jura Slibar\Desktop\NeoData"
seg_dir = os.path.join(base, "segmented_negative")
json_dir = os.path.join(base, "TRAIN", "negative")

for img_file in sorted(os.listdir(seg_dir))[:10]:
    if not img_file.endswith('.jpg'):
        continue
    
    img_path = os.path.join(seg_dir, img_file)
    
    # Find GT
    import re
    base_name = img_file.replace('_facade_segmented', '').replace('_segmented', '')
    base_name = os.path.splitext(base_name)[0]
    json_name = re.sub(r'_(\d+)$', r' \1', base_name) + '.json'
    json_path = os.path.join(json_dir, json_name)
    
    gt = "N/A"
    if os.path.exists(json_path):
        with open(json_path) as f:
            gt_data = json.load(f)
            if 'defects' in gt_data:
                gt = ", ".join([f"{d['component']} {d['type']} at {d['location']}" for d in gt_data['defects']])
    
    print(f"\n{img_file[:35]}...")
    print(f"  GT: {gt}")
    analyze_zones(img_path)
