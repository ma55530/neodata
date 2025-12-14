"""
Glass Defect Detection

Detects cracks and damage in glass panels using:
1. Edge detection within glass regions (magenta in SAM)
2. Line detection for cracks (Hough lines)
3. Texture anomalies
"""

import cv2
import numpy as np
import os
import json
from typing import Dict, List, Tuple


ZONES = {
    "top-left": (0.0, 0.0, 0.33, 0.33),
    "top-center": (0.33, 0.0, 0.66, 0.33),
    "top-right": (0.66, 0.0, 1.0, 0.33),
    "center-left": (0.0, 0.33, 0.33, 0.66),
    "center": (0.33, 0.33, 0.66, 0.66),
    "center-right": (0.66, 0.33, 1.0, 0.66),
    "bottom-left": (0.0, 0.66, 0.33, 1.0),
    "bottom-center": (0.33, 0.66, 0.66, 1.0),
    "bottom-right": (0.66, 0.66, 1.0, 1.0),
}


def get_zone(x: float, y: float, w: int, h: int) -> str:
    """Get zone name for a point."""
    nx, ny = x / w, y / h
    for zone_name, (x1, y1, x2, y2) in ZONES.items():
        if x1 <= nx <= x2 and y1 <= ny <= y2:
            return zone_name
    return "unknown"


def extract_glass_mask(img: np.ndarray) -> np.ndarray:
    """Extract glass regions from segmented image (magenta/pink areas)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Magenta/pink = glass in SAM output
    # HSV for magenta: H ~140-170, high S, medium-high V
    mask1 = cv2.inRange(hsv, np.array([140, 50, 50]), np.array([180, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))  # Wrap around
    return cv2.bitwise_or(mask1, mask2)


def detect_cracks(img: np.ndarray, glass_mask: np.ndarray) -> List[Dict]:
    """
    Detect crack-like lines within glass regions.
    Cracks appear as thin, elongated edges.
    """
    cracks = []
    h, w = img.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply mask to focus on glass
    glass_gray = cv2.bitwise_and(gray, gray, mask=glass_mask)
    
    # Edge detection
    edges = cv2.Canny(glass_gray, 30, 100)
    
    # Morphological operations to enhance crack-like structures
    kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    vertical_cracks = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_line)
    
    kernel_line_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    horizontal_cracks = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_line_h)
    
    all_cracks = cv2.bitwise_or(vertical_cracks, horizontal_cracks)
    
    # Hough line detection for crack patterns (very strict - real cracks are long)
    lines = cv2.HoughLinesP(all_cracks, 1, np.pi/180, threshold=100,
                            minLineLength=120, maxLineGap=2)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            # Only very long lines are real cracks
            if 120 < length < 500:
                mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
                zone = get_zone(mid_x, mid_y, w, h)
                
                cracks.append({
                    'type': 'crack',
                    'start': (int(x1), int(y1)),
                    'end': (int(x2), int(y2)),
                    'length': float(length),
                    'zone': zone
                })
    
    return cracks


def detect_glass_damage(img: np.ndarray, glass_mask: np.ndarray) -> List[Dict]:
    """
    Detect damage/chips in glass using texture analysis.
    Damaged areas have irregular texture compared to smooth glass.
    """
    damages = []
    h, w = img.shape[:2]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate local variance (high variance = potential damage)
    kernel_size = 15
    local_mean = cv2.blur(gray.astype(float), (kernel_size, kernel_size))
    local_sq_mean = cv2.blur((gray.astype(float))**2, (kernel_size, kernel_size))
    local_var = local_sq_mean - local_mean**2
    
    # Only consider glass regions
    local_var[glass_mask == 0] = 0
    
    # Threshold for high variance regions (99th percentile - very strict)
    threshold = np.percentile(local_var[glass_mask > 0], 99) if np.any(glass_mask > 0) else 100
    damage_mask = (local_var > threshold).astype(np.uint8) * 255
    
    # Find contours of damaged regions
    contours, _ = cv2.findContours(damage_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000:  # Only large damage regions
            continue
        
        x, y, bw, bh = cv2.boundingRect(cnt)
        center_x, center_y = x + bw/2, y + bh/2
        zone = get_zone(center_x, center_y, w, h)
        
        damages.append({
            'type': 'damage',
            'bbox': (int(x), int(y), int(bw), int(bh)),
            'area': float(area),
            'zone': zone
        })
    
    return damages


def analyze_glass_coverage(img: np.ndarray, glass_mask: np.ndarray) -> Dict:
    """Analyze glass coverage per zone to detect missing glass."""
    h, w = img.shape[:2]
    zone_analysis = {}
    
    for zone_name, (x1, y1, x2, y2) in ZONES.items():
        px1, py1 = int(x1 * w), int(y1 * h)
        px2, py2 = int(x2 * w), int(y2 * h)
        
        zone_mask = glass_mask[py1:py2, px1:px2]
        zone_area = (px2 - px1) * (py2 - py1)
        
        if zone_area > 0:
            coverage = np.sum(zone_mask > 0) / zone_area
            zone_analysis[zone_name] = coverage
    
    return zone_analysis


def detect_glass_defects(image_path: str) -> Dict:
    """
    Main function to detect all glass defects.
    """
    img = cv2.imread(image_path)
    if img is None:
        return {'error': f'Could not load {image_path}'}
    
    h, w = img.shape[:2]
    
    # Extract glass mask
    glass_mask = extract_glass_mask(img)
    total_glass = np.sum(glass_mask > 0)
    
    if total_glass < 500:
        return {
            'image': os.path.basename(image_path),
            'glass_pixels': int(total_glass),
            'status': 'no_glass_detected',
            'defects': []
        }
    
    # Detect cracks
    cracks = detect_cracks(img, glass_mask)
    
    # Detect damage/chips
    damages = detect_glass_damage(img, glass_mask)
    
    # Analyze coverage
    coverage = analyze_glass_coverage(img, glass_mask)
    
    # Build defects list - only ONE crack and ONE damage per image
    defects = []
    
    # Add crack defect - only ONE (the longest)
    if cracks:
        longest_crack = max(cracks, key=lambda c: c['length'])
        # Only report if crack is long enough
        if longest_crack['length'] > 150:
            defects.append({
                'component': 'glass',
                'type': 'crack',
                'location': longest_crack['zone'],
                'confidence': min(0.6 + longest_crack['length'] / 200, 0.9),
                'details': f"length: {longest_crack['length']:.0f}px"
            })
    
    # Add damage defect - only ONE (the largest)
    if damages:
        largest_damage = max(damages, key=lambda d: d['area'])
        defects.append({
            'component': 'glass',
            'type': 'damaged',
            'location': largest_damage['zone'],
            'confidence': min(0.5 + largest_damage['area'] / 2000, 0.85),
            'details': f"area: {largest_damage['area']:.0f}px²"
        })
    
    # Check for missing glass (low coverage where expected)
    avg_coverage = np.mean(list(coverage.values())) if coverage else 0
    for zone, cov in coverage.items():
        if avg_coverage > 0.1 and cov < 0.05:  # Zone has much less glass than average
            defects.append({
                'component': 'glass',
                'type': 'missing',
                'location': zone,
                'confidence': 0.7,
                'details': f"coverage: {cov:.1%}"
            })
    
    return {
        'image': os.path.basename(image_path),
        'glass_pixels': int(total_glass),
        'glass_coverage': float(total_glass / (w * h)),
        'cracks_found': len(cracks),
        'damages_found': len(damages),
        'defects': defects
    }


if __name__ == "__main__":
    import sys
    
    print("="*60)
    print("Glass Defect Detection")
    print("="*60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    seg_dir = os.path.join(parent_dir, "segmented_negative")
    
    results = []
    
    for img_file in sorted(os.listdir(seg_dir))[:10]:
        if not img_file.lower().endswith(('.jpg', '.png')):
            continue
        
        img_path = os.path.join(seg_dir, img_file)
        result = detect_glass_defects(img_path)
        results.append(result)
        
        n_defects = len(result.get('defects', []))
        print(f"  {img_file[:40]}: {n_defects} glass defects")
    
    # Save
    output_path = os.path.join(base_dir, "glass_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {output_path}")
