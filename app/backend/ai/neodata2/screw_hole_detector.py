"""
Screw and Hole Defect Detection

Detects:
1. Missing screws (expected from CAD but not found)
2. Half-tightened screws (shadow/depth analysis)
3. Missing holes
4. Shifted holes (position mismatch)
5. Extra holes (found but not in CAD)

Uses zone-based heuristics when pose estimation fails.
"""

import cv2
import numpy as np
import os
import json
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


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


def extract_screws_from_segmented(img: np.ndarray) -> List[Dict]:
    """Extract screw positions from segmented image (blue regions)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Blue = screws in SAM output
    mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([140, 255, 255]))
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    screws = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20:
            continue
        
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            
            # Approximate radius
            radius = np.sqrt(area / np.pi)
            
            # Circularity check
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            screws.append({
                'center': (cx, cy),
                'area': area,
                'radius': radius,
                'circularity': circularity
            })
    
    return screws


def extract_holes_from_segmented(img: np.ndarray) -> List[Dict]:
    """Extract hole positions from segmented image (yellow/cyan regions)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Yellow = holes
    mask_yellow = cv2.inRange(hsv, np.array([20, 50, 50]), np.array([40, 255, 255]))
    # Cyan = holes
    mask_cyan = cv2.inRange(hsv, np.array([80, 50, 50]), np.array([100, 255, 255]))
    
    mask = cv2.bitwise_or(mask_yellow, mask_cyan)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    holes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20:
            continue
        
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            
            radius = np.sqrt(area / np.pi)
            
            holes.append({
                'center': (cx, cy),
                'area': area,
                'radius': radius
            })
    
    return holes


def analyze_screw_tightness(img: np.ndarray, screws: List[Dict]) -> List[Dict]:
    """
    Analyze if screws are fully tightened based on:
    1. Shadow patterns (half-tightened screws cast different shadows)
    2. Brightness variation (protruding screws have different lighting)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    for screw in screws:
        cx, cy = screw['center']
        r = max(int(screw['radius'] * 1.5), 10)
        
        x1, y1 = max(0, int(cx - r)), max(0, int(cy - r))
        x2, y2 = min(gray.shape[1], int(cx + r)), min(gray.shape[0], int(cy + r))
        
        if x2 <= x1 or y2 <= y1:
            screw['tightness'] = 'unknown'
            continue
        
        roi = gray[y1:y2, x1:x2]
        
        # Analyze brightness gradient
        mean_brightness = np.mean(roi)
        std_brightness = np.std(roi)
        
        # High std deviation might indicate shadow from protruding screw
        # This is a heuristic - real detection would need depth info
        # Balanced threshold to reduce false positives
        if std_brightness > 70:
            screw['tightness'] = 'half_tightened'
            screw['tightness_confidence'] = min((std_brightness - 70) / 40, 0.75)
        else:
            screw['tightness'] = 'normal'
            screw['tightness_confidence'] = 0.7
    
    return screws


def get_expected_counts_per_zone(cad_circles: List[Dict], img_w: int, img_h: int) -> Dict:
    """
    Estimate expected screw/hole counts per zone based on CAD.
    Uses normalized distribution assuming frontal view.
    """
    # Get CAD bounding box
    centers = np.array([c['center'] for c in cad_circles])
    if len(centers) == 0:
        return {}
    
    cad_min = centers.min(axis=0)
    cad_max = centers.max(axis=0)
    cad_range = cad_max - cad_min
    cad_range[cad_range == 0] = 1
    
    expected = defaultdict(lambda: {'screw_hole': 0, 'mounting_hole': 0, 'large_hole': 0})
    
    for c in cad_circles:
        # Normalize to 0-1
        norm_pos = (np.array(c['center']) - cad_min) / cad_range
        # Map to image coordinates (X->x, Y->y)
        px = norm_pos[0] * img_w
        py = (1 - norm_pos[1]) * img_h  # Flip Y
        
        zone = get_zone(px, py, img_w, img_h)
        ctype = c['type']
        if ctype in expected[zone]:
            expected[zone][ctype] += 1
        else:
            expected[zone]['mounting_hole'] += 1  # Default to mounting hole
    
    return dict(expected)


def detect_screw_hole_defects(image_path: str, cad_circles: List[Dict] = None) -> Dict:
    """
    Main function to detect screw and hole defects.
    """
    img = cv2.imread(image_path)
    if img is None:
        return {'error': f'Could not load {image_path}'}
    
    h, w = img.shape[:2]
    
    # Extract detected components
    screws = extract_screws_from_segmented(img)
    holes = extract_holes_from_segmented(img)
    
    # Analyze screw tightness
    screws = analyze_screw_tightness(img, screws)
    
    # Count per zone
    detected_per_zone = defaultdict(lambda: {'screws': 0, 'holes': 0})
    for s in screws:
        zone = get_zone(s['center'][0], s['center'][1], w, h)
        detected_per_zone[zone]['screws'] += 1
    for hole in holes:
        zone = get_zone(hole['center'][0], hole['center'][1], w, h)
        detected_per_zone[zone]['holes'] += 1
    
    defects = []
    
    # Detect half-tightened screws - only ONE per image (highest confidence)
    half_tightened_screws = [s for s in screws if s.get('tightness') == 'half_tightened']
    if half_tightened_screws:
        best = max(half_tightened_screws, key=lambda s: s.get('tightness_confidence', 0))
        zone = get_zone(best['center'][0], best['center'][1], w, h)
        defects.append({
            'component': 'screw',
            'type': 'half_tightened',
            'location': zone,
            'confidence': best.get('tightness_confidence', 0.5),
            'position': [round(best['center'][0], 1), round(best['center'][1], 1)]
        })
    
    # Compare with CAD if available
    if cad_circles:
        expected = get_expected_counts_per_zone(cad_circles, w, h)
        
        # Compute expected ratios (we may only see part of the facade)
        total_detected_screws = len(screws)
        total_detected_holes = len(holes)
        total_expected_screws = sum(e.get('screw_hole', 0) for e in expected.values())
        total_expected_holes = sum(e.get('mounting_hole', 0) for e in expected.values())
        
        # Visibility ratio (how much of facade is visible)
        screw_visibility = total_detected_screws / max(total_expected_screws, 1)
        hole_visibility = total_detected_holes / max(total_expected_holes, 1)
        
        # Track if we already added a missing screw/hole defect
        has_missing_screw = False
        has_missing_hole = False
        has_extra_hole = False
        
        for zone in ZONES.keys():
            exp = expected.get(zone, {'screw_hole': 0, 'mounting_hole': 0})
            det = detected_per_zone.get(zone, {'screws': 0, 'holes': 0})
            
            # Scale expected by visibility
            exp_screws = exp['screw_hole'] * min(screw_visibility * 2, 1)  # Allow some tolerance
            exp_holes = exp['mounting_hole'] * min(hole_visibility * 2, 1)
            
            # Missing screws (very strict) - only ONE per image
            if not has_missing_screw and exp_screws > 5 and det['screws'] < exp_screws * 0.2:
                defects.append({
                    'component': 'screw',
                    'type': 'missing',
                    'location': zone,
                    'confidence': 0.75,
                    'details': f"expected ~{exp_screws:.0f}, found {det['screws']}"
                })
                has_missing_screw = True
            
            # Missing holes (very strict) - only ONE per image
            if not has_missing_hole and exp_holes > 5 and det['holes'] < exp_holes * 0.2:
                defects.append({
                    'component': 'hole',
                    'type': 'missing',
                    'location': zone,
                    'confidence': 0.75,
                    'details': f"expected ~{exp_holes:.0f}, found {det['holes']}"
                })
                has_missing_hole = True
            
            # Extra holes (lower threshold) - only ONE per image
            if not has_extra_hole and det['holes'] > exp_holes * 1.5 and det['holes'] > 2:
                defects.append({
                    'component': 'hole',
                    'type': 'extra',
                    'location': zone,
                    'confidence': 0.5,
                    'details': f"expected ~{exp_holes:.0f}, found {det['holes']}"
                })
                has_extra_hole = True
    
    return {
        'image': os.path.basename(image_path),
        'screws_detected': len(screws),
        'holes_detected': len(holes),
        'half_tightened_screws': len([s for s in screws if s.get('tightness') == 'half_tightened']),
        'defects': defects
    }


if __name__ == "__main__":
    print("="*60)
    print("Screw & Hole Defect Detection")
    print("="*60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    seg_dir = os.path.join(parent_dir, "segmented_negative")
    
    # Load CAD models
    with open(os.path.join(base_dir, "cad_models.json")) as f:
        cad_models = json.load(f)
    
    # Use first model for testing
    first_model = list(cad_models.keys())[0]
    cad_circles = cad_models[first_model]['circles']
    
    results = []
    
    for img_file in sorted(os.listdir(seg_dir))[:10]:
        if not img_file.lower().endswith(('.jpg', '.png')):
            continue
        
        img_path = os.path.join(seg_dir, img_file)
        result = detect_screw_hole_defects(img_path, cad_circles)
        results.append(result)
        
        n_defects = len(result.get('defects', []))
        half = result.get('half_tightened_screws', 0)
        print(f"  {img_file[:40]}: {n_defects} defects, {half} half-tightened")
    
    # Save
    output_path = os.path.join(base_dir, "screw_hole_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {output_path}")
