"""
Seal (Brtva) Defect Detection

Detects:
1. Missing seals (no black regions where expected)
2. Damaged/torn seals (irregular edges, gaps in seal line)

SAM segments seals as BLACK regions.
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


def extract_seal_mask(img: np.ndarray) -> np.ndarray:
    """Extract seal regions from segmented image (black/dark areas)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Black = seals in SAM output (low saturation, low value)
    mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 80, 60]))
    return mask


def analyze_seal_continuity(seal_mask: np.ndarray) -> Dict:
    """
    Analyze if seals are continuous or have breaks/tears.
    Seals should be continuous lines along edges.
    """
    h, w = seal_mask.shape
    
    # Find contours of seal regions
    contours, _ = cv2.findContours(seal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    seal_info = {
        'total_seal_pixels': int(np.sum(seal_mask > 0)),
        'num_seal_segments': len(contours),
        'segments': []
    }
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:  # Skip tiny noise
            continue
            
        # Get bounding box
        x, y, bw, bh = cv2.boundingRect(cnt)
        
        # Calculate aspect ratio - seals are usually elongated
        aspect_ratio = max(bw, bh) / max(min(bw, bh), 1)
        
        # Perimeter
        perimeter = cv2.arcLength(cnt, True)
        
        # Compactness (4*pi*area / perimeter^2) - lower = more elongated
        compactness = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        center_x, center_y = x + bw/2, y + bh/2
        zone = get_zone(center_x, center_y, w, h)
        
        seal_info['segments'].append({
            'area': area,
            'bbox': (x, y, bw, bh),
            'aspect_ratio': aspect_ratio,
            'compactness': compactness,
            'zone': zone,
            'is_elongated': aspect_ratio > 3,  # Seals are usually elongated
        })
    
    return seal_info


def detect_seal_damage(img: np.ndarray, seal_mask: np.ndarray) -> List[Dict]:
    """
    Detect damaged/torn seals by analyzing edge irregularity.
    Damaged seals have rough, irregular edges.
    """
    damages = []
    h, w = img.shape[:2]
    
    # Find edges within seal regions
    edges = cv2.Canny(seal_mask, 50, 150)
    
    # Look for areas with high edge density (irregular seal edges)
    kernel = np.ones((15, 15), np.float32) / 225
    edge_density = cv2.filter2D(edges.astype(np.float32), -1, kernel)
    
    # High edge density in seal areas = potential damage
    seal_edge_density = edge_density * (seal_mask > 0).astype(np.float32)
    
    # Find regions with high edge density
    threshold = np.percentile(seal_edge_density[seal_mask > 0], 95) if np.any(seal_mask > 0) else 100
    damage_mask = (seal_edge_density > threshold).astype(np.uint8) * 255
    
    contours, _ = cv2.findContours(damage_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
            
        x, y, bw, bh = cv2.boundingRect(cnt)
        center_x, center_y = x + bw/2, y + bh/2
        zone = get_zone(center_x, center_y, w, h)
        
        damages.append({
            'type': 'torn',
            'bbox': (x, y, bw, bh),
            'area': area,
            'zone': zone
        })
    
    return damages


def detect_seal_defects(image_path: str) -> Dict:
    """
    Main function to detect seal defects in a segmented image.
    Returns ONE defect per type (missing or damaged).
    """
    img = cv2.imread(image_path)
    if img is None:
        return {'error': f'Could not load {image_path}', 'defects': []}
    
    h, w = img.shape[:2]
    
    # Extract seal mask (black regions)
    seal_mask = extract_seal_mask(img)
    total_seal = np.sum(seal_mask > 0)
    
    defects = []
    
    # Analyze seal continuity
    seal_info = analyze_seal_continuity(seal_mask)
    
    # Check for missing seals - if very few seal pixels detected
    seal_coverage = total_seal / (w * h)
    
    # Analyze per zone
    zone_seal_coverage = {}
    for zone_name, (x1, y1, x2, y2) in ZONES.items():
        px1, py1 = int(x1 * w), int(y1 * h)
        px2, py2 = int(x2 * w), int(y2 * h)
        
        zone_mask = seal_mask[py1:py2, px1:px2]
        zone_area = (px2 - px1) * (py2 - py1)
        zone_seal_coverage[zone_name] = np.sum(zone_mask > 0) / zone_area if zone_area > 0 else 0
    
    # Find zone with lowest coverage (potential missing seal)
    if zone_seal_coverage:
        avg_coverage = np.mean(list(zone_seal_coverage.values()))
        
        # Find zones with significantly lower coverage
        missing_zones = [(z, c) for z, c in zone_seal_coverage.items() 
                        if avg_coverage > 0.005 and c < avg_coverage * 0.1]
        
        if missing_zones:
            # Report ONE missing seal (worst zone)
            worst_zone, worst_coverage = min(missing_zones, key=lambda x: x[1])
            defects.append({
                'component': 'seal',
                'type': 'missing',
                'location': worst_zone,
                'confidence': 0.75,
                'details': f'coverage: {worst_coverage*100:.1f}%'
            })
    
    # Detect seal damage/tears
    damages = detect_seal_damage(img, seal_mask)
    
    if damages:
        # Report ONE damaged seal (largest damage area)
        worst_damage = max(damages, key=lambda d: d['area'])
        # Only if damage is significant
        if worst_damage['area'] > 500:
            defects.append({
                'component': 'seal',
                'type': 'damaged',
                'location': worst_damage['zone'],
                'confidence': 0.75,
                'details': f"area: {worst_damage['area']:.0f}px²"
            })
    
    return {
        'image': os.path.basename(image_path),
        'seal_pixels': int(total_seal),
        'seal_coverage': float(seal_coverage),
        'num_seal_segments': seal_info['num_seal_segments'],
        'defects': defects
    }


def main():
    """Test seal detection on all segmented images."""
    print("="*60)
    print("Seal (Brtva) Defect Detection")
    print("="*60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    seg_dir = os.path.join(base_dir, "segmented_negative")
    
    if not os.path.exists(seg_dir):
        seg_dir = os.path.join(os.path.dirname(base_dir), "segmented_negative")
    
    results = []
    
    for img_file in sorted(os.listdir(seg_dir))[:10]:  # Test on first 10
        if not img_file.lower().endswith(('.jpg', '.png')):
            continue
        
        img_path = os.path.join(seg_dir, img_file)
        result = detect_seal_defects(img_path)
        results.append(result)
        
        n = len(result.get('defects', []))
        coverage = result.get('seal_coverage', 0) * 100
        print(f"  {img_file[:40]}: {n} defects, seal coverage: {coverage:.2f}%")
    
    print(f"\nTotal images tested: {len(results)}")
    print(f"Total seal defects: {sum(len(r.get('defects',[])) for r in results)}")


if __name__ == "__main__":
    main()
