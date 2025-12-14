"""
Edge-based Damage Detection for Tin Panels

Detects damaged (but present) tin by analyzing:
1. Edge density - dents/scratches create more edges
2. Texture irregularity - damaged areas have inconsistent texture
3. Color variation - damaged tin may have discoloration
"""

import cv2
import numpy as np
import os
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class DamageRegion:
    """A detected damage region."""
    zone: str
    damage_type: str  # "dent", "scratch", "discoloration"
    severity: float  # 0-1
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    
    def to_dict(self):
        return {
            'zone': self.zone,
            'damage_type': self.damage_type,
            'severity': round(self.severity, 2),
            'confidence': round(self.confidence, 2),
            'bbox': list(self.bbox)
        }


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


def extract_tin_mask(img: np.ndarray) -> np.ndarray:
    """Extract tin regions from segmented image (green areas)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Green = tin in SAM output
    mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
    return mask


def analyze_edge_density(img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Analyze edge density within tin regions.
    High edge density in tin = potential damage (dents, scratches).
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Only consider edges within tin regions
    tin_edges = cv2.bitwise_and(edges, edges, mask=mask)
    
    # Compute edge density per zone
    h, w = img.shape[:2]
    zone_stats = {}
    
    for zone_name, (x1, y1, x2, y2) in ZONES.items():
        px1, py1 = int(x1 * w), int(y1 * h)
        px2, py2 = int(x2 * w), int(y2 * h)
        
        zone_mask = mask[py1:py2, px1:px2]
        zone_edges = tin_edges[py1:py2, px1:px2]
        
        tin_area = np.sum(zone_mask > 0)
        edge_pixels = np.sum(zone_edges > 0)
        
        if tin_area > 100:  # Minimum tin area to analyze
            edge_density = edge_pixels / tin_area
            zone_stats[zone_name] = {
                'tin_area': tin_area,
                'edge_pixels': edge_pixels,
                'edge_density': edge_density
            }
    
    return tin_edges, zone_stats


def analyze_texture_variance(img: np.ndarray, mask: np.ndarray) -> Dict:
    """
    Analyze texture variance within tin regions.
    High variance = potential damage or irregularity.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    
    zone_variance = {}
    
    for zone_name, (x1, y1, x2, y2) in ZONES.items():
        px1, py1 = int(x1 * w), int(y1 * h)
        px2, py2 = int(x2 * w), int(y2 * h)
        
        zone_gray = gray[py1:py2, px1:px2]
        zone_mask = mask[py1:py2, px1:px2]
        
        # Get pixels within tin mask
        tin_pixels = zone_gray[zone_mask > 0]
        
        if len(tin_pixels) > 100:
            variance = np.var(tin_pixels)
            mean_intensity = np.mean(tin_pixels)
            
            # Local variance using sliding window
            kernel_size = 15
            local_mean = cv2.blur(zone_gray.astype(float), (kernel_size, kernel_size))
            local_sq_mean = cv2.blur((zone_gray.astype(float))**2, (kernel_size, kernel_size))
            local_var = local_sq_mean - local_mean**2
            local_var[zone_mask == 0] = 0
            
            max_local_var = np.max(local_var)
            mean_local_var = np.mean(local_var[zone_mask > 0])
            
            zone_variance[zone_name] = {
                'global_variance': variance,
                'mean_intensity': mean_intensity,
                'max_local_variance': max_local_var,
                'mean_local_variance': mean_local_var
            }
    
    return zone_variance


def analyze_color_anomalies(img: np.ndarray, mask: np.ndarray) -> Dict:
    """
    Detect color anomalies within tin regions.
    Discoloration, rust, or paint damage shows as color deviation.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]
    
    zone_color = {}
    
    for zone_name, (x1, y1, x2, y2) in ZONES.items():
        px1, py1 = int(x1 * w), int(y1 * h)
        px2, py2 = int(x2 * w), int(y2 * h)
        
        zone_hsv = hsv[py1:py2, px1:px2]
        zone_mask = mask[py1:py2, px1:px2]
        
        # Get HSV values within tin
        h_vals = zone_hsv[:,:,0][zone_mask > 0]
        s_vals = zone_hsv[:,:,1][zone_mask > 0]
        v_vals = zone_hsv[:,:,2][zone_mask > 0]
        
        if len(h_vals) > 100:
            zone_color[zone_name] = {
                'hue_std': np.std(h_vals),
                'sat_std': np.std(s_vals),
                'val_std': np.std(v_vals),
                'hue_range': np.ptp(h_vals),  # peak-to-peak
            }
    
    return zone_color


def detect_damage_regions(img: np.ndarray, mask: np.ndarray) -> List[DamageRegion]:
    """
    Detect specific damage regions using connected component analysis
    on high-edge-density areas.
    """
    damages = []
    h, w = img.shape[:2]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Multi-scale edge detection (higher thresholds to reduce noise)
    edges1 = cv2.Canny(gray, 50, 150)
    edges2 = cv2.Canny(gray, 80, 200)
    edges_combined = cv2.bitwise_or(edges1, edges2)
    
    # Only edges within tin
    tin_edges = cv2.bitwise_and(edges_combined, edges_combined, mask=mask)
    
    # Dilate to connect nearby edges
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(tin_edges, kernel, iterations=2)
    
    # Find contours of high-edge regions
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:  # Skip tiny regions
            continue
        
        x, y, bw, bh = cv2.boundingRect(cnt)
        
        # Analyze the region
        region_mask = mask[y:y+bh, x:x+bw]
        region_edges = tin_edges[y:y+bh, x:x+bw]
        
        tin_area = np.sum(region_mask > 0)
        edge_area = np.sum(region_edges > 0)
        
        if tin_area > 0:
            edge_ratio = edge_area / tin_area
            
            # High edge ratio indicates damage
            if edge_ratio > 0.15:  # Threshold for "significant" edges
                zone = get_zone(x + bw/2, y + bh/2, w, h)
                
                # Classify damage type based on shape
                aspect = bw / max(bh, 1)
                if aspect > 3 or aspect < 0.33:
                    damage_type = "scratch"  # Long thin = scratch
                else:
                    damage_type = "dent"  # Compact = dent
                
                severity = min(edge_ratio / 0.3, 1.0)  # Normalize to 0-1
                confidence = 0.5 + severity * 0.3  # 0.5-0.8 based on severity
                
                damages.append(DamageRegion(
                    zone=zone,
                    damage_type=damage_type,
                    severity=severity,
                    confidence=confidence,
                    bbox=(x, y, bw, bh)
                ))
    
    return damages


def detect_tin_damage(image_path: str) -> Dict:
    """
    Main function to detect damage in tin panels.
    Returns comprehensive damage analysis.
    """
    img = cv2.imread(image_path)
    if img is None:
        return {'error': f'Could not load {image_path}'}
    
    h, w = img.shape[:2]
    
    # Extract tin mask
    tin_mask = extract_tin_mask(img)
    total_tin = np.sum(tin_mask > 0)
    
    if total_tin < 1000:
        return {
            'image': os.path.basename(image_path),
            'tin_pixels': total_tin,
            'status': 'insufficient_tin',
            'damages': []
        }
    
    # Run all analyses
    edges, edge_stats = analyze_edge_density(img, tin_mask)
    texture_stats = analyze_texture_variance(img, tin_mask)
    color_stats = analyze_color_anomalies(img, tin_mask)
    
    # Detect specific damage regions
    damage_regions = detect_damage_regions(img, tin_mask)
    
    # Aggregate zone analysis
    zone_damage_scores = {}
    
    for zone_name in ZONES.keys():
        score = 0.0
        factors = []
        
        # Edge density factor
        if zone_name in edge_stats:
            ed = edge_stats[zone_name]['edge_density']
            if ed > 0.1:  # High edge density
                score += 0.3
                factors.append(f"high_edge_density:{ed:.2f}")
        
        # Texture variance factor
        if zone_name in texture_stats:
            mv = texture_stats[zone_name]['mean_local_variance']
            if mv > 500:  # High local variance
                score += 0.3
                factors.append(f"high_texture_var:{mv:.0f}")
        
        # Color variation factor
        if zone_name in color_stats:
            hs = color_stats[zone_name]['hue_std']
            if hs > 10:  # Significant hue variation
                score += 0.2
                factors.append(f"color_var:{hs:.1f}")
        
        # Damage region count
        zone_damages = [d for d in damage_regions if d.zone == zone_name]
        if zone_damages:
            score += 0.2 * min(len(zone_damages), 3)
            factors.append(f"damage_regions:{len(zone_damages)}")
        
        if score > 0:
            zone_damage_scores[zone_name] = {
                'score': min(score, 1.0),
                'factors': factors
            }
    
    # Convert damage regions to defects - only ONE per image (the most severe)
    defects = []
    if damage_regions:
        worst_damage = max(damage_regions, key=lambda d: d.confidence * d.severity)
        defects.append({
            'component': 'tin',
            'type': 'damaged',
            'location': worst_damage.zone,
            'confidence': worst_damage.confidence,
            'damage_type': worst_damage.damage_type,
            'severity': worst_damage.severity
        })
    
    # Add zone-level damage ONLY if no damage_regions and high score - ONE only
    if not defects:
        high_score_zones = [(z, info) for z, info in zone_damage_scores.items() if info['score'] > 0.5]
        if high_score_zones:
            best_zone, best_info = max(high_score_zones, key=lambda x: x[1]['score'])
            defects.append({
                'component': 'tin',
                'type': 'damaged',
                'location': best_zone,
                'confidence': best_info['score'],
                'damage_type': 'surface_anomaly',
                'factors': best_info['factors']
            })
    
    return {
        'image': os.path.basename(image_path),
        'tin_pixels': int(total_tin),
        'tin_coverage': float(total_tin / (w * h)),
        'zone_analysis': zone_damage_scores,
        'damage_regions': [d.to_dict() for d in damage_regions],
        'defects': defects
    }


def main():
    """Run damage detection on all segmented images."""
    print("="*60)
    print("Edge-Based Tin Damage Detection")
    print("="*60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    seg_dir = os.path.join(parent_dir, "segmented_negative")
    json_dir = os.path.join(parent_dir, "TRAIN", "negative")
    
    results = []
    
    # Load ground truth
    import re
    
    for img_file in sorted(os.listdir(seg_dir)):
        if not img_file.lower().endswith(('.jpg', '.png')):
            continue
        
        img_path = os.path.join(seg_dir, img_file)
        
        # Find GT
        base_name = img_file.replace('_facade_segmented', '').replace('_segmented', '')
        base_name = os.path.splitext(base_name)[0]
        json_name = re.sub(r'_(\d+)$', r' \1', base_name) + '.json'
        json_path = os.path.join(json_dir, json_name)
        
        gt = None
        if os.path.exists(json_path):
            with open(json_path) as f:
                gt_data = json.load(f)
                gt = gt_data.get('defects', [])
        
        # Run detection
        result = detect_tin_damage(img_path)
        result['ground_truth'] = gt
        results.append(result)
        
        # Print summary
        n_defects = len(result.get('defects', []))
        gt_str = f", GT: {gt}" if gt else ""
        print(f"  {img_file[:40]}: {n_defects} damage regions{gt_str}")
    
    # Save results
    output_path = os.path.join(base_dir, "damage_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {output_path}")
    
    # Compute metrics
    print("\n" + "="*60)
    print("MATCHING WITH GROUND TRUTH")
    print("="*60)
    
    # Location groups for matching
    location_groups = {
        'left': ['top-left', 'center-left', 'bottom-left'],
        'right': ['top-right', 'center-right', 'bottom-right'],
        'top': ['top-left', 'top-center', 'top-right'],
        'bottom': ['bottom-left', 'bottom-center', 'bottom-right'],
        'center': ['center', 'center-left', 'center-right', 'top-center', 'bottom-center'],
    }
    
    total_gt = 0
    total_det = 0
    total_match = 0
    
    for r in results:
        gt = r.get('ground_truth') or []
        defects = r.get('defects', [])
        
        # Only count tin-related GT
        tin_gt = [g for g in gt if g.get('component', '').lower() == 'tin']
        total_gt += len(tin_gt)
        total_det += len(defects)
        
        # Match
        for g in tin_gt:
            gt_loc = g.get('location', '').lower().replace('_', '-')
            gt_type = g.get('type', '').lower()
            
            # Get acceptable locations
            acceptable_locs = [gt_loc]
            for group_name, locs in location_groups.items():
                if group_name in gt_loc or gt_loc in group_name:
                    acceptable_locs.extend(locs)
            
            for d in defects:
                loc_match = any(d['location'] in acc or acc in d['location'] 
                               for acc in acceptable_locs)
                
                if loc_match and d['type'] in ['damaged', 'missing']:
                    total_match += 1
                    break
    
    print(f"Tin GT defects: {total_gt}")
    print(f"Detected damages: {total_det}")
    print(f"Matches: {total_match}")
    if total_gt > 0:
        print(f"Recall: {total_match/total_gt:.1%}")
    if total_det > 0:
        print(f"Precision: {total_match/total_det:.1%}")


if __name__ == "__main__":
    main()
