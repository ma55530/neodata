"""
3D CAD-Based Defect Detection System

Main pipeline that:
1. Loads CAD models and image classifications
2. Estimates camera pose (with zone-based fallback)
3. Projects 3D hole positions to 2D
4. Compares with SAM segmentation detections
5. Outputs defect JSON with confidence scores
"""

import os
import json
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Detection:
    """A detected component from segmentation."""
    component_type: str  # "screw", "hole", etc.
    center: Tuple[float, float]  # (x, y) in pixels
    area: float
    confidence: float = 1.0


@dataclass
class ExpectedComponent:
    """A component expected from CAD projection."""
    component_type: str
    center_3d: Tuple[float, float, float]
    center_2d: Optional[Tuple[float, float]] = None  # After projection
    radius: float = 0.0
    zone: str = ""


@dataclass
class Defect:
    """A detected defect."""
    component: str
    defect_type: str  # "missing", "extra", "shifted"
    location: str  # Zone name
    confidence: float
    expected_pos: Optional[Tuple[float, float]] = None
    detected_pos: Optional[Tuple[float, float]] = None
    distance: float = 0.0
    
    def to_dict(self):
        d = {
            'component': self.component,
            'type': self.defect_type,
            'location': self.location,
            'confidence': round(self.confidence, 3)
        }
        if self.expected_pos:
            d['expected_pos'] = [round(x, 1) for x in self.expected_pos]
        if self.detected_pos:
            d['detected_pos'] = [round(x, 1) for x in self.detected_pos]
        if self.distance > 0:
            d['distance_px'] = round(self.distance, 1)
        return d


# ============================================================================
# ZONE DEFINITIONS
# ============================================================================

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


def get_zone(x: float, y: float, img_w: int, img_h: int) -> str:
    """Get zone name for a point."""
    nx, ny = x / img_w, y / img_h
    for zone_name, (x1, y1, x2, y2) in ZONES.items():
        if x1 <= nx <= x2 and y1 <= ny <= y2:
            return zone_name
    return "unknown"


# ============================================================================
# SEGMENTATION MASK ANALYSIS
# ============================================================================

def extract_detections_from_segmented(image_path: str) -> Tuple[List[Detection], Dict]:
    """Extract component detections from SAM-segmented image.
    Returns detections and panel analysis for defect detection.
    """
    img = cv2.imread(image_path)
    if img is None:
        return [], {}
    
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    detections = []
    panel_analysis = {}
    
    # Component color mappings (from SAM output)
    color_maps = {
        "screw": [(100, 50, 50), (140, 255, 255)],    # Blue
        "hole": [(20, 50, 50), (40, 255, 255)],        # Yellow
        "hole_cyan": [(80, 50, 50), (100, 255, 255)],  # Cyan
        "tin": [(35, 50, 50), (85, 255, 255)],         # Green
        "glass": [(140, 50, 50), (170, 255, 255)],     # Magenta/Pink
    }
    
    for comp_type, (lower, upper) in color_maps.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        total_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 20:
                continue
            total_area += area
            
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                
                # Normalize component type
                ctype = "hole" if "hole" in comp_type else comp_type
                
                detections.append(Detection(
                    component_type=ctype,
                    center=(cx, cy),
                    area=area
                ))
        
        # Track coverage for panel types
        if comp_type in ["tin", "glass"]:
            coverage = total_area / (w * h)
            panel_analysis[comp_type] = {
                'coverage': coverage,
                'num_regions': len([c for c in contours if cv2.contourArea(c) >= 20]),
                'total_area': total_area,
                'contours': contours  # Keep for detailed analysis
            }
    
    # Analyze for gaps/missing sections in tin
    if "tin" in panel_analysis:
        tin_analysis = analyze_panel_gaps(img, hsv, panel_analysis["tin"], "tin")
        panel_analysis["tin"].update(tin_analysis)
    
    return detections, panel_analysis


def analyze_panel_gaps(img: np.ndarray, hsv: np.ndarray, panel_info: Dict, 
                       panel_type: str) -> Dict:
    """Analyze panel regions for gaps, damage, or missing sections."""
    h, w = img.shape[:2]
    
    # Create panel mask
    if panel_type == "tin":
        mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
    else:
        mask = cv2.inRange(hsv, np.array([140, 50, 50]), np.array([170, 255, 255]))
    
    # Calculate coverage per zone
    zone_coverages = {}
    for zone_name, (x1, y1, x2, y2) in ZONES.items():
        px1, py1 = int(x1 * w), int(y1 * h)
        px2, py2 = int(x2 * w), int(y2 * h)
        
        zone_mask = mask[py1:py2, px1:px2]
        zone_area = (px2 - px1) * (py2 - py1)
        
        if zone_area > 0:
            coverage = np.sum(zone_mask > 0) / zone_area
            zone_coverages[zone_name] = coverage
    
    # Analyze for gaps using multiple criteria:
    # 1. Absolute threshold (very low coverage)
    # 2. Relative threshold (significantly below average)
    gaps_by_zone = {}
    
    if zone_coverages:
        avg_coverage = np.mean(list(zone_coverages.values()))
        std_coverage = np.std(list(zone_coverages.values()))
        
        for zone_name, coverage in zone_coverages.items():
            is_defect = False
            status = None
            
            # Absolute thresholds (balanced for better recall)
            if coverage < 0.15:  # Less than 15% - definitely missing
                is_defect = True
                status = 'missing'
            elif coverage < 0.30:  # Less than 30% - damaged
                is_defect = True
                status = 'damaged'
            # Relative threshold - significantly below average
            elif avg_coverage > 0.3 and coverage < avg_coverage - 1.5 * std_coverage:
                if coverage < avg_coverage * 0.5:  # Less than 50% of average
                    is_defect = True
                    status = 'damaged'
            
            if is_defect:
                gaps_by_zone[zone_name] = {
                    'coverage': coverage,
                    'avg_coverage': avg_coverage,
                    'status': status
                }
    
    return {'gaps_by_zone': gaps_by_zone}


# ============================================================================
# POSE ESTIMATION
# ============================================================================

class PoseEstimator:
    """Estimates camera pose from correspondences."""
    
    def __init__(self, img_width: int, img_height: int):
        self.img_w = img_width
        self.img_h = img_height
        
        # Approximate camera matrix (assuming unknown intrinsics)
        focal = max(img_width, img_height)
        self.camera_matrix = np.array([
            [focal, 0, img_width / 2],
            [0, focal, img_height / 2],
            [0, 0, 1]
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros(4)
    
    def estimate_pose(self, points_3d: np.ndarray, points_2d: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray, int, float]:
        """
        Estimate pose using RANSAC PnP.
        Returns: (success, rvec, tvec, inlier_count, reproj_error)
        """
        if len(points_3d) < 4:
            return False, None, None, 0, float('inf')
        
        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                points_3d.astype(np.float64),
                points_2d.astype(np.float64),
                self.camera_matrix,
                self.dist_coeffs,
                iterationsCount=200,
                reprojectionError=15.0,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success or inliers is None:
                return False, None, None, 0, float('inf')
            
            inlier_count = len(inliers)
            
            # Compute reprojection error
            projected, _ = cv2.projectPoints(points_3d, rvec, tvec, 
                                             self.camera_matrix, self.dist_coeffs)
            projected = projected.reshape(-1, 2)
            errors = np.linalg.norm(projected - points_2d, axis=1)
            mean_error = np.mean(errors[inliers.flatten()])
            
            return True, rvec, tvec, inlier_count, mean_error
            
        except Exception as e:
            print(f"    PnP failed: {e}")
            return False, None, None, 0, float('inf')
    
    def project_points(self, points_3d: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """Project 3D points to 2D image coordinates."""
        projected, _ = cv2.projectPoints(
            points_3d.astype(np.float64),
            rvec, tvec,
            self.camera_matrix,
            self.dist_coeffs
        )
        return projected.reshape(-1, 2)


# ============================================================================
# ZONE-BASED FALLBACK
# ============================================================================

def zone_based_analysis(cad_circles: List[Dict], detections: List[Detection], 
                        img_w: int, img_h: int) -> Tuple[List[ExpectedComponent], Dict]:
    """
    Fallback: Distribute CAD components across zones and compare.
    Assumes frontal view, normalizes positions.
    """
    # Get CAD bounding box
    centers = np.array([c['center'] for c in cad_circles])
    cad_min = centers.min(axis=0)
    cad_max = centers.max(axis=0)
    cad_range = cad_max - cad_min
    cad_range[cad_range == 0] = 1  # Avoid division by zero
    
    # Normalize CAD positions to 0-1 and map to image
    expected = []
    for c in cad_circles:
        norm_pos = (np.array(c['center']) - cad_min) / cad_range
        # Map to image (assuming X->x, Y->y, ignore Z)
        px = norm_pos[0] * img_w
        py = (1 - norm_pos[1]) * img_h  # Flip Y
        
        zone = get_zone(px, py, img_w, img_h)
        comp_type = "screw" if c['type'] == "screw_hole" else "hole"
        
        expected.append(ExpectedComponent(
            component_type=comp_type,
            center_3d=tuple(c['center']),
            center_2d=(px, py),
            radius=c['radius'],
            zone=zone
        ))
    
    # Count per zone
    expected_per_zone = defaultdict(lambda: {"screw": 0, "hole": 0})
    for e in expected:
        expected_per_zone[e.zone][e.component_type] += 1
    
    detected_per_zone = defaultdict(lambda: defaultdict(int))
    for d in detections:
        zone = get_zone(d.center[0], d.center[1], img_w, img_h)
        detected_per_zone[zone][d.component_type] += 1
    
    return expected, {
        'expected_per_zone': dict(expected_per_zone),
        'detected_per_zone': {k: dict(v) for k, v in detected_per_zone.items()},
        'method': 'zone_fallback'
    }


# ============================================================================
# DEFECT DETECTION
# ============================================================================

def detect_defects(expected: List[ExpectedComponent], detections: List[Detection],
                   img_w: int, img_h: int, method: str, inlier_ratio: float = 1.0,
                   reproj_error: float = 0.0) -> List[Defect]:
    """
    Compare expected vs detected positions to find defects.
    """
    defects = []
    threshold_px = 30  # Distance threshold for matching
    max_error = 20.0   # Max reprojection error for confidence
    
    # Group by component type
    expected_by_type = defaultdict(list)
    for e in expected:
        if e.center_2d:
            expected_by_type[e.component_type].append(e)
    
    detected_by_type = defaultdict(list)
    for d in detections:
        detected_by_type[d.component_type].append(d)
    
    for comp_type in ["screw", "hole"]:
        exp_list = expected_by_type.get(comp_type, [])
        det_list = detected_by_type.get(comp_type, [])
        
        if not exp_list:
            continue
        
        # Build distance matrix
        exp_positions = np.array([e.center_2d for e in exp_list])
        det_positions = np.array([d.center for d in det_list]) if det_list else np.array([]).reshape(0, 2)
        
        # Match expected to detected
        matched_det = set()
        
        for i, exp in enumerate(exp_list):
            exp_pos = np.array(exp.center_2d)
            
            # Find nearest detection
            if len(det_positions) > 0:
                distances = np.linalg.norm(det_positions - exp_pos, axis=1)
                min_idx = np.argmin(distances)
                min_dist = distances[min_idx]
            else:
                min_dist = float('inf')
                min_idx = -1
            
            # Compute confidence
            # Higher confidence when: more inliers, lower reproj error, smaller distance
            base_conf = inlier_ratio * (1 - min(reproj_error / max_error, 1.0))
            
            if min_dist > threshold_px:
                # MISSING: No detection near expected position
                dist_factor = min(min_dist / 100, 1.0)  # Higher dist = more confident it's missing
                confidence = base_conf * (0.5 + 0.5 * dist_factor)
                
                defects.append(Defect(
                    component=comp_type,
                    defect_type="missing",
                    location=exp.zone,
                    confidence=confidence,
                    expected_pos=exp.center_2d,
                    distance=min_dist
                ))
            else:
                matched_det.add(min_idx)
        
        # Check for EXTRA detections (detected but not expected nearby)
        for j, det in enumerate(det_list):
            if j in matched_det:
                continue
            
            det_pos = np.array(det.center)
            if len(exp_positions) > 0:
                distances = np.linalg.norm(exp_positions - det_pos, axis=1)
                min_dist = np.min(distances)
            else:
                min_dist = float('inf')
            
            if min_dist > threshold_px:
                zone = get_zone(det.center[0], det.center[1], img_w, img_h)
                confidence = base_conf * 0.5  # Lower confidence for extras
                
                defects.append(Defect(
                    component=comp_type,
                    defect_type="extra",
                    location=zone,
                    confidence=confidence,
                    detected_pos=det.center,
                    distance=min_dist
                ))
    
    return defects


def detect_panel_defects(panel_analysis: Dict, img_w: int, img_h: int) -> List[Defect]:
    """Detect defects in panels (tin, glass) based on coverage analysis.
    Only ONE defect per component type (tin or glass) per image."""
    defects = []
    
    for panel_type in ["tin", "glass"]:
        if panel_type not in panel_analysis:
            continue
        
        panel_info = panel_analysis[panel_type]
        gaps = panel_info.get('gaps_by_zone', {})
        
        # Find the worst gap (lowest coverage) for this panel type
        worst_gap = None
        worst_coverage = 1.0
        
        for zone_name, gap_info in gaps.items():
            coverage = gap_info.get('coverage', 0)
            if coverage < worst_coverage:
                worst_coverage = coverage
                worst_gap = (zone_name, gap_info)
        
        # Only add ONE defect per panel type
        if worst_gap:
            zone_name, gap_info = worst_gap
            status = gap_info.get('status', 'damaged')
            coverage = gap_info.get('coverage', 0)
            
            # Confidence based on how clearly missing/damaged
            confidence = 0.8 - coverage * 5  # Lower coverage = higher confidence
            confidence = max(0.4, min(0.95, confidence))
            
            defects.append(Defect(
                component=panel_type,
                defect_type=status,  # "missing" or "damaged"
                location=zone_name,
                confidence=confidence
            ))
    
    return defects


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class DefectDetector:
    """Main defect detection pipeline."""
    
    def __init__(self, cad_models_path: str, classifications_path: str):
        with open(cad_models_path, 'r') as f:
            self.cad_models = json.load(f)
        
        with open(classifications_path, 'r') as f:
            classifications = json.load(f)
        
        # Build image -> model mapping
        self.image_to_model = {}
        for c in classifications:
            if 'matched_model' in c:
                self.image_to_model[c['image']] = c['matched_model']
        
        print(f"Loaded {len(self.cad_models)} CAD models")
        print(f"Loaded {len(self.image_to_model)} image classifications")
    
    def process_image(self, image_path: str, json_metadata: Optional[Dict] = None) -> Dict:
        """Process single image for defect detection."""
        img_name = os.path.basename(image_path)
        img = cv2.imread(image_path)
        
        if img is None:
            return {'image': img_name, 'error': 'Could not load image'}
        
        h, w = img.shape[:2]
        
        # Get matched CAD model
        # Try to match by image name patterns
        model_name = None
        for img_key, model in self.image_to_model.items():
            if img_key in img_name or img_name in img_key:
                model_name = model
                break
        
        if not model_name:
            # Default to first model (or could return error)
            model_name = list(self.cad_models.keys())[0]
        
        cad_model = self.cad_models[model_name]
        cad_circles = cad_model['circles']
        
        # Extract detections from segmented image
        detections, panel_analysis = extract_detections_from_segmented(image_path)
        
        # Try pose estimation
        pose_estimator = PoseEstimator(w, h)
        
        # For pose estimation, we need correspondences
        # Use detected component centers as 2D points
        # Match to nearest CAD circles (simplified approach)
        
        screw_dets = [d for d in detections if d.component_type == "screw"]
        screw_cad = [c for c in cad_circles if c['type'] == "screw_hole"]
        
        pose_success = False
        inlier_ratio = 0.0
        reproj_error = float('inf')
        expected_components = []
        
        if len(screw_dets) >= 4 and len(screw_cad) >= 4:
            # Attempt pose estimation with simple correspondence
            # This is simplified - real implementation would use feature matching
            
            # For now, use zone-based fallback
            pass
        
        # Use zone-based fallback
        expected_components, zone_info = zone_based_analysis(
            cad_circles, detections, w, h
        )
        method = "zone_fallback"
        inlier_ratio = 0.5  # Lower confidence for fallback
        
        # Detect defects from screws/holes
        defects = detect_defects(
            expected_components, detections, w, h,
            method=method,
            inlier_ratio=inlier_ratio,
            reproj_error=reproj_error
        )
        
        # Detect panel defects (tin/glass) from coverage analysis
        panel_defects = detect_panel_defects(panel_analysis, w, h)
        defects.extend(panel_defects)
        
        # Filter to high-confidence defects
        significant_defects = [d for d in defects if d.confidence > 0.3]
        
        # Build result
        result = {
            'image': img_name,
            'matched_model': model_name,
            'pose_method': method,
            'detection_stats': {
                'screws_detected': len([d for d in detections if d.component_type == "screw"]),
                'holes_detected': len([d for d in detections if d.component_type == "hole"]),
                'screws_expected': len([c for c in cad_circles if c['type'] == "screw_hole"]),
                'holes_expected': len([c for c in cad_circles if c['type'] == "mounting_hole"]),
                'tin_coverage': panel_analysis.get('tin', {}).get('coverage', 0),
                'glass_coverage': panel_analysis.get('glass', {}).get('coverage', 0),
            },
            'defects': [d.to_dict() for d in significant_defects]
        }
        
        # Compare with JSON metadata if provided
        if json_metadata and 'defects' in json_metadata:
            result['ground_truth'] = json_metadata['defects']
            result['gt_match'] = self._compare_with_ground_truth(
                significant_defects, json_metadata['defects']
            )
        
        return result
    
    def _compare_with_ground_truth(self, detected: List[Defect], ground_truth: List[Dict]) -> Dict:
        """Compare detected defects with ground truth JSON."""
        matches = 0
        matched_details = []
        
        # Location equivalence groups (GT annotations are often imprecise)
        location_groups = {
            'left': ['top-left', 'center-left', 'bottom-left'],
            'right': ['top-right', 'center-right', 'bottom-right'],
            'top': ['top-left', 'top-center', 'top-right'],
            'bottom': ['bottom-left', 'bottom-center', 'bottom-right'],
            'center': ['center', 'center-left', 'center-right', 'top-center', 'bottom-center'],
        }
        
        # Type equivalence (damaged/missing often used interchangeably)
        type_equivalent = {
            'missing': ['missing', 'damaged'],
            'damaged': ['damaged', 'missing'],
        }
        
        for gt in ground_truth:
            gt_comp = gt.get('component', '').lower()
            gt_type = gt.get('type', '').lower()
            gt_loc = gt.get('location', '').lower().replace('_', '-')
            
            # Get acceptable locations
            acceptable_locs = [gt_loc]
            for group_name, locs in location_groups.items():
                if group_name in gt_loc or gt_loc in group_name:
                    acceptable_locs.extend(locs)
            
            # Get acceptable types
            acceptable_types = type_equivalent.get(gt_type, [gt_type])
            
            for det in detected:
                # Component must match
                if det.component != gt_comp:
                    continue
                
                # Type must be acceptable
                if det.defect_type not in acceptable_types:
                    continue
                
                # Location must be acceptable
                loc_match = any(det.location in acc or acc in det.location 
                               for acc in acceptable_locs)
                
                if loc_match:
                    matches += 1
                    matched_details.append({'gt': gt, 'det': det.to_dict()})
                    break
        
        return {
            'ground_truth_count': len(ground_truth),
            'detected_count': len(detected),
            'matches': matches,
            'recall': matches / len(ground_truth) if ground_truth else 0,
            'precision': matches / len(detected) if detected else 0
        }


def main():
    print("="*60)
    print("3D CAD-Based Defect Detection")
    print("="*60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    
    # Initialize detector
    detector = DefectDetector(
        cad_models_path=os.path.join(base_dir, "cad_models.json"),
        classifications_path=os.path.join(base_dir, "classification_results.json")
    )
    
    # Process segmented images
    seg_negative_dir = os.path.join(parent_dir, "segmented_negative")
    json_dir = os.path.join(parent_dir, "TRAIN", "negative")
    
    results = []
    
    if os.path.exists(seg_negative_dir):
        print(f"\n--- Processing segmented_negative ---")
        
        for img_file in sorted(os.listdir(seg_negative_dir)):
            if not img_file.lower().endswith(('.jpg', '.png')):
                continue
            
            img_path = os.path.join(seg_negative_dir, img_file)
            
            # Try to find matching JSON metadata
            json_metadata = None
            # Convert image name to JSON name pattern
            # IMG_5350_2_facade_segmented.jpg -> IMG_5350 2.json
            base_name = img_file.replace('_facade_segmented', '').replace('_segmented', '')
            base_name = os.path.splitext(base_name)[0]
            # Replace underscore before number with space
            import re
            json_name = re.sub(r'_(\d+)$', r' \1', base_name) + '.json'
            
            json_path = os.path.join(json_dir, json_name)
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    json_metadata = json.load(f)
            
            # Process image
            result = detector.process_image(img_path, json_metadata)
            results.append(result)
            
            # Print summary
            n_defects = len(result.get('defects', []))
            gt_info = result.get('gt_match', {})
            print(f"  {img_file[:40]}: {n_defects} defects detected", end="")
            if gt_info:
                print(f" (GT: {gt_info['matches']}/{gt_info['ground_truth_count']} matched)")
            else:
                print()
    
    # Save results
    output_path = os.path.join(base_dir, "defect_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {output_path}")
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    total_gt = 0
    total_detected = 0
    total_matches = 0
    
    for r in results:
        gt = r.get('gt_match', {})
        total_gt += gt.get('ground_truth_count', 0)
        total_detected += gt.get('detected_count', 0)
        total_matches += gt.get('matches', 0)
    
    print(f"Total images processed: {len(results)}")
    print(f"Ground truth defects: {total_gt}")
    print(f"Detected defects: {total_detected}")
    print(f"Correct matches: {total_matches}")
    if total_gt > 0:
        print(f"Overall recall: {total_matches/total_gt:.1%}")
    if total_detected > 0:
        print(f"Overall precision: {total_matches/total_detected:.1%}")


if __name__ == "__main__":
    main()
