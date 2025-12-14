"""
Combined Defect Detection Pipeline v3 (Improved Segmentation)

Detects ALL defect types using segmented_output_improved:
1. Lim (tin): Nedostaje, oštećen
2. Vijak (screw): Nedostaje, na pola pritegnut
3. Rupa (hole): Nedostaje, pomaknuta, višak
4. Staklo (glass): Pukotina, oštećenje
5. Brtva (seal): Nedostaje, oštećena/strgana

Uses:
- Original JPEGs for classification (v2 classifier)
- Mask JSON files from SAM improved segmentation (_masks.json)
- Component coverage analysis from pixel-level masks

Outputs unified defect results with confidence scores.
"""

import os
import sys
import json
import re

# =============================================================================
# HYPERPARAMETERS - Tune these values to optimize detection
# =============================================================================

# TIN DETECTION THRESHOLDS
GAP_THRESHOLD = 13          # Zone coverage < this = potential gap
GOOD_COVERAGE = 14          # Zone coverage >= this = has tin  
CENTER_MIN_COVERAGE = 5   # Minimum center zone coverage       

# TIN DETECTION CONDITIONS
MIN_GAPS_STRONG = 3        # Gaps for high confidence
MIN_GAPS_MEDIUM = 1        # Gaps for medium confidence        
MIN_GAPS_WEAK = 2          # Gaps for low confidence
MIN_COVERED_FOR_WEAK = 1   # Covered zones for weak detection  

# CONFIDENCE SCORES
CONFIDENCE_STRONG = 0.85   # Confidence for 3+ gaps
CONFIDENCE_MEDIUM = 0.75   # Confidence for 2 gaps
CONFIDENCE_WEAK = 0.7     # Confidence for 1 gap
CONFIDENCE_THRESHOLD = 0.6  # Min confidence to report


# SCREW/GLASS/SEAL DETECTION (currently disabled)
ENABLE_SCREW_DETECTION = True
ENABLE_GLASS_DETECTION = True
ENABLE_SEAL_DETECTION = False

# =============================================================================


def combine_defects(*defect_lists) -> list:
    """
    Merge defects from all detectors.
    Keep ONE defect per (component, type) combination per image.
    E.g., one "screw missing" AND one "screw half_tightened" can both exist.
    """
    # First collect all defects
    all_defects = []
    for defect_list in defect_lists:
        if not defect_list:
            continue
        for d in defect_list:
            if isinstance(d, dict):
                all_defects.append(d)
    
    # Deduplicate: keep only ONE per (component, type) - pick highest confidence
    best_by_type = {}
    for d in all_defects:
        key = (d.get('component', ''), d.get('type', ''))
        conf = d.get('confidence', 0.5)
        if key not in best_by_type or conf > best_by_type[key].get('confidence', 0):
            best_by_type[key] = d
    
    return list(best_by_type.values())


def run_combined_detection():
    """Run the full combined detection pipeline."""
    print("="*60)
    print("COMBINED DEFECT DETECTION PIPELINE v3 (improved segmentation)")
    print("="*60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    
    # Use new segmented_output_improved structure (both positive and negative)
    seg_improved_base = os.path.join(parent_dir, "segmented_output_improved")
    
    # Validation ground truth folder
    validation_dir = os.path.join(parent_dir, "validation")
    
    # Check if improved segmentation exists
    if not os.path.exists(seg_improved_base):
        print(f"ERROR: {seg_improved_base} not found!")
        print("Run the SAM_Improved_Segmentation.ipynb notebook first.")
        return []
    
    # Load v2 classification results (from original JPEGs)
    v2_class_path = os.path.join(base_dir, "classification_results_v2.json")
    if os.path.exists(v2_class_path):
        with open(v2_class_path) as f:
            v2_classifications = json.load(f)
        # Build mapping: image_name -> model
        jpeg_to_model = {c['image']: c['matched_model'] for c in v2_classifications}
        print(f"Loaded {len(jpeg_to_model)} JPEG classifications (v2)")
    else:
        jpeg_to_model = {}
        print("WARNING: v2 classifications not found, run image_classifier_v2.py first")
    
    # Load CAD models
    with open(os.path.join(base_dir, "cad_models.json")) as f:
        cad_models = json.load(f)
    
    results = []
    
    # Process both positive and negative folders
    for folder in ['negative', 'positive']:
        seg_improved_dir = os.path.join(seg_improved_base, folder)
        if not os.path.exists(seg_improved_dir):
            print(f"  Skipping {folder}/ - not found")
            continue
        
        print(f"\n--- Processing {folder}/ ---")
        
        # Iterate over mask JSON files in segmented_output_improved
        for mask_json in sorted(os.listdir(seg_improved_dir)):
            if not mask_json.endswith('_masks.json'):
                continue
            
            mask_json_path = os.path.join(seg_improved_dir, mask_json)
            
            # Load mask data
            with open(mask_json_path) as f:
                mask_data = json.load(f)
            
            # Extract base name: IMG_5349_2_facade_masks.json -> IMG_5349_2_facade
            base_name = mask_json.replace('_masks.json', '')
            
            # Map to original JPEG name: IMG_5349_2_facade -> IMG_5349 2.jpg
            jpeg_base = base_name.replace('_facade', '')
            jpeg_name = re.sub(r'_(\d+)$', r' \1', jpeg_base) + '.jpg'
            
            # Also try without number suffix: IMG_5470_facade -> IMG_5470.jpg
            jpeg_name_simple = jpeg_base + '.jpg'
            
            # Get CAD model from v2 classifier (original JPEG)
            matched_model = jpeg_to_model.get(jpeg_name)
            if not matched_model:
                jpeg_name_alt = jpeg_base.replace('_', ' ') + '.jpg'
                matched_model = jpeg_to_model.get(jpeg_name_alt)
            if not matched_model:
                matched_model = jpeg_to_model.get(jpeg_name_simple, list(cad_models.keys())[0])
            
            # Find GT JSON in validation folder
            # Try multiple name formats
            gt = None
            json_names_to_try = [
                re.sub(r'_(\d+)$', r' \1', jpeg_base) + '.json',  # IMG_5349 2.json
                jpeg_base + '.json',  # IMG_5470.json
                jpeg_base.replace('_', ' ') + '.json',  # IMG 5470.json
            ]
            
            for json_name in json_names_to_try:
                json_path = os.path.join(validation_dir, json_name)
                if os.path.exists(json_path):
                    with open(json_path) as f:
                        gt_data = json.load(f)
                        gt = gt_data.get('defects', [])
                    break
            
            # Get image size from mask data
            img_size = mask_data.get('size', [1536, 688])
            w, h = img_size[0], img_size[1]
            total_pixels = w * h
            
            # Get CAD circles for the matched model
            cad_circles = cad_models[matched_model]['circles']
            
            # Analyze components from mask data
            components = mask_data.get('components', {})
            
            # Calculate coverage from mask JSON
            tin_masks = components.get('tin', [])
            glass_masks = components.get('glass', [])
            screw_masks = components.get('screw', [])
            hole_masks = components.get('hole', [])
            seal_masks = components.get('seal', [])
            
            # Total coverage per component (coverage_percent is already in %)
            tin_coverage = sum(m.get('coverage_percent', 0) for m in tin_masks)  # in %
            glass_coverage = sum(m.get('coverage_percent', 0) for m in glass_masks)
            screw_count = len(screw_masks)
            hole_count = len(hole_masks)
            seal_coverage = sum(m.get('coverage_percent', 0) for m in seal_masks)
            
            # Detect defects based on ZONE analysis
            # Divide image into 3x3 grid and check each zone
            defects = []
            
            # Analyze tin by zones - look for gaps where tin SHOULD be
            zones = ['top-left', 'top', 'top-right', 'left', 'center', 'right', 
                     'bottom-left', 'bottom', 'bottom-right']
            zone_tin = {z: 0 for z in zones}
            
            for m in tin_masks:
                bbox = m.get('bbox')  # [x_min, y_min, x_max, y_max]
                if bbox:
                    x_min, y_min, x_max, y_max = bbox
                    cx = (x_min + x_max) / 2
                    cy = (y_min + y_max) / 2
                    
                    # Determine zone (3x3 grid)
                    col = 'left' if cx < w/3 else ('right' if cx > 2*w/3 else '')
                    row = 'top' if cy < h/3 else ('bottom' if cy > 2*h/3 else '')
                    
                    if row and col:
                        zone = f"{row}-{col}"
                    elif row:
                        zone = row
                    elif col:
                        zone = col
                    else:
                        zone = 'center'
                    
                    zone_tin[zone] += m.get('coverage_percent', 0)
            
            # === TIN DEFECT DETECTION ===
            # Strategy: Zone asymmetry detection (uses hyperparameters from top of file)
            expected_tin_zones = ['top', 'bottom', 'left', 'right']
            
            # Find zones with low coverage (potential gaps)
            tin_gaps = [z for z in expected_tin_zones if zone_tin[z] < GAP_THRESHOLD]
            
            # Find zones with good coverage
            well_covered = [z for z in expected_tin_zones if zone_tin[z] >= GOOD_COVERAGE]
            
            center_coverage = zone_tin.get('center', 0)
            
            # Zone asymmetry detection
            if center_coverage > CENTER_MIN_COVERAGE and len(well_covered) >= 1:
                if len(tin_gaps) >= MIN_GAPS_STRONG:
                    defects.append({
                        'component': 'tin',
                        'type': 'missing',
                        'confidence': CONFIDENCE_STRONG,
                        'zones_missing': tin_gaps,
                        'zones_present': well_covered,
                        'zone_coverage': zone_tin,
                        'source': 'zone_analysis'
                    })
                elif len(tin_gaps) >= MIN_GAPS_MEDIUM and len(well_covered) >= 1:
                    defects.append({
                        'component': 'tin',
                        'type': 'missing',
                        'confidence': CONFIDENCE_MEDIUM,
                        'zones_missing': tin_gaps,
                        'zones_present': well_covered,
                        'zone_coverage': zone_tin,
                        'source': 'zone_analysis'
                    })
                elif len(tin_gaps) >= MIN_GAPS_WEAK and len(well_covered) >= MIN_COVERED_FOR_WEAK:
                    defects.append({
                        'component': 'tin',
                        'type': 'missing',
                        'confidence': CONFIDENCE_WEAK,
                        'zones_missing': tin_gaps,
                        'zones_present': well_covered,
                        'zone_coverage': zone_tin,
                        'source': 'zone_analysis'
                    })
            
            # === SCREW DEFECT DETECTION ===
            if ENABLE_SCREW_DETECTION:
                # Label as missing screw if a circular hole exists without a screw in it
                import numpy as np
                screw_centers = []
                for m in screw_masks:
                    bbox = m.get('bbox')
                    if bbox:
                        x_min, y_min, x_max, y_max = bbox
                        cx = (x_min + x_max) / 2
                        cy = (y_min + y_max) / 2
                        screw_centers.append((cx, cy))
                # For each hole, check if a screw is present nearby
                threshold = min(w, h) * 0.08  # 8% of image size as matching threshold
                missing_flagged = False
                for m in hole_masks:
                    if missing_flagged:
                        break
                    bbox = m.get('bbox')
                    area = m.get('area_pixels', m.get('area', 0))
                    if bbox and area > 0:
                        # Only consider very small, very circular holes (area < 0.2% of image, aspect ratio ~1)
                        if area < 0.002 * w * h:
                            x_min, y_min, x_max, y_max = bbox
                            width = x_max - x_min
                            height = y_max - y_min
                            aspect_ratio = width / height if height > 0 else 0
                            if 0.6 < aspect_ratio < 1.5:
                                cx = (x_min + x_max) / 2
                                cy = (y_min + y_max) / 2
                                # Check if any screw is close to this hole
                                found = False
                                for sx, sy in screw_centers:
                                    dist = np.hypot(sx - cx, sy - cy)
                                    if dist < threshold:
                                        found = True
                                        break
                                if not found:
                                    defects.append({
                                        'component': 'screw',
                                        'type': 'missing',
                                        'confidence': 0.85,
                                        'hole_position': [cx, cy],
                                        'source': 'hole_without_screw'
                                    })
                                    missing_flagged = True
            
            # === GLASS DEFECT DETECTION ===
            if ENABLE_GLASS_DETECTION:
                broken_glass_masks = components.get('broken_glass', [])
                if broken_glass_masks:
                    defects.append({
                        'component': 'glass',
                        'type': 'cracked',
                        'confidence': 0.85,
                        'source': 'broken_glass_mask'
                    })
            
            # === SEAL DEFECT DETECTION ===
            if ENABLE_SEAL_DETECTION:
                # Add seal detection logic here
                pass
            
            # Filter: only keep defects above confidence threshold
            high_conf = [d for d in defects if d.get('confidence', 0) >= CONFIDENCE_THRESHOLD]
            
            # Build result
            result = {
                'image': mask_json,
                'folder': folder,
                'original_jpeg': jpeg_name,
                'matched_model': matched_model,
                'detection_stats': {
                    'tin_coverage': tin_coverage,
                    'glass_coverage': glass_coverage,
                    'screw_count': screw_count,
                    'hole_count': hole_count,
                    'seal_coverage': seal_coverage,
                    'tin_masks': len(tin_masks),
                    'glass_masks': len(glass_masks),
                },
                'defects': high_conf,
                'all_defects': defects,
                'ground_truth': gt
            }
            
            # Match with GT
            if gt:
                result['gt_match'] = match_with_ground_truth(high_conf, gt)
            
            results.append(result)
            
            # Print
            n = len(high_conf)
            gt_str = ""
            if gt:
                matches = result.get('gt_match', {}).get('matches', 0)
                gt_str = f" (GT: {matches}/{len(gt)} matched)"
            print(f"  {mask_json[:45]}: {n} defects{gt_str}")
    
    # Save
    output_path = os.path.join(base_dir, "combined_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {output_path}")
    
    # Summary
    print("\n" + "="*60)
    print("COMBINED PIPELINE RESULTS")
    print("="*60)
    
    total_gt = 0
    total_det = 0
    total_match = 0
    
    for r in results:
        gt_match = r.get('gt_match', {})
        total_gt += gt_match.get('ground_truth_count', 0)
        total_det += gt_match.get('detected_count', 0)
        total_match += gt_match.get('matches', 0)
    
    # Calculate score: TP=+1, FP=-1, FN=0
    tp = total_match
    fp = total_det - total_match
    fn = total_gt - total_match
    score = tp - fp
    
    print(f"Total images: {len(results)}")
    print(f"Ground truth defects: {total_gt}")
    print(f"Detected defects: {total_det}")
    print(f"Correct matches (TP): {tp}")
    print(f"False positives (FP): {fp}")
    print(f"Missed (FN): {fn}")
    print(f"\nSCORE: {score} (TP - FP = {tp} - {fp})")
    if total_gt > 0:
        print(f"RECALL: {total_match/total_gt:.1%}")
    if total_det > 0:
        print(f"PRECISION: {total_match/total_det:.1%}")
    
    return results


def match_with_ground_truth(detected: list, ground_truth: list) -> dict:
    """Match detected defects with ground truth.
    Since each image has at most ONE defect per type, we match by component+type.
    Location is secondary - if we detect tin missing/damaged anywhere, it's a match.
    """
    type_equivalent = {
        'missing': ['missing', 'damaged', 'half_tightened'],  # half_tightened screw = problem
        'damaged': ['damaged', 'missing', 'crack', 'torn', 'cracked'],
        'torn': ['torn', 'damaged', 'missing'],  # seal torn = damaged/missing
        'cracked': ['cracked', 'damaged', 'crack'],  # glass cracked
        'crack': ['crack', 'cracked', 'damaged'],
    }
    
    matches = 0
    matched_gt = []
    
    for gt in ground_truth:
        gt_comp = gt.get('component', '').lower()
        gt_type = gt.get('type', '').lower()
        
        # Get acceptable types
        acceptable_types = type_equivalent.get(gt_type, [gt_type])
        
        # Match if we have ANY defect of same component and equivalent type
        for det in detected:
            det_comp = det.get('component', '').lower()
            det_type = det.get('type', '').lower()
            
            if det_comp == gt_comp and det_type in acceptable_types:
                matches += 1
                matched_gt.append(gt)
                break
    
    return {
        'ground_truth_count': len(ground_truth),
        'detected_count': len(detected),
        'matches': matches,
        'matched_gt': matched_gt,
        'recall': matches / len(ground_truth) if ground_truth else 0,
        'precision': matches / len(detected) if detected else 0
    }


if __name__ == "__main__":
    run_combined_detection()