"""
Image Classifier v2 - Uses Original JPEGs (not segmented images)

Classifies facade photos to CAD models based on:
1. Aspect ratio comparison
2. Edge structure (Hough lines for panel layout)
3. Corner/keypoint density
4. Overall geometric proportions
"""

import os
import json
import cv2
import numpy as np
from typing import Dict, List, Tuple


def extract_geometric_features(image_path: str) -> Dict:
    """
    Extract geometric features from original JPEG image.
    """
    img = cv2.imread(image_path)
    if img is None:
        return {'error': f'Could not load {image_path}'}
    
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Basic dimensions
    aspect_ratio = w / h
    
    # 2. Edge detection for structure analysis
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (w * h)
    
    # 3. Hough lines for panel structure
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                            minLineLength=50, maxLineGap=10)
    
    horizontal_lines = 0
    vertical_lines = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            if angle < 15 or angle > 165:  # Horizontal
                horizontal_lines += 1
            elif 75 < angle < 105:  # Vertical
                vertical_lines += 1
    
    # 4. Corner detection (Harris)
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    corner_count = np.sum(corners > 0.01 * corners.max())
    corner_density = corner_count / (w * h) * 10000
    
    # 5. Contour analysis for panel detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    large_contours = [c for c in contours if cv2.contourArea(c) > (w * h * 0.01)]
    
    # 6. Color analysis (average colors can indicate facade type)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_hue = np.mean(hsv[:,:,0])
    mean_sat = np.mean(hsv[:,:,1])
    mean_val = np.mean(hsv[:,:,2])
    
    # 7. Texture analysis (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    return {
        'width': w,
        'height': h,
        'aspect_ratio': aspect_ratio,
        'edge_density': edge_density,
        'horizontal_lines': horizontal_lines,
        'vertical_lines': vertical_lines,
        'line_ratio': horizontal_lines / max(vertical_lines, 1),
        'corner_density': corner_density,
        'large_contours': len(large_contours),
        'mean_hue': mean_hue,
        'mean_sat': mean_sat,
        'mean_val': mean_val,
        'texture_variance': laplacian_var
    }


def load_cad_signatures(cad_models_path: str) -> Dict:
    """
    Create signatures from CAD models for matching.
    """
    with open(cad_models_path, 'r') as f:
        cad_models = json.load(f)
    
    signatures = {}
    for model_name, model_data in cad_models.items():
        # bounding_box is [min_x, min_y, min_z, max_x, max_y, max_z]
        bbox = model_data['bounding_box']
        circles = model_data['circles']
        dims = model_data.get('dimensions', {})
        
        # CAD dimensions
        dx = dims.get('width', bbox[3] - bbox[0])
        dy = dims.get('height', bbox[4] - bbox[1])
        dz = dims.get('depth', bbox[5] - bbox[2])
        
        # Aspect ratio depends on viewing direction
        # Assuming frontal view: X is width, Z is height
        aspect_xz = dx / max(dz, 1)
        aspect_xy = dx / max(dy, 1)
        aspect_yz = dy / max(dz, 1)
        
        # Component counts
        screw_holes = len([c for c in circles if c['type'] == 'screw_hole'])
        mounting_holes = len([c for c in circles if c['type'] == 'mounting_hole'])
        
        # Component density
        total_circles = len(circles)
        area_xz = dx * dz
        circle_density = total_circles / max(area_xz, 1) * 1e6  # per mm²
        
        signatures[model_name] = {
            'dimensions': {'dx': dx, 'dy': dy, 'dz': dz},
            'aspect_xz': aspect_xz,
            'aspect_xy': aspect_xy,
            'aspect_yz': aspect_yz,
            'screw_holes': screw_holes,
            'mounting_holes': mounting_holes,
            'total_circles': total_circles,
            'circle_density': circle_density,
            'screw_ratio': screw_holes / max(mounting_holes, 1)
        }
    
    return signatures


def classify_image(features: Dict, cad_signatures: Dict) -> Tuple[str, float, Dict]:
    """
    Match image features to best CAD model.
    Returns: (model_name, confidence, scores)
    """
    scores = {}
    
    img_aspect = features['aspect_ratio']
    
    for model_name, sig in cad_signatures.items():
        score = 0.0
        
        # Aspect ratio matching (try all orientations)
        aspects = [sig['aspect_xz'], sig['aspect_xy'], sig['aspect_yz'],
                   1/sig['aspect_xz'], 1/sig['aspect_xy'], 1/sig['aspect_yz']]
        
        best_aspect_match = min([abs(img_aspect - a) for a in aspects])
        aspect_score = max(0, 1 - best_aspect_match / 2)
        score += aspect_score * 0.3
        
        # Line structure matching
        # More complex facades have more lines
        expected_complexity = sig['total_circles'] / 1000  # Normalize
        actual_complexity = (features['horizontal_lines'] + features['vertical_lines']) / 100
        complexity_match = 1 - min(abs(expected_complexity - actual_complexity) / max(expected_complexity, 1), 1)
        score += complexity_match * 0.2
        
        # Corner density matching
        expected_corners = sig['total_circles'] * 0.5  # Each hole creates edges
        corner_ratio = features['corner_density'] / max(expected_corners / 100, 0.1)
        corner_score = max(0, 1 - abs(1 - corner_ratio))
        score += corner_score * 0.2
        
        # Texture variance (more holes = more texture)
        texture_score = min(features['texture_variance'] / 1000, 1)
        score += texture_score * 0.15
        
        # Edge density
        edge_score = min(features['edge_density'] * 10, 1)
        score += edge_score * 0.15
        
        scores[model_name] = {
            'total': score,
            'aspect_score': aspect_score,
            'complexity_score': complexity_match,
            'corner_score': corner_score
        }
    
    # Find best match
    best_model = max(scores, key=lambda k: scores[k]['total'])
    best_score = scores[best_model]['total']
    
    return best_model, best_score, scores


def main():
    print("="*60)
    print("Image Classifier v2 - Using Original JPEGs")
    print("="*60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    
    # Load CAD signatures
    cad_path = os.path.join(base_dir, "cad_models.json")
    cad_signatures = load_cad_signatures(cad_path)
    print(f"Loaded {len(cad_signatures)} CAD model signatures")
    
    # Process original JPEGs
    jpeg_dirs = [
        os.path.join(parent_dir, "TRAIN_JPG", "negative"),
        os.path.join(parent_dir, "TRAIN_JPG", "positive"),
    ]
    
    results = []
    model_counts = {}
    
    for jpeg_dir in jpeg_dirs:
        if not os.path.exists(jpeg_dir):
            continue
        
        category = os.path.basename(jpeg_dir)
        print(f"\n--- Processing {category} ---")
        
        for img_file in sorted(os.listdir(jpeg_dir)):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            img_path = os.path.join(jpeg_dir, img_file)
            
            # Extract features
            features = extract_geometric_features(img_path)
            if 'error' in features:
                print(f"  ERROR: {img_file}: {features['error']}")
                continue
            
            # Classify
            model, confidence, scores = classify_image(features, cad_signatures)
            
            results.append({
                'image': img_file,
                'category': category,
                'matched_model': model,
                'confidence': round(confidence, 3),
                'features': {
                    'aspect_ratio': round(features['aspect_ratio'], 3),
                    'edge_density': round(features['edge_density'], 4),
                    'h_lines': features['horizontal_lines'],
                    'v_lines': features['vertical_lines'],
                    'corner_density': round(features['corner_density'], 2)
                }
            })
            
            model_counts[model] = model_counts.get(model, 0) + 1
            
            print(f"  {img_file[:35]}: {model[:25]}... (conf: {confidence:.2f})")
    
    # Save results
    output_path = os.path.join(base_dir, "classification_results_v2.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {output_path}")
    
    # Summary
    print("\n" + "="*60)
    print("CLASSIFICATION SUMMARY")
    print("="*60)
    print(f"Total images classified: {len(results)}")
    print("\nDistribution by CAD model:")
    for model, count in sorted(model_counts.items()):
        print(f"  {model[:40]}: {count} images")


if __name__ == "__main__":
    main()
