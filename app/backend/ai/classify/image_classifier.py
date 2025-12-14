"""
Image to CAD Model Classifier

Automatically classifies facade photos to the correct CAD model
based on features extracted from SAM segmentation.
"""

import os
import json
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ImageFeatures:
    """Features extracted from an image for classification."""
    filename: str
    width: int
    height: int
    aspect_ratio: float
    screw_count: int = 0
    hole_count: int = 0
    component_positions: List[Tuple[float, float]] = None  # Normalized positions
    
    def to_dict(self):
        return {
            'filename': self.filename,
            'resolution': f"{self.width}x{self.height}",
            'aspect_ratio': round(self.aspect_ratio, 3),
            'screw_count': self.screw_count,
            'hole_count': self.hole_count,
        }


class ImageClassifier:
    """Classifies images to CAD models."""
    
    def __init__(self, cad_models_path: str):
        """Load CAD model signatures for matching."""
        with open(cad_models_path, 'r') as f:
            self.cad_models = json.load(f)
        
        print(f"Loaded {len(self.cad_models)} CAD models:")
        for name, model in self.cad_models.items():
            sig = model['signature']
            print(f"  {name}: {sig['screw_holes']} screws, {sig['mounting_holes']} holes, "
                  f"aspect={sig['aspect_ratio']}")
    
    def extract_features_from_segmented(self, image_path: str) -> ImageFeatures:
        """Extract classification features from a SAM-segmented image."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        h, w = img.shape[:2]
        features = ImageFeatures(
            filename=os.path.basename(image_path),
            width=w,
            height=h,
            aspect_ratio=w / h
        )
        
        # Convert to HSV for color-based detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Detect BLUE regions (screws in SAM output)
        # Blue in HSV: H=100-140
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([140, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Detect YELLOW/CYAN regions (holes in SAM output)
        lower_yellow = np.array([20, 50, 50])
        upper_yellow = np.array([40, 255, 255])
        lower_cyan = np.array([80, 50, 50])
        upper_cyan = np.array([100, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        cyan_mask = cv2.inRange(hsv, lower_cyan, upper_cyan)
        hole_mask = yellow_mask | cyan_mask
        
        # Count connected components (screws)
        screw_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        features.screw_count = len([c for c in screw_contours if cv2.contourArea(c) > 20])
        
        # Count connected components (holes)
        hole_contours, _ = cv2.findContours(hole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        features.hole_count = len([c for c in hole_contours if cv2.contourArea(c) > 20])
        
        # Get component centroids (normalized)
        positions = []
        for cnt in screw_contours:
            if cv2.contourArea(cnt) > 20:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"] / w
                    cy = M["m01"] / M["m00"] / h
                    positions.append((cx, cy))
        features.component_positions = positions
        
        return features
    
    def extract_features_opencv(self, image_path: str) -> ImageFeatures:
        """Extract features using pure OpenCV (for non-segmented images)."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        h, w = img.shape[:2]
        features = ImageFeatures(
            filename=os.path.basename(image_path),
            width=w,
            height=h,
            aspect_ratio=w / h
        )
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Blob detection for screws
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 30
        params.maxArea = 2000
        params.filterByCircularity = True
        params.minCircularity = 0.4
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray)
        features.screw_count = len(keypoints)
        
        # Hough circles for holes
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.2, 20,
                                   param1=50, param2=30, minRadius=5, maxRadius=50)
        features.hole_count = len(circles[0]) if circles is not None else 0
        
        return features
    
    def classify(self, features: ImageFeatures) -> Tuple[str, float]:
        """
        Classify image to best matching CAD model.
        Returns (model_name, confidence).
        """
        best_match = None
        best_score = -1
        
        for model_name, model_data in self.cad_models.items():
            sig = model_data['signature']
            
            # Compute similarity score
            # 1. Aspect ratio similarity (CAD width/height vs image width/height)
            cad_aspect = sig['aspect_ratio']
            img_aspect = features.aspect_ratio
            aspect_sim = 1.0 - min(abs(cad_aspect - img_aspect) / max(cad_aspect, img_aspect), 1.0)
            
            # 2. Component count ratio similarity
            # Normalize by total circles in CAD
            cad_total = sig['total_circles']
            img_total = features.screw_count + features.hole_count
            
            # Scale factor: how many image detections per CAD circle (varies with distance/resolution)
            # We don't know exact scale, so compare ratios
            cad_screw_ratio = sig['screw_holes'] / cad_total if cad_total > 0 else 0
            img_screw_ratio = features.screw_count / img_total if img_total > 0 else 0
            ratio_sim = 1.0 - min(abs(cad_screw_ratio - img_screw_ratio), 1.0)
            
            # Combined score
            score = aspect_sim * 0.4 + ratio_sim * 0.6
            
            if score > best_score:
                best_score = score
                best_match = model_name
        
        return best_match, best_score
    
    def classify_image(self, image_path: str, use_segmented: bool = True) -> Dict:
        """Full classification pipeline for one image."""
        try:
            if use_segmented:
                features = self.extract_features_from_segmented(image_path)
            else:
                features = self.extract_features_opencv(image_path)
            
            model_name, confidence = self.classify(features)
            
            return {
                'image': os.path.basename(image_path),
                'matched_model': model_name,
                'confidence': round(confidence, 3),
                'features': features.to_dict()
            }
        except Exception as e:
            return {
                'image': os.path.basename(image_path),
                'error': str(e)
            }


def main():
    print("="*60)
    print("Image to CAD Model Classifier")
    print("="*60)
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    cad_models_path = os.path.join(base_dir, "cad_models.json")
    
    # Initialize classifier
    classifier = ImageClassifier(cad_models_path)
    
    # Test on segmented images
    seg_positive_dir = os.path.join(parent_dir, "segmented_positive")
    seg_negative_dir = os.path.join(parent_dir, "segmented_negative")
    
    results = []
    
    # Process segmented positive images
    if os.path.exists(seg_positive_dir):
        print(f"\n--- Processing segmented_positive ---")
        for img_file in os.listdir(seg_positive_dir):
            if img_file.lower().endswith(('.jpg', '.png')):
                img_path = os.path.join(seg_positive_dir, img_file)
                result = classifier.classify_image(img_path, use_segmented=True)
                results.append(result)
                print(f"  {result['image']}: {result.get('matched_model', 'ERROR')[:30]}... "
                      f"(conf={result.get('confidence', 0):.2f})")
    
    # Process segmented negative images
    if os.path.exists(seg_negative_dir):
        print(f"\n--- Processing segmented_negative ---")
        for img_file in os.listdir(seg_negative_dir):
            if img_file.lower().endswith(('.jpg', '.png')):
                img_path = os.path.join(seg_negative_dir, img_file)
                result = classifier.classify_image(img_path, use_segmented=True)
                results.append(result)
                print(f"  {result['image']}: {result.get('matched_model', 'ERROR')[:30]}... "
                      f"(conf={result.get('confidence', 0):.2f})")
    
    # Save results
    output_path = os.path.join(base_dir, "classification_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved classification results to: {output_path}")
    
    # Summary
    print("\n" + "="*60)
    print("CLASSIFICATION SUMMARY")
    print("="*60)
    model_counts = {}
    for r in results:
        model = r.get('matched_model', 'ERROR')
        model_counts[model] = model_counts.get(model, 0) + 1
    
    for model, count in sorted(model_counts.items()):
        print(f"  {model}: {count} images")


if __name__ == "__main__":
    main()
