"""
Parameter Optimization for Combined Defect Detector

This script automatically tunes the hyperparameters of combined_detector.py
to maximize detection performance (Score = TP - FP).

Uses multiple optimization strategies:
1. Grid Search - exhaustive search over parameter ranges
2. Random Search - random sampling of parameter space
3. Bayesian-like optimization - iterative refinement around best parameters

Outputs:
- Best parameter combination found
- Performance history
- Recommendation for combined_detector.py hyperparameters
"""

import os
import sys
import json
import re
import itertools
import random
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
from copy import deepcopy

# =============================================================================
# PARAMETER SEARCH SPACE
# =============================================================================

PARAM_RANGES = {
    # TIN DETECTION THRESHOLDS
    'GAP_THRESHOLD': list(range(5, 25, 2)),         # Zone coverage < this = gap
    'GOOD_COVERAGE': list(range(4, 20, 2)),         # Zone coverage >= this = has tin
    'CENTER_MIN_COVERAGE': list(range(5, 30, 5)),   # Min center coverage to trigger
    
    # TIN DETECTION CONDITIONS
    'MIN_GAPS_STRONG': [2, 3, 4],                   # Gaps needed for high confidence
    'MIN_GAPS_MEDIUM': [1, 2, 3],                   # Gaps needed for medium confidence
    'MIN_GAPS_WEAK': [1, 2],                        # Gaps needed for low confidence
    'MIN_COVERED_FOR_WEAK': [1, 2, 3],              # Covered zones needed for weak detection
    
    # CONFIDENCE THRESHOLDS
    'CONFIDENCE_THRESHOLD': [0.60, 0.65, 0.70, 0.75, 0.80],  # Min confidence to report
}

# Default values (current settings in combined_detector.py)
DEFAULT_PARAMS = {
    'GAP_THRESHOLD': 13,
    'GOOD_COVERAGE': 8,
    'CENTER_MIN_COVERAGE': 15,
    'MIN_GAPS_STRONG': 3,
    'MIN_GAPS_MEDIUM': 2,
    'MIN_GAPS_WEAK': 1,
    'MIN_COVERED_FOR_WEAK': 2,
    'CONFIDENCE_STRONG': 0.85,
    'CONFIDENCE_MEDIUM': 0.75,
    'CONFIDENCE_WEAK': 0.70,
    'CONFIDENCE_THRESHOLD': 0.70,
}


@dataclass
class EvaluationResult:
    """Stores results from one parameter evaluation."""
    params: Dict[str, Any]
    tp: int = 0
    fp: int = 0
    fn: int = 0
    score: int = 0  # TP - FP
    recall: float = 0.0
    precision: float = 0.0
    f1_score: float = 0.0
    total_images: int = 0
    
    def __str__(self):
        return f"Score={self.score} (TP={self.tp}, FP={self.fp}, FN={self.fn}) | Recall={self.recall:.1%}, Precision={self.precision:.1%}, F1={self.f1_score:.2f}"


class DetectorEvaluator:
    """
    Evaluates the combined detector with given parameters.
    Reimplements detection logic inline to avoid modifying the original file.
    """
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.parent_dir = os.path.dirname(self.base_dir)
        
        # Load required data once
        self._load_data()
    
    def _load_data(self):
        """Load all necessary data files."""
        # Load v2 classifications
        v2_class_path = os.path.join(self.base_dir, "classification_results_v2.json")
        if os.path.exists(v2_class_path):
            with open(v2_class_path) as f:
                v2_classifications = json.load(f)
            self.jpeg_to_model = {c['image']: c['matched_model'] for c in v2_classifications}
        else:
            self.jpeg_to_model = {}
            print("WARNING: classification_results_v2.json not found")
        
        # Load CAD models
        with open(os.path.join(self.base_dir, "cad_models.json")) as f:
            self.cad_models = json.load(f)
        
        # Load all mask files and ground truth
        self.mask_data = []
        seg_improved_base = os.path.join(self.parent_dir, "segmented_output_improved")
        validation_dir = os.path.join(self.parent_dir, "validation")
        
        for folder in ['negative', 'positive']:
            seg_dir = os.path.join(seg_improved_base, folder)
            if not os.path.exists(seg_dir):
                continue
            
            for mask_json in sorted(os.listdir(seg_dir)):
                if not mask_json.endswith('_masks.json'):
                    continue
                
                mask_json_path = os.path.join(seg_dir, mask_json)
                with open(mask_json_path) as f:
                    mask_content = json.load(f)
                
                # Extract names
                base_name = mask_json.replace('_masks.json', '')
                jpeg_base = base_name.replace('_facade', '')
                jpeg_name = re.sub(r'_(\d+)$', r' \1', jpeg_base) + '.jpg'
                jpeg_name_simple = jpeg_base + '.jpg'
                
                # Get matched model
                matched_model = self.jpeg_to_model.get(jpeg_name)
                if not matched_model:
                    jpeg_name_alt = jpeg_base.replace('_', ' ') + '.jpg'
                    matched_model = self.jpeg_to_model.get(jpeg_name_alt)
                if not matched_model:
                    matched_model = self.jpeg_to_model.get(jpeg_name_simple, list(self.cad_models.keys())[0])
                
                # Find ground truth
                gt = None
                json_names_to_try = [
                    re.sub(r'_(\d+)$', r' \1', jpeg_base) + '.json',
                    jpeg_base + '.json',
                    jpeg_base.replace('_', ' ') + '.json',
                ]
                
                for json_name in json_names_to_try:
                    json_path = os.path.join(validation_dir, json_name)
                    if os.path.exists(json_path):
                        with open(json_path) as f:
                            gt_data = json.load(f)
                            gt = gt_data.get('defects', [])
                        break
                
                self.mask_data.append({
                    'mask_json': mask_json,
                    'folder': folder,
                    'mask_content': mask_content,
                    'matched_model': matched_model,
                    'ground_truth': gt or []
                })
        
        print(f"Loaded {len(self.mask_data)} images for evaluation")
    
    def evaluate(self, params: Dict[str, Any]) -> EvaluationResult:
        """
        Evaluate detector with given parameters.
        Returns metrics without modifying any files.
        """
        total_gt = 0
        total_det = 0
        total_match = 0
        
        for data in self.mask_data:
            mask_content = data['mask_content']
            gt = data['ground_truth']
            
            # Get image size
            img_size = mask_content.get('size', [1536, 688])
            w, h = img_size[0], img_size[1]
            
            # Get components
            components = mask_content.get('components', {})
            tin_masks = components.get('tin', [])
            
            # === ZONE ANALYSIS (same logic as combined_detector.py) ===
            zones = ['top-left', 'top', 'top-right', 'left', 'center', 'right', 
                     'bottom-left', 'bottom', 'bottom-right']
            zone_tin = {z: 0 for z in zones}
            
            for m in tin_masks:
                bbox = m.get('bbox')
                if bbox:
                    x_min, y_min, x_max, y_max = bbox
                    cx = (x_min + x_max) / 2
                    cy = (y_min + y_max) / 2
                    
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
            
            # === TIN DEFECT DETECTION (with tunable params) ===
            defects = []
            expected_tin_zones = ['top', 'bottom', 'left', 'right']
            
            tin_gaps = [z for z in expected_tin_zones if zone_tin[z] < params['GAP_THRESHOLD']]
            well_covered = [z for z in expected_tin_zones if zone_tin[z] >= params['GOOD_COVERAGE']]
            center_coverage = zone_tin.get('center', 0)
            
            if center_coverage > params['CENTER_MIN_COVERAGE'] and len(well_covered) >= 1:
                if len(tin_gaps) >= params['MIN_GAPS_STRONG']:
                    defects.append({
                        'component': 'tin',
                        'type': 'missing',
                        'confidence': params.get('CONFIDENCE_STRONG', 0.85),
                    })
                elif len(tin_gaps) >= params['MIN_GAPS_MEDIUM'] and len(well_covered) >= 1:
                    defects.append({
                        'component': 'tin',
                        'type': 'missing',
                        'confidence': params.get('CONFIDENCE_MEDIUM', 0.75),
                    })
                elif len(tin_gaps) >= params['MIN_GAPS_WEAK'] and len(well_covered) >= params['MIN_COVERED_FOR_WEAK']:
                    defects.append({
                        'component': 'tin',
                        'type': 'missing',
                        'confidence': params.get('CONFIDENCE_WEAK', 0.70),
                    })
            
            # Filter by confidence threshold
            detected = [d for d in defects if d.get('confidence', 0) >= params['CONFIDENCE_THRESHOLD']]
            
            # === MATCH WITH GROUND TRUTH ===
            # Only count metrics for images that have ground truth
            # (same logic as combined_detector.py - negative folder images don't affect FP)
            if gt:
                matches = self._match_with_gt(detected, gt)
                
                gt_count = len(gt)
                det_count = len(detected)
                
                total_gt += gt_count
                total_det += det_count
                total_match += matches
        
        # Calculate metrics (same as combined_detector.py)
        tp = total_match
        fp = total_det - total_match
        fn = total_gt - total_match
        score = tp - fp
        recall = tp / total_gt if total_gt > 0 else 0.0
        precision = tp / total_det if total_det > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return EvaluationResult(
            params=params,
            tp=tp,
            fp=fp,
            fn=fn,
            score=score,
            recall=recall,
            precision=precision,
            f1_score=f1,
            total_images=len(self.mask_data)
        )
    
    def _match_with_gt(self, detected: list, ground_truth: list) -> int:
        """Match detected defects with ground truth."""
        type_equivalent = {
            'missing': ['missing', 'damaged', 'half_tightened'],
            'damaged': ['damaged', 'missing', 'crack', 'torn', 'cracked'],
            'torn': ['torn', 'damaged', 'missing'],
            'cracked': ['cracked', 'damaged', 'crack'],
            'crack': ['crack', 'cracked', 'damaged'],
        }
        
        matches = 0
        for gt in ground_truth:
            gt_comp = gt.get('component', '').lower()
            gt_type = gt.get('type', '').lower()
            acceptable_types = type_equivalent.get(gt_type, [gt_type])
            
            for det in detected:
                det_comp = det.get('component', '').lower()
                det_type = det.get('type', '').lower()
                
                if det_comp == gt_comp and det_type in acceptable_types:
                    matches += 1
                    break
        
        return matches


class ParameterOptimizer:
    """Optimizes detection parameters using various strategies."""
    
    def __init__(self, evaluator: DetectorEvaluator):
        self.evaluator = evaluator
        self.results_history: List[EvaluationResult] = []
        self.best_result: EvaluationResult = None
    
    def grid_search(self, param_subset: List[str] = None, max_combinations: int = 1000) -> EvaluationResult:
        """
        Exhaustive grid search over parameter space.
        
        Args:
            param_subset: List of parameter names to tune (None = all)
            max_combinations: Maximum number of combinations to try
        """
        print("\n" + "="*60)
        print("GRID SEARCH OPTIMIZATION")
        print("="*60)
        
        # Select parameters to tune
        if param_subset:
            search_params = {k: PARAM_RANGES[k] for k in param_subset if k in PARAM_RANGES}
        else:
            search_params = PARAM_RANGES
        
        # Generate all combinations
        param_names = list(search_params.keys())
        param_values = list(search_params.values())
        all_combinations = list(itertools.product(*param_values))
        
        print(f"Parameters: {param_names}")
        print(f"Total combinations: {len(all_combinations)}")
        
        # Sample if too many combinations
        if len(all_combinations) > max_combinations:
            print(f"Sampling {max_combinations} combinations...")
            all_combinations = random.sample(all_combinations, max_combinations)
        
        best_score = float('-inf')
        best_result = None
        
        for i, combo in enumerate(all_combinations):
            params = DEFAULT_PARAMS.copy()
            for name, value in zip(param_names, combo):
                params[name] = value
            
            result = self.evaluator.evaluate(params)
            self.results_history.append(result)
            
            if result.score > best_score:
                best_score = result.score
                best_result = result
                print(f"[{i+1}/{len(all_combinations)}] NEW BEST: {result}")
            
            # Progress update every 100 iterations
            if (i + 1) % 100 == 0:
                print(f"[{i+1}/{len(all_combinations)}] Current best score: {best_score}")
        
        self.best_result = best_result
        return best_result
    
    def random_search(self, n_iterations: int = 200) -> EvaluationResult:
        """
        Random search over parameter space.
        Often more efficient than grid search for high-dimensional spaces.
        """
        print("\n" + "="*60)
        print("RANDOM SEARCH OPTIMIZATION")
        print("="*60)
        print(f"Iterations: {n_iterations}")
        
        best_score = float('-inf')
        best_result = None
        
        for i in range(n_iterations):
            # Random sample from each parameter range
            params = DEFAULT_PARAMS.copy()
            for name, values in PARAM_RANGES.items():
                params[name] = random.choice(values)
            
            result = self.evaluator.evaluate(params)
            self.results_history.append(result)
            
            if result.score > best_score:
                best_score = result.score
                best_result = result
                print(f"[{i+1}/{n_iterations}] NEW BEST: {result}")
        
        if best_result and (self.best_result is None or best_result.score > self.best_result.score):
            self.best_result = best_result
        
        return best_result
    
    def local_search(self, start_params: Dict[str, Any] = None, max_iterations: int = 100) -> EvaluationResult:
        """
        Hill-climbing local search starting from given parameters.
        Explores neighboring parameter values to find local optimum.
        """
        print("\n" + "="*60)
        print("LOCAL SEARCH (HILL CLIMBING)")
        print("="*60)
        
        current_params = (start_params or DEFAULT_PARAMS).copy()
        current_result = self.evaluator.evaluate(current_params)
        self.results_history.append(current_result)
        
        print(f"Starting: {current_result}")
        
        improved = True
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # Try modifying each parameter
            for param_name in PARAM_RANGES.keys():
                current_value = current_params.get(param_name)
                if current_value is None:
                    continue
                
                values = PARAM_RANGES[param_name]
                if current_value not in values:
                    continue
                
                current_idx = values.index(current_value)
                
                # Try neighbors (previous and next values)
                for new_idx in [current_idx - 1, current_idx + 1]:
                    if 0 <= new_idx < len(values):
                        test_params = current_params.copy()
                        test_params[param_name] = values[new_idx]
                        
                        test_result = self.evaluator.evaluate(test_params)
                        self.results_history.append(test_result)
                        
                        if test_result.score > current_result.score:
                            current_params = test_params
                            current_result = test_result
                            improved = True
                            print(f"[{iteration}] Improved {param_name}: {test_result}")
                            break
                
                if improved:
                    break
        
        if self.best_result is None or current_result.score > self.best_result.score:
            self.best_result = current_result
        
        return current_result
    
    def bayesian_like_search(self, n_initial: int = 50, n_iterations: int = 100) -> EvaluationResult:
        """
        Pseudo-Bayesian optimization:
        1. Random exploration phase
        2. Exploitation phase focusing on promising regions
        """
        print("\n" + "="*60)
        print("BAYESIAN-LIKE OPTIMIZATION")
        print("="*60)
        
        # Phase 1: Random exploration
        print(f"\nPhase 1: Exploration ({n_initial} random samples)...")
        exploration_results = []
        
        for i in range(n_initial):
            params = DEFAULT_PARAMS.copy()
            for name, values in PARAM_RANGES.items():
                params[name] = random.choice(values)
            
            result = self.evaluator.evaluate(params)
            self.results_history.append(result)
            exploration_results.append(result)
            
            if i % 20 == 0:
                print(f"  [{i}/{n_initial}] Best so far: {max(r.score for r in exploration_results)}")
        
        # Phase 2: Exploitation around best parameters
        print(f"\nPhase 2: Exploitation ({n_iterations} focused samples)...")
        
        # Get top 10% of results
        sorted_results = sorted(exploration_results, key=lambda r: r.score, reverse=True)
        top_results = sorted_results[:max(1, len(sorted_results) // 10)]
        
        for i in range(n_iterations):
            # Sample from neighborhood of good solutions
            base_result = random.choice(top_results)
            params = base_result.params.copy()
            
            # Mutate 1-3 random parameters
            n_mutations = random.randint(1, 3)
            mutation_params = random.sample(list(PARAM_RANGES.keys()), min(n_mutations, len(PARAM_RANGES)))
            
            for name in mutation_params:
                values = PARAM_RANGES[name]
                current = params.get(name)
                if current in values:
                    idx = values.index(current)
                    # Small step: prefer neighboring values
                    new_idx = idx + random.choice([-1, 0, 1])
                    new_idx = max(0, min(len(values) - 1, new_idx))
                    params[name] = values[new_idx]
                else:
                    params[name] = random.choice(values)
            
            result = self.evaluator.evaluate(params)
            self.results_history.append(result)
            
            # Update top results if this is better
            if result.score > top_results[-1].score:
                top_results.append(result)
                top_results = sorted(top_results, key=lambda r: r.score, reverse=True)[:max(1, n_initial // 10)]
                print(f"  [{i}/{n_iterations}] NEW TOP: {result}")
        
        best_result = max(self.results_history, key=lambda r: r.score)
        if self.best_result is None or best_result.score > self.best_result.score:
            self.best_result = best_result
        
        return best_result
    
    def run_full_optimization(self) -> EvaluationResult:
        """Run all optimization strategies and return best overall result."""
        print("\n" + "="*70)
        print("FULL OPTIMIZATION PIPELINE")
        print("="*70)
        
        # 1. Evaluate default parameters first
        print("\n>>> Evaluating DEFAULT parameters...")
        default_result = self.evaluator.evaluate(DEFAULT_PARAMS)
        self.results_history.append(default_result)
        print(f"Default: {default_result}")
        
        # 2. Random search for broad exploration
        random_best = self.random_search(n_iterations=200)
        print(f"\nRandom Search Best: {random_best}")
        
        # 3. Local search from best random result
        local_best = self.local_search(start_params=random_best.params, max_iterations=50)
        print(f"\nLocal Search Best: {local_best}")
        
        # 4. Grid search on most impactful parameters
        key_params = ['GAP_THRESHOLD', 'GOOD_COVERAGE', 'CENTER_MIN_COVERAGE', 'CONFIDENCE_THRESHOLD']
        grid_best = self.grid_search(param_subset=key_params, max_combinations=500)
        print(f"\nGrid Search Best: {grid_best}")
        
        # 5. Bayesian-like search for fine-tuning
        bayesian_best = self.bayesian_like_search(n_initial=50, n_iterations=100)
        print(f"\nBayesian Search Best: {bayesian_best}")
        
        # Final best
        self.best_result = max(self.results_history, key=lambda r: r.score)
        
        return self.best_result


def generate_report(optimizer: ParameterOptimizer, output_path: str):
    """Generate a detailed optimization report."""
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_evaluations': len(optimizer.results_history),
        'best_result': {
            'params': optimizer.best_result.params,
            'score': optimizer.best_result.score,
            'tp': optimizer.best_result.tp,
            'fp': optimizer.best_result.fp,
            'fn': optimizer.best_result.fn,
            'recall': optimizer.best_result.recall,
            'precision': optimizer.best_result.precision,
            'f1_score': optimizer.best_result.f1_score,
        },
        'default_params': DEFAULT_PARAMS,
        'improvement': {
            'default_score': None,
            'best_score': optimizer.best_result.score,
            'improvement': None,
        },
        'top_10_configurations': [],
        'parameter_recommendations': {},
    }
    
    # Find default result
    for result in optimizer.results_history:
        if result.params == DEFAULT_PARAMS:
            report['improvement']['default_score'] = result.score
            report['improvement']['improvement'] = optimizer.best_result.score - result.score
            break
    
    # Top 10 configurations
    sorted_results = sorted(optimizer.results_history, key=lambda r: r.score, reverse=True)
    unique_results = []
    seen_params = set()
    for r in sorted_results:
        param_tuple = tuple(sorted(r.params.items()))
        if param_tuple not in seen_params:
            seen_params.add(param_tuple)
            unique_results.append(r)
            if len(unique_results) >= 10:
                break
    
    for r in unique_results:
        report['top_10_configurations'].append({
            'params': r.params,
            'score': r.score,
            'recall': r.recall,
            'precision': r.precision,
        })
    
    # Parameter analysis - what values appear most in top results
    top_n = min(20, len(unique_results))
    param_counts = {name: {} for name in PARAM_RANGES.keys()}
    for r in unique_results[:top_n]:
        for name in PARAM_RANGES.keys():
            value = r.params.get(name)
            if value is not None:
                param_counts[name][value] = param_counts[name].get(value, 0) + 1
    
    for name, counts in param_counts.items():
        if counts:
            best_value = max(counts.items(), key=lambda x: x[1])[0]
            report['parameter_recommendations'][name] = {
                'recommended': best_value,
                'frequency_in_top_20': counts,
            }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report


def print_recommendations(report: dict):
    """Print parameter recommendations in a format ready to paste into combined_detector.py."""
    
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    
    best = report['best_result']
    print(f"\nBest Score: {best['score']} (TP={best['tp']}, FP={best['fp']}, FN={best['fn']})")
    print(f"Recall: {best['recall']:.1%} | Precision: {best['precision']:.1%} | F1: {best['f1_score']:.2f}")
    
    if report['improvement']['default_score'] is not None:
        imp = report['improvement']
        print(f"\nImprovement over default: {imp['default_score']} -> {imp['best_score']} (+{imp['improvement']})")
    
    print("\n" + "-"*70)
    print("RECOMMENDED HYPERPARAMETERS (copy to combined_detector.py):")
    print("-"*70)
    
    params = best['params']
    print(f"""
# =============================================================================
# OPTIMIZED HYPERPARAMETERS (Generated {datetime.now().strftime('%Y-%m-%d %H:%M')})
# =============================================================================

# TIN DETECTION THRESHOLDS
GAP_THRESHOLD = {params.get('GAP_THRESHOLD', 13)}          # Zone coverage < this = potential gap
GOOD_COVERAGE = {params.get('GOOD_COVERAGE', 8)}          # Zone coverage >= this = has tin
CENTER_MIN_COVERAGE = {params.get('CENTER_MIN_COVERAGE', 15)}   # Minimum center zone coverage

# TIN DETECTION CONDITIONS
MIN_GAPS_STRONG = {params.get('MIN_GAPS_STRONG', 3)}        # Gaps for high confidence
MIN_GAPS_MEDIUM = {params.get('MIN_GAPS_MEDIUM', 2)}        # Gaps for medium confidence
MIN_GAPS_WEAK = {params.get('MIN_GAPS_WEAK', 1)}          # Gaps for low confidence
MIN_COVERED_FOR_WEAK = {params.get('MIN_COVERED_FOR_WEAK', 2)}   # Covered zones for weak detection

# CONFIDENCE SCORES
CONFIDENCE_STRONG = {params.get('CONFIDENCE_STRONG', 0.85)}   # Confidence for 3+ gaps
CONFIDENCE_MEDIUM = {params.get('CONFIDENCE_MEDIUM', 0.75)}   # Confidence for 2 gaps
CONFIDENCE_WEAK = {params.get('CONFIDENCE_WEAK', 0.70)}     # Confidence for 1 gap
CONFIDENCE_THRESHOLD = {params.get('CONFIDENCE_THRESHOLD', 0.70)}  # Min confidence to report
""")


def main():
    """Main optimization entry point."""
    print("="*70)
    print("COMBINED DETECTOR PARAMETER OPTIMIZATION")
    print("="*70)
    print(f"Started: {datetime.now()}")
    
    # Initialize evaluator
    print("\nInitializing evaluator...")
    evaluator = DetectorEvaluator()
    
    # Initialize optimizer
    optimizer = ParameterOptimizer(evaluator)
    
    # Run full optimization
    best_result = optimizer.run_full_optimization()
    
    # Generate report
    base_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(base_dir, "optimization_report.json")
    report = generate_report(optimizer, report_path)
    
    print(f"\nReport saved to: {report_path}")
    
    # Print recommendations
    print_recommendations(report)
    
    print("\n" + "="*70)
    print(f"OPTIMIZATION COMPLETE - {len(optimizer.results_history)} configurations tested")
    print(f"Finished: {datetime.now()}")
    print("="*70)
    
    return best_result


if __name__ == "__main__":
    main()
