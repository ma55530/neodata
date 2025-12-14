"""Quick analysis of detection results."""
import json

with open('defect_results.json') as f:
    data = json.load(f)

print("Images with 0 detections but have GT defects:")
print("-" * 60)
for r in data:
    gt = r.get('ground_truth', [])
    if gt and len(r['defects']) == 0:
        tin = r['detection_stats']['tin_coverage']
        print(f"Image: {r['image'][:30]}")
        print(f"  Tin coverage: {tin:.0%}")
        print(f"  GT: {gt}")
        print()

print("\nImages that matched GT:")
print("-" * 60)
for r in data:
    gt_match = r.get('gt_match', {})
    if gt_match.get('matches', 0) > 0:
        print(f"Image: {r['image'][:30]}")
        print(f"  Detected: {r['defects']}")
        print(f"  GT: {r.get('ground_truth', [])}")
        print()
