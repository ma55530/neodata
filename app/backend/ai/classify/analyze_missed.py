import json

# Load results
results = json.load(open(r'c:\Users\Jura Slibar\Desktop\NeoData\neodata2\combined_results.json'))

# Separate positive (no GT) from negative (has GT)
positive = [r for r in results if r.get('folder') == 'positive']
negative = [r for r in results if r.get('folder') == 'negative']

print("POSITIVE images (no defects expected):")
for r in positive:
    stats = r.get('detection_stats', {})
    detected = len(r.get('defects', []))
    print(f"  {r['image'][:30]}: tin={stats.get('tin_coverage',0):.0f}%, detected={detected}")
    
print()
print("NEGATIVE images (have defects in GT):")
for r in negative:
    stats = r.get('detection_stats', {})
    gt = r.get('ground_truth', [])
    detected = len(r.get('defects', []))
    gt_str = ', '.join([f"{g.get('component')}" for g in (gt or [])])
    print(f"  {r['image'][:30]}: tin={stats.get('tin_coverage',0):.0f}%, GT={gt_str}, det={detected}")
