import json

results = json.load(open(r'c:\Users\Jura Slibar\Desktop\NeoData\neodata2\combined_results.json'))

print("Tin coverage analysis:")
print("-" * 60)

# Separate by folder and GT
negative_with_gt = []
positive = []

for r in results:
    stats = r['detection_stats']
    tin = stats['tin_coverage']
    masks = stats['tin_masks']
    gt = r.get('ground_truth', [])
    
    if r['folder'] == 'positive':
        positive.append((r['image'][:25], tin, masks, 'NO_DEFECT'))
    elif gt:
        gt_types = [f"{g['component']}-{g['type']}" for g in gt]
        negative_with_gt.append((r['image'][:25], tin, masks, gt_types))

print("\nNEGATIVE (has defects) - tin coverage:")
for name, tin, masks, gt in sorted(negative_with_gt, key=lambda x: x[1]):
    print(f"  {name}: {tin:.1f}% ({masks} masks) GT: {gt}")

print("\nPOSITIVE (no defects) - tin coverage:")
for name, tin, masks, gt in sorted(positive, key=lambda x: x[1]):
    print(f"  {name}: {tin:.1f}% ({masks} masks)")
