import json
results = json.load(open(r'c:\Users\Jura Slibar\Desktop\NeoData\neodata2\combined_results.json'))

print("FP images (positive folder with detected defects):")
for r in results:
    if r.get('folder') == 'positive' and len(r.get('defects',[])) > 0:
        d = r['defects'][0]
        zc = d.get('zone_coverage', {})
        print(f"\n{r['image'][:30]}:")
        print(f"  Center={zc.get('center',0):.1f}, Top={zc.get('top',0):.1f}, Bottom={zc.get('bottom',0):.1f}, Left={zc.get('left',0):.1f}, Right={zc.get('right',0):.1f}")
        print(f"  Gaps: {d.get('zones_missing')}, Covered: {d.get('zones_present')}")
