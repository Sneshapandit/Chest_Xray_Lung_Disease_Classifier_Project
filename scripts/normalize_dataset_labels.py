import os
import csv

ROOT = os.path.abspath(os.path.dirname(__file__) + os.sep + '..')
RAW_ROOT = os.path.join(ROOT, 'data', 'raw')
INPUT = os.path.join(ROOT, 'dataset_labels.csv')
OUTPUT = os.path.join(ROOT, 'dataset_labels_normalized.csv')

# build a map of basename -> relative path under data/raw
mapping = {}
for dirpath, dirnames, filenames in os.walk(RAW_ROOT):
    for fn in filenames:
        mapping.setdefault(fn, []).append(os.path.relpath(os.path.join(dirpath, fn), ROOT))

print(f'Found {sum(len(v) for v in mapping.values())} files under {RAW_ROOT}')

if not os.path.exists(INPUT):
    print('Input file not found:', INPUT)
else:
    with open(INPUT, newline='', encoding='utf-8') as inf, open(OUTPUT, 'w', newline='', encoding='utf-8') as outf:
        reader = csv.reader(inf)
        writer = csv.writer(outf)
        changed = 0
        total = 0
        for row in reader:
            total += 1
            if not row:
                writer.writerow(row)
                continue
            path = row[0]
            # if path is absolute or contains backslashes, try normalize
            base = os.path.basename(path)
            if base in mapping and mapping[base]:
                # if multiple matches, pick the first
                new_rel = mapping[base][0]
                new_path = new_rel.replace('\\', '/')
                row[0] = new_path
                changed += 1
            writer.writerow(row)
    print(f'Wrote {OUTPUT}: {changed}/{total} paths normalized')
