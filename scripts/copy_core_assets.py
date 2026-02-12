import os
import shutil

ROOT = os.path.abspath(os.path.dirname(__file__) + os.sep + '..')
SRC_BASE1 = os.path.join(ROOT, 'processed_data-20260212T135828Z-1-001', 'processed_data')
SRC_BASE2 = os.path.join(ROOT, 'processed_data-20260212T135828Z-1-002', 'processed_data')
SRC_BASE4 = os.path.join(ROOT, 'processed_data-20260212T135828Z-1-004', 'processed_data')
DST = os.path.join(ROOT, 'data', 'processed')

os.makedirs(DST, exist_ok=True)

candidates = [
    (os.path.join(SRC_BASE1, 'classifier_model.keras'), os.path.join(DST, 'classifier_model.keras')),
    (os.path.join(SRC_BASE1, 'pca_model.pkl'), os.path.join(DST, 'pca_model.pkl')),
    (os.path.join(SRC_BASE1, 'label_to_index.npy'), os.path.join(DST, 'label_to_index.npy')),
    (os.path.join(SRC_BASE1, 'labels.npy'), os.path.join(DST, 'labels.npy')),
    (os.path.join(SRC_BASE1, 'train_test_split', 'X_test.npy'), os.path.join(DST, 'train_test_split', 'X_test.npy')),
    (os.path.join(SRC_BASE1, 'train_test_split', 'y_test.npy'), os.path.join(DST, 'train_test_split', 'y_test.npy')),
    (os.path.join(SRC_BASE2, 'resnet_feature_extractor.keras'), os.path.join(DST, 'resnet_feature_extractor.keras')),
    (os.path.join(SRC_BASE4, 'images.npy'), os.path.join(DST, 'images.npy')),
]

# copy train_test_split dir
TRAIN_DST = os.path.join(DST, 'train_test_split')
os.makedirs(TRAIN_DST, exist_ok=True)

# copy densenet folder if present
SRC_DENSENET = os.path.join(SRC_BASE1, 'densenet')
DST_DENSENET = os.path.join(DST, 'densenet')
if os.path.isdir(SRC_DENSENET):
    shutil.copytree(SRC_DENSENET, DST_DENSENET, dirs_exist_ok=True)
    print('Copied densenet folder to', DST_DENSENET)
else:
    print('No densenet folder found at', SRC_DENSENET)

for src, dst in candidates:
    if os.path.exists(src):
        dst_dir = os.path.dirname(dst)
        os.makedirs(dst_dir, exist_ok=True)
        try:
            shutil.copy2(src, dst)
            print('Copied', src, '->', dst)
        except Exception as e:
            print('Failed to copy', src, '->', dst, e)
    else:
        print('Source not found, skipping:', src)

print('Done copying core assets.')
