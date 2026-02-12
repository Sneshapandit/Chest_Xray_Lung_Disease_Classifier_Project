import os
import shutil

ROOT = os.path.abspath(os.path.dirname(__file__) + os.sep + '..')
SRC = os.path.join(ROOT, 'pre2kpro-20250116T135723Z-001-20260212T135746Z-1-001', 'pre2kpro-20250116T135723Z-001', 'pre2kpro')
DST = os.path.join(ROOT, 'data', 'raw', 'pre2kpro')

if os.path.isdir(SRC):
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    shutil.copytree(SRC, DST, dirs_exist_ok=True)
    print('Copied raw dataset to', DST)
else:
    print('Source raw dataset not found at', SRC)
