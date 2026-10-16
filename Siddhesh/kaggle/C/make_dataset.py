"""Package the inputs of notebook C (PCPA windows + ViTPose pose/crop cache, ~100 MB) as a Kaggle dataset folder.

    python kaggle/C/make_dataset.py            # from Siddhesh/ ; writes kaggle/pg_data/
    kaggle datasets create -p kaggle/pg_data   # first time (private); later: kaggle datasets version -p kaggle/pg_data -m "msg"
"""
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, 'kaggle', 'pg_data')
os.makedirs(OUT, exist_ok=True)
shutil.copy(os.path.join(ROOT, 'data', 'samples_ctx', 'jaad_beh.pkl'), os.path.join(OUT, 'jaad_beh.pkl'))
dst = os.path.join(OUT, 'pose_beh')
if not os.path.exists(dst):
    shutil.copytree(os.path.join(ROOT, 'data', 'pose_beh'), dst)
json.dump({'title': 'JAAD PedGraph data', 'id': 'siddheshnadkarni/jaad-pedgraph-data',
           'licenses': [{'name': 'other'}]}, open(os.path.join(OUT, 'dataset-metadata.json'), 'w'), indent=1)
print('wrote', OUT, '->  kaggle datasets create -p', OUT)
