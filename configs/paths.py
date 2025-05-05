import os

DATA_ROOT = os.path.join('./', 'data')

IO_CHECKINS = os.path.join(DATA_ROOT, 'checkins')
if not os.path.exists(IO_CHECKINS):
    raise FileNotFoundError(f"Checkins directory not found: {IO_CHECKINS}")

OUTPUT_ROOT = os.path.join(DATA_ROOT, 'output')
if not os.path.exists(OUTPUT_ROOT):
    os.makedirs(OUTPUT_ROOT)

RESULTS_ROOT = os.path.join('./', 'results')
if not os.path.exists(RESULTS_ROOT):
    os.makedirs(RESULTS_ROOT)