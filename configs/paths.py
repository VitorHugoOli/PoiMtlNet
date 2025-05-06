import os

DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'))
RESULTS_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results'))


IO_CHECKINS = os.path.join(DATA_ROOT, 'checkins')
if not os.path.exists(IO_CHECKINS):
    raise FileNotFoundError(f"Checkins directory not found: {IO_CHECKINS}")

OUTPUT_ROOT = os.path.join(DATA_ROOT, 'output')
if not os.path.exists(OUTPUT_ROOT):
    os.makedirs(OUTPUT_ROOT)

if not os.path.exists(RESULTS_ROOT):
    os.makedirs(RESULTS_ROOT)