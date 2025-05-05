import os

IO_ROOT = '/Users/vitor/Desktop/mestrado/ingred/data'

IO_CHECKINS = os.path.join(IO_ROOT, 'checkins')
if not os.path.exists(IO_CHECKINS):
    pass

OUTPUT_ROOT = os.path.join(IO_ROOT, 'output')
if not os.path.exists(OUTPUT_ROOT):
    os.makedirs(OUTPUT_ROOT)

RESULTS_PATH = os.path.join(IO_ROOT, 'results')
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)
