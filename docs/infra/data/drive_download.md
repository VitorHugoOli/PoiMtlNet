# docs/infra/data/drive_download.md — Fetching data from Google Drive

This is the canonical recipe for bootstrapping data on a fresh machine that doesn't have a local data clone.

## When to use

- Fresh RunPod / Lightning / H100 pod that needs the project's input data.
- Any environment where you can't `git clone` data (we deliberately gitignore the input parquets — they're 2-3 GB per state).
- Replacing a state's data after a re-generation pass.

## Drive layout (canonical)

```
<DRIVE_ROOT>/PoiMtlNet/
├── data/
│   ├── checkins/
│   │   ├── Alabama.parquet
│   │   ├── Arizona.parquet
│   │   ├── California.parquet
│   │   ├── Florida.parquet
│   │   ├── Georgia.parquet
│   │   └── Texas.parquet
│   └── miscellaneous/
│       ├── tl_2022_01_tract_AL/    # TIGER census tracts per state
│       ├── tl_2022_04_tract_AZ/
│       ├── tl_2022_06_tract_CA/
│       ├── tl_2022_12_tract_FL/
│       ├── tl_2022_13_tract_GA/
│       └── tl_2022_48_tract_TX/
├── output/
│   ├── check2hgi/<state>/
│   │   ├── check_embeddings.parquet
│   │   ├── region_embeddings.parquet
│   │   └── region_transition.npz
│   ├── hgi/<state>/
│   │   └── poi_embeddings.csv
│   └── ...                          # other engines (dgi, time2vec, etc.)
└── results/                          # optional, for syncing finished runs back
```

## Download script

`scripts/phase3_download_drive.py` is the canonical download utility. Reads a state name and pulls the corresponding `data/checkins/<State>.parquet` + `data/miscellaneous/tl_2022_*_tract_*/` + relevant `output/<engine>/<state>/*` from Drive via gdown.

```bash
python scripts/phase3_download_drive.py --state florida
```

For multi-state bootstrap:
```bash
for s in alabama arizona florida california texas; do
  python scripts/phase3_download_drive.py --state $s
done
```

## Direct gdown patterns (when the script doesn't fit)

```bash
pip install gdown

# Whole folder by ID
gdown --folder https://drive.google.com/drive/folders/<FOLDER_ID> -O .

# Single file
gdown <FILE_ID> -O data/checkins/Florida.parquet
```

Get folder IDs from the Drive UI: right-click → Share → Copy link → take the long ID after `folders/` or `file/d/`.

## Drive credentials

`gdown` uses anonymous access for public folders. For shared-but-not-public folders, set up `~/.config/gdown/cookies.txt`:

```bash
# On a browser-equipped machine, log into Drive and export cookies via a browser extension
# Then on the pod:
mkdir -p ~/.config/gdown
scp local-cookies.txt user@pod:~/.config/gdown/cookies.txt
```

Or use the `--id` flow with a service account JSON if you've set one up.

## Sanity check

After download:

```bash
ls -la data/checkins/Florida.parquet                    # ~2.3 GB
ls data/miscellaneous/tl_2022_12_tract_FL/              # multiple shapefile parts
ls -la output/check2hgi/florida/check_embeddings.parquet # ~1-2 GB depending on epoch count
```

If anything is missing or the parquet won't open with `pandas.read_parquet`, re-download — the gdown stream can truncate on flaky networks.
