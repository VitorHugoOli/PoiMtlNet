#!/usr/bin/env python3
"""Per-file comment census for src/*.tex and src/chapters/*.tex.

A COMMENT LINE is a line whose first non-blank character is '%'.  A BLOCK is a maximal run of
consecutive comment lines (blank lines break a block).  Prints the table and, with --blocks, every
block of >= N lines with its first line, so a human can see what the long ones say.
"""
import glob, json, re, sys

FILES = sorted(glob.glob("src/*.tex")) + sorted(glob.glob("src/chapters/*.tex"))
MIN = int(sys.argv[1]) if len(sys.argv) > 1 else 8

rows, blocks = [], []
for f in FILES:
    lines = open(f).read().split("\n")
    if lines and lines[-1] == "": lines.pop()
    iscom = [bool(re.match(r"[ \t]*%", L)) for L in lines]
    n_com = sum(iscom)
    # blocks
    i, fileblocks = 0, []
    while i < len(lines):
        if iscom[i]:
            j = i
            while j < len(lines) and iscom[j]: j += 1
            fileblocks.append((i + 1, j - i, lines[i].strip()))
            i = j
        else:
            i += 1
    signoff = sum(L.count("[NEEDS SIGN-OFF") for L in lines)
    rows.append({"file": f, "total": len(lines), "comment": n_com,
                 "pct": round(100 * n_com / max(len(lines), 1), 1),
                 "blocks": len(fileblocks),
                 "max_block": max([b[1] for b in fileblocks], default=0),
                 "blocks_ge_min": sum(1 for b in fileblocks if b[1] >= MIN),
                 "lines_in_blocks_ge_min": sum(b[1] for b in fileblocks if b[1] >= MIN),
                 "signoff": signoff})
    for st, ln, first in fileblocks:
        if ln >= MIN: blocks.append({"file": f, "line": st, "len": ln, "first": first[:100]})

print(f"{'file':34s} {'tot':>5s} {'com':>5s} {'pct':>6s} {'blks':>5s} {'max':>4s} {'>=%d'%MIN:>5s} {'lines':>6s} {'signoff':>7s}")
for r in rows:
    print(f"{r['file']:34s} {r['total']:5d} {r['comment']:5d} {r['pct']:6.1f} {r['blocks']:5d} {r['max_block']:4d} {r['blocks_ge_min']:5d} {r['lines_in_blocks_ge_min']:6d} {r['signoff']:7d}")
T = lambda k: sum(r[k] for r in rows)
print(f"{'TOTAL':34s} {T('total'):5d} {T('comment'):5d} {100*T('comment')/T('total'):6.1f} {T('blocks'):5d} {'':4s} {T('blocks_ge_min'):5d} {T('lines_in_blocks_ge_min'):6d} {T('signoff'):7d}")
json.dump({"rows": rows, "blocks": blocks}, open(sys.argv[2] if len(sys.argv) > 2 else "src_utils/_round7/_trimwork/census.json", "w"), indent=1)
print(f"\n{len(blocks)} blocks of >= {MIN} comment lines")
