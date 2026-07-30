#!/usr/bin/env python3
"""Both-direction self-test for fingerprint.py (AGENT_GUARDRAILS 4b V3 / BRIEF 'validate any new
checker in BOTH directions').  Builds two 2-page PDFs that differ by ONE WORD on page 2, plus a
byte-identical copy, and asserts the comparator's verdict on each pair.
"""
import json, os, shutil, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fingerprint as FP

DOC = r"""\documentclass{article}\usepackage[a4paper]{geometry}\begin{document}
Page one text, unchanged in both variants of this self-test.
\newpage
Page two says the word %WORD%, which is the only difference between the two builds.
\end{document}
"""

def latex_path():
    # same source the Makefile reads its PATH from
    out = subprocess.run(["bash", "-c",
        ". ../../texenv.sh >/dev/null 2>&1 && echo $PATH"],
        capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
    return out.stdout.strip()

def build(word, tmp, path):
    src = os.path.join(tmp, f"t_{word}.tex")
    open(src, "w").write(DOC.replace("%WORD%", word))
    env = dict(os.environ, PATH=path)
    r = subprocess.run(["pdflatex", "-interaction=batchmode", "-halt-on-error",
                        os.path.basename(src)], cwd=tmp, env=env, capture_output=True)
    pdf = os.path.join(tmp, f"t_{word}.pdf")
    assert r.returncode == 0 and os.path.exists(pdf), f"build failed for {word}: rc={r.returncode}"
    return pdf

def main():
    path = latex_path()
    assert path, "could not resolve a TeX PATH"
    fails = []
    with tempfile.TemporaryDirectory() as tmp:
        a_pdf = build("ALPHA", tmp, path)
        b_pdf = build("BRAVO", tmp, path)
        c_pdf = os.path.join(tmp, "copy.pdf"); shutil.copyfile(a_pdf, c_pdf)
        fa, fb, fc = (FP.fingerprint(p) for p in (a_pdf, b_pdf, c_pdf))
        for f in (fa, fb, fc): f.pop("_text")

        # DIRECTION 1: defect present -> must report DIFFER, and name page 2 only
        v, d = FP.compare(fa, fb)
        if v != "DIFFER": fails.append(f"one-word change reported {v}, expected DIFFER")
        pages = [p["page"] for p in d["differing_pages"] if isinstance(p, dict)]
        if pages != [2]: fails.append(f"one-word change localized to {pages}, expected [2]")

        # DIRECTION 2: no defect -> must report IDENTICAL
        v2, _ = FP.compare(fa, fc)
        if v2 != "IDENTICAL": fails.append(f"byte-identical copy reported {v2}, expected IDENTICAL")

        # DIRECTION 3: page count moved -> must report DIFFER even before hashing pages
        fa2 = json.loads(json.dumps(fa)); fa2["n_pages"] += 1
        v3, d3 = FP.compare(fa, fa2)
        if v3 != "DIFFER": fails.append(f"page-count move reported {v3}, expected DIFFER")

    print("SELFTEST", "PASS" if not fails else "FAIL")
    for f in fails: print("  -", f)
    return 1 if fails else 0

sys.exit(main())
