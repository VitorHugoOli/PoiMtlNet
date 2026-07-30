#!/usr/bin/env python3
"""Extract the text layer of a PDF, page by page, to a JSON fingerprint.

Used by src_utils/_round7/27_comment_trim.md to prove a comment-only edit changed
nothing on the page.  Per page: sha256 of the extracted text + char count.  Also a
whole-document sha256 over the concatenation, and the page count.

Self-test (both directions) is in selftest_fingerprint.py: it must report DIFFER for
a PDF whose text moved and IDENTICAL for a byte-identical copy.
"""
import hashlib, json, sys
import pypdfium2 as pdfium


def fingerprint(path):
    doc = pdfium.PdfDocument(path)
    pages = []
    joined = []
    for i in range(len(doc)):
        tp = doc[i].get_textpage()
        txt = tp.get_text_range()
        pages.append({
            "page": i + 1,
            "chars": len(txt),
            "sha256": hashlib.sha256(txt.encode("utf-8")).hexdigest(),
        })
        joined.append(txt)
    return {
        "path": path,
        "n_pages": len(doc),
        "doc_sha256": hashlib.sha256("\x0c".join(joined).encode("utf-8")).hexdigest(),
        "total_chars": sum(p["chars"] for p in pages),
        "pages": pages,
        "_text": joined,
    }


def compare(a, b):
    """Return (verdict, details).  Compares page count, per-page hashes, doc hash."""
    out = {"n_pages": (a["n_pages"], b["n_pages"]),
           "doc_sha256_equal": a["doc_sha256"] == b["doc_sha256"],
           "total_chars": (a["total_chars"], b["total_chars"]),
           "differing_pages": []}
    if a["n_pages"] != b["n_pages"]:
        out["differing_pages"] = ["PAGE COUNT MOVED"]
        return "DIFFER", out
    for pa, pb in zip(a["pages"], b["pages"]):
        if pa["sha256"] != pb["sha256"]:
            out["differing_pages"].append(
                {"page": pa["page"], "chars": (pa["chars"], pb["chars"])})
    verdict = "IDENTICAL" if (out["doc_sha256_equal"] and not out["differing_pages"]) else "DIFFER"
    return verdict, out


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        a = json.load(open(sys.argv[2])); b = json.load(open(sys.argv[3]))
        v, d = compare(a, b)
        print(v)
        print(json.dumps(d, indent=1))
        sys.exit(0 if v == "IDENTICAL" else 1)
    fp = fingerprint(sys.argv[1])
    text = fp.pop("_text")
    json.dump(fp, open(sys.argv[2], "w"), indent=1)
    open(sys.argv[2] + ".txt", "w").write("\x0c".join(text))
    print(f"{fp['path']}: {fp['n_pages']} pages, {fp['total_chars']} chars, doc {fp['doc_sha256'][:16]}")
