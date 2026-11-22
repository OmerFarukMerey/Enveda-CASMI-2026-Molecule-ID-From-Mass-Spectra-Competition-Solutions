"""Prepare the public version of a scored kernel: same code cells (byte-identical), the notice cell replaced by a
title + description + credits, metadata is_private False and a readable title. The kernel id stays the same, so the
public score of the submitted version stays attached to the notebook.
usage: python -I publish.py <slug> <score> <description.md> [credits file, default credits.md]   -> work/publish/<slug>/ (push it with kaggle kernels push -p)
"""
import json, os, sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
slug, score, desc_path = sys.argv[1], sys.argv[2], sys.argv[3]
src_dir = os.path.join(HERE, "kernels", slug)
nb = json.load(open(os.path.join(src_dir, f"{slug}.ipynb"), encoding="utf-8"))
meta = json.load(open(os.path.join(src_dir, "kernel-metadata.json"), encoding="utf-8"))
desc = open(desc_path, encoding="utf-8").read()
credits = open(os.path.join(HERE, sys.argv[4] if len(sys.argv) > 4 else "credits.md"), encoding="utf-8").read()
desc = desc.replace("{score}", score).replace("{credits}", credits)
title_line = desc.splitlines()[0].lstrip("# ").strip()
src = lambda c: c["source"] if isinstance(c["source"], str) else "".join(c["source"])
assert nb["cells"][0]["cell_type"] == "markdown" and src(nb["cells"][0]).startswith("# CASMI26 " + slug)
code_before = [src(c) for c in nb["cells"] if c["cell_type"] == "code"]
nb["cells"][0]["source"] = desc.splitlines(keepends=True)
base_nb = os.environ.get("PUBLISH_BASE_NB")   # restore the base notebook's own write-up (its cell 0) after ours
if base_nb:
    b0 = json.load(open(base_nb, encoding="utf-8"))["cells"][0]
    assert b0["cell_type"] == "markdown"
    nb["cells"].insert(1, b0)
# a moved FR cell: its "Ours . 2" markdown goes right before it
fr_md = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "markdown" and src(c).startswith("## Ours · 2: FR 3-25")]
fr_code = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code" and "_F3_VERSION" in src(c)]
if len(fr_md) == 1 and len(fr_code) == 1 and fr_md[0] != fr_code[0] - 1:
    md = nb["cells"].pop(fr_md[0])
    j = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code" and "_F3_VERSION" in src(c)][0]
    nb["cells"].insert(j, md)
assert [src(c) for c in nb["cells"] if c["cell_type"] == "code"] == code_before  # code untouched
meta.update(title=title_line[:50], is_private=False, keywords=["chemistry"])
out = os.path.join(HERE, "publish", slug); os.makedirs(out, exist_ok=True)
json.dump(nb, open(os.path.join(out, f"{slug}.ipynb"), "w", encoding="utf-8", newline="\n"), ensure_ascii=False, indent=1)
json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w", encoding="utf-8"), indent=1)
print("prepared", out, "| title:", meta["title"], "| code cells unchanged:", len(code_before))
