"""Turn the report Markdown into a Lark-friendly Markdown variant.

Lark imports a single line break as a hard break and collapses ordered-list
numbering, so this variant:
- joins wrapped paragraph/blockquote lines into one line,
- writes ordered steps as `N、` text with a blank line between items,
- keeps tables, code fences and the image line intact.
"""
from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SRC = HERE / "report" / "sdpr_forward_network_dataset.md"
OUT = HERE / "kb" / "01_feishu.md"

CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")


def smart_join(lines):
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not out:
            out = line
        elif CJK.search(out[-1]) and CJK.search(line[0]):
            out += line
        else:
            out += " " + line
    return out


def is_block_start(s):
    return (
        re.match(r"^#{1,3}\s+", s)
        or s.startswith("|")
        or s.startswith("> ")
        or re.match(r"^[-*]\s+", s)
        or re.match(r"^\d+\.\s+", s)
        or s.startswith("```")
        or s.startswith("![")
    )


def main():
    lines = SRC.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# ")
    lines = lines[1:]
    out = []
    i = 0
    while i < len(lines):
        raw = lines[i]
        s = raw.strip()
        if not s:
            if out and out[-1] != "":
                out.append("")
            i += 1
            continue

        if s.startswith("```"):
            out.append(s)
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                out.append(lines[i])
                i += 1
            out.append("```")
            i += 1
            continue

        if re.match(r"^#{1,3}\s+", s):
            out.append(s)
            i += 1
            continue

        if s.startswith("|"):
            while i < len(lines) and lines[i].strip().startswith("|"):
                out.append(lines[i].strip())
                i += 1
            continue

        if s.startswith("!["):
            out.append(s.replace("](figures/pipeline.png)", "](@./report/figures/pipeline.png)"))
            i += 1
            continue

        if s.startswith("> "):
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip()[1:].strip())
                i += 1
            out.append("> " + smart_join(buf))
            out.append("")
            continue

        m = re.match(r"^(\d+)\.\s+(.*)$", s)
        if m:
            n, rest = m.group(1), m.group(2)
            cont = []
            i += 1
            while i < len(lines):
                nxt = lines[i].strip()
                if not nxt or is_block_start(nxt):
                    break
                cont.append(nxt)
                i += 1
            out.append(f"{n}、{smart_join([rest] + cont)}")
            out.append("")
            continue

        if re.match(r"^[-*]\s+", s):
            item = re.sub(r"^[-*]\s+", "", s)
            cont = []
            i += 1
            while i < len(lines):
                nxt = lines[i].strip()
                if not nxt or is_block_start(nxt):
                    break
                cont.append(nxt)
                i += 1
            out.append("- " + smart_join([item] + cont))
            continue

        buf = []
        while i < len(lines):
            nxt = lines[i].strip()
            if not nxt or is_block_start(nxt):
                break
            buf.append(nxt)
            i += 1
        out.append(smart_join(buf))
        out.append("")

    text = "\n".join(out).strip() + "\n"
    # Lark Markdown treats angle-bracket autolinks as XML; use normal links.
    text = text.replace(
        "<https://github.com/utiasASRL/sdprlayer>",
        "[github.com/utiasASRL/sdprlayer](https://github.com/utiasASRL/sdprlayer)",
    )
    text = text.replace(
        "<https://github.com/utiasASRL/deep_learned_visual_features/tree/mat-weight-sdp-version>",
        "[deep_learned_visual_features（mat-weight-sdp-version）](https://github.com/utiasASRL/deep_learned_visual_features/tree/mat-weight-sdp-version)",
    )
    text = text.replace(
        "<http://asrl.utias.utoronto.ca/datasets/2020-vtr-dataset/>",
        "[asrl.utias.utoronto.ca/datasets/2020-vtr-dataset](http://asrl.utias.utoronto.ca/datasets/2020-vtr-dataset/)",
    )
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
