"""Create a standalone interactive HTML viewer from predict.py JSONL output."""

from __future__ import annotations

import argparse
import html
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import document_text, load_jsonl


def overlaps(labels, start, end):
    return any(max(start, int(x["start"])) < min(end, int(x["end"])) for x in labels)


def payload(row):
    response = row.get("response", "")
    return {
        "id": str(row.get("id_name", "")),
        "document": document_text(row),
        "response": response,
        "tokens": [
            {
                "start": int(t["char_start"]), "end": int(t["char_end"]), "text": t.get("text", response[int(t["char_start"]):int(t["char_end"])]),
                "gold": bool(overlaps(row.get("labels", []), int(t["char_start"]), int(t["char_end"]))),
                "predicted": bool(t.get("predicted", False)), "score": float(t.get("max_score", 0.0)),
                "evidence": t.get("predicted_token_positions", []), "evidence_tokens": t.get("predicted_tokens", []),
                "evidence_scores": t.get("top_k_scores", []),
            }
            for t in row.get("tokens", [])
        ],
    }


TEMPLATE = '''<!doctype html><meta charset="utf-8"><title>HalluSpan_EviAlign viewer</title>
<style>body{font:15px system-ui;margin:24px;max-width:1400px}article{border:1px solid #ddd;border-radius:8px;padding:16px;margin:16px 0}.doc,.response,.info{white-space:pre-wrap;line-height:1.8}.doc mark{background:#ffd166}.tok{cursor:pointer}.gold{font-weight:700}.pred{border-bottom:3px solid #d62828}.sel{background:#bde0fe}.score{color:#555;font:12px ui-monospace}button{margin:4px}</style>
<h1>HalluSpan_EviAlign evidence viewer</h1><p>Bold: gold hallucination. Red underline: predicted hallucination. Click a response token to highlight its retrieved document evidence.</p>
<div id="app"></div><script>const EXAMPLES=__DATA__;
const esc=s=>s.replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function renderText(ex,i){let p=0,s=''; for(const [j,t] of ex.tokens.entries()){s+=esc(ex.response.slice(p,t.start));let c='tok '+(t.gold?'gold ':'')+(t.predicted?'pred ':'');s+=`<span class="${c}" data-e="${i}" data-t="${j}">${esc(ex.response.slice(t.start,t.end))}</span>`;p=t.end}return s+esc(ex.response.slice(p))}
function render(){app.innerHTML=EXAMPLES.map((e,i)=>`<article><b>${esc(e.id)}</b><h3>Document</h3><div class="doc" id="d${i}">${esc(e.document)}</div><h3>Response</h3><div class="response">${renderText(e,i)}</div><div class="info" id="x${i}">Click a response token.</div></article>`).join('');document.querySelectorAll('.tok').forEach(n=>n.onclick=()=>select(+n.dataset.e,+n.dataset.t))}
function select(i,j){const e=EXAMPLES[i],t=e.tokens[j],d=document.getElementById('d'+i);let p=0,s='';const ranges=t.evidence||[];for(const r of ranges.sort((a,b)=>a.start-b.start)){s+=esc(e.document.slice(p,r.start))+'<mark>'+esc(e.document.slice(r.start,r.end))+'</mark>';p=r.end}s+=esc(e.document.slice(p));d.innerHTML=s;document.querySelectorAll(`[data-e="${i}"]`).forEach(x=>x.classList.remove('sel'));document.querySelector(`[data-e="${i}"][data-t="${j}"]`).classList.add('sel');document.getElementById('x'+i).innerHTML=`<b>${esc(t.text)}</b> &nbsp; score=${t.score.toFixed(4)}<br>`+(t.evidence_tokens||[]).map((x,k)=>`${k+1}. ${esc(x)} (${Number((t.evidence_scores||[])[k]||0).toFixed(4)})`).join('<br>')}
render();</script>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl")
    parser.add_argument("--output_html", default="")
    parser.add_argument("--num_examples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = load_jsonl(args.input_jsonl)
    if args.num_examples and len(rows) > args.num_examples:
        rows = random.Random(args.seed).sample(rows, args.num_examples)
    output = Path(args.output_html) if args.output_html else Path(args.input_jsonl).with_suffix(".html")
    output.write_text(TEMPLATE.replace("__DATA__", json.dumps([payload(row) for row in rows], ensure_ascii=False)), encoding="utf-8")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
