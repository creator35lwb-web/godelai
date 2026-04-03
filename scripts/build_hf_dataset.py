"""
Build HuggingFace-compatible dataset from GodelAI conflict data.
Flattens all 107 items into a unified schema suitable for HF Datasets.

Author: L (GodelAI CEO) — MACP v2.2 "Identity"
"""

import json
import csv
from pathlib import Path
from datetime import datetime


def extract_text_content(item):
    """Extract the primary text content from any conflict item type."""
    parts = []
    
    item_type = item.get("type", "unknown")
    
    if item_type == "contradictory_facts":
        if "title" in item:
            parts.append(f"Title: {item['title']}")
        if "fact_a" in item:
            fa = item["fact_a"]
            if isinstance(fa, dict):
                parts.append(f"Fact A: {fa.get('statement', '')} Evidence: {fa.get('evidence', '')}")
            else:
                parts.append(f"Fact A: {fa}")
        if "fact_b" in item:
            fb = item["fact_b"]
            if isinstance(fb, dict):
                parts.append(f"Fact B: {fb.get('statement', '')} Evidence: {fb.get('evidence', '')}")
            else:
                parts.append(f"Fact B: {fb}")
        if "resolution" in item:
            parts.append(f"Resolution: {item['resolution']}")
        if "resolution_explanation" in item:
            parts.append(f"Resolution: {item['resolution_explanation']}")
    
    elif item_type == "ethical_dilemma":
        if "scenario" in item:
            parts.append(f"Scenario: {item['scenario']}")
        if "option_a" in item:
            oa = item["option_a"]
            if isinstance(oa, dict):
                parts.append(f"Option A: {oa.get('choice', '')} Reasoning: {oa.get('reasoning', '')}")
            else:
                parts.append(f"Option A: {oa}")
        if "option_b" in item:
            ob = item["option_b"]
            if isinstance(ob, dict):
                parts.append(f"Option B: {ob.get('choice', '')} Reasoning: {ob.get('reasoning', '')}")
            else:
                parts.append(f"Option B: {ob}")
        if "perspectives" in item and isinstance(item["perspectives"], list):
            for p in item["perspectives"]:
                if isinstance(p, dict):
                    parts.append(f"Perspective ({p.get('viewpoint', 'unknown')}): {p.get('argument', '')}")
        if "why_unresolvable" in item:
            parts.append(f"Why unresolvable: {item['why_unresolvable']}")
    
    elif item_type == "perspective_conflict":
        if "issue" in item:
            parts.append(f"Issue: {item['issue']}")
        if "perspectives" in item and isinstance(item["perspectives"], list):
            for p in item["perspectives"]:
                if isinstance(p, dict):
                    stakeholder = p.get("stakeholder", p.get("viewpoint", "unknown"))
                    position = p.get("position", p.get("argument", ""))
                    reasoning = p.get("reasoning", "")
                    parts.append(f"Perspective ({stakeholder}): {position} {reasoning}")
        if "synthesis_challenge" in item:
            parts.append(f"Synthesis challenge: {item['synthesis_challenge']}")
    
    elif item_type == "temporal_conflict":
        if "title" in item:
            parts.append(f"Title: {item['title']}")
        if "timeline" in item and isinstance(item["timeline"], list):
            for t in item["timeline"]:
                if isinstance(t, dict):
                    parts.append(f"Period {t.get('period', '?')}: {t.get('belief', '')} (Status: {t.get('status', '')})")
        if "query" in item:
            parts.append(f"Query: {item['query']}")
        if "lesson" in item:
            parts.append(f"Lesson: {item['lesson']}")
    
    return " ".join(parts)


def build_dataset():
    base = Path("datasets/conflict")
    
    files = [
        ("contradictory_facts", base / "contradictory_facts/scientific_paradoxes.json"),
        ("contradictory_facts", base / "contradictory_facts/expanded_paradoxes.json"),
        ("ethical_dilemmas", base / "ethical_dilemmas/core_dilemmas.json"),
        ("ethical_dilemmas", base / "ethical_dilemmas/expanded_dilemmas.json"),
        ("perspective_conflicts", base / "perspective_conflicts/ai_governance.json"),
        ("perspective_conflicts", base / "perspective_conflicts/expanded_perspectives.json"),
        ("temporal_conflicts", base / "temporal_conflicts/evolving_knowledge.json"),
        ("temporal_conflicts", base / "temporal_conflicts/expanded_temporal.json"),
    ]
    
    rows = []
    for category, filepath in files:
        if not filepath.exists():
            continue
        with open(filepath) as f:
            data = json.load(f)
        items = data.get("data", data) if isinstance(data, dict) else data
        source = "original" if "expanded" not in filepath.name else "expanded_v2"
        
        for item in items:
            text = extract_text_content(item)
            rows.append({
                "id": item.get("id", f"{category}_{len(rows)}"),
                "category": category,
                "domain": item.get("domain", "general"),
                "type": item.get("type", category),
                "text": text,
                "source": source,
                "raw_json": json.dumps(item),
            })
    
    # Write as JSONL (HuggingFace standard)
    out_dir = Path("hf_dataset")
    out_dir.mkdir(exist_ok=True)
    
    jsonl_path = out_dir / "godelai_conflict_data.jsonl"
    with open(jsonl_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    
    # Also write train split
    train_path = out_dir / "train.jsonl"
    with open(train_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    
    print(f"Built {len(rows)} items → {jsonl_path}")
    print(f"Categories: {set(r['category'] for r in rows)}")
    print(f"Domains: {set(r['domain'] for r in rows)}")
    
    # Stats
    for cat in sorted(set(r["category"] for r in rows)):
        count = sum(1 for r in rows if r["category"] == cat)
        avg_len = sum(len(r["text"]) for r in rows if r["category"] == cat) / count
        print(f"  {cat}: {count} items, avg text length: {avg_len:.0f} chars")
    
    return rows


if __name__ == "__main__":
    build_dataset()
