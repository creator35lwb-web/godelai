"""
Conflict Dataset Generator for GodelAI C-S-P Framework
=======================================================
Uses LLM (gpt-4.1-mini) to generate high-quality conflict data
targeting the T-Score range of 0.3-0.5 for C-S-P activation.

Categories:
1. Contradictory Facts (scientific paradoxes, historical disputes)
2. Ethical Dilemmas (trolley problems, AI ethics, medical ethics)
3. Perspective Conflicts (AI governance, climate policy, economic systems)
4. Temporal Conflicts (evolving scientific consensus, belief updates)

Author: L (GodelAI CEO) — MACP v2.2 "Identity"
Date: April 3, 2026
"""

import os
import json
import sys
from datetime import datetime
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("Installing openai...")
    os.system("pip3 install openai -q")
    from openai import OpenAI

client = OpenAI()

DATASET_DIR = Path(__file__).parent.parent / "datasets" / "conflict"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

GENERATION_PROMPTS = {
    "contradictory_facts": {
        "file": "contradictory_facts/expanded_paradoxes.json",
        "count": 20,
        "system": """You are a dataset engineer for the GodelAI C-S-P continual learning framework.
Generate contradictory fact pairs that create genuine cognitive tension — where both statements
have strong evidence but appear to contradict each other. These are used to test gradient diversity
in neural networks. Focus on: physics paradoxes, biological contradictions, economic paradoxes,
historical revisionism, mathematical counterintuitions.""",
        "user": """Generate {count} contradictory fact pairs for AI training. Each item must have:
- A clear domain (physics/biology/economics/history/mathematics)
- fact_a: A well-evidenced statement (1-2 sentences)
- fact_b: A contradictory but equally well-evidenced statement (1-2 sentences)  
- resolution: How experts currently reconcile the contradiction (1 sentence)
- conflict_intensity: "low" | "medium" | "high" (how genuinely contradictory they are)

Return ONLY valid JSON array, no markdown:
[
  {{
    "id": "contradiction_expanded_{n:03d}",
    "type": "contradictory_facts",
    "domain": "...",
    "title": "...",
    "fact_a": {{"statement": "...", "evidence": "..."}},
    "fact_b": {{"statement": "...", "evidence": "..."}},
    "resolution": "...",
    "conflict_intensity": "high"
  }}
]"""
    },
    "ethical_dilemmas": {
        "file": "ethical_dilemmas/expanded_dilemmas.json",
        "count": 25,
        "system": """You are a dataset engineer for the GodelAI C-S-P continual learning framework.
Generate ethical dilemmas where there is NO clear correct answer — scenarios where reasonable
people with good values would disagree. These create gradient diversity because the model must
hold multiple conflicting value frameworks simultaneously. Focus on: AI ethics, medical ethics,
environmental ethics, social justice, economic fairness, privacy vs security.""",
        "user": """Generate {count} ethical dilemmas for AI training. Each must be genuinely
unresolvable — not a trick question with an obvious answer. Include:
- scenario: A concrete situation requiring a decision (2-3 sentences)
- option_a: First choice with its ethical justification
- option_b: Second choice with its ethical justification
- ethical_frameworks: Which frameworks support each option (utilitarian/deontological/virtue/care)
- why_unresolvable: Why reasonable people genuinely disagree (1 sentence)
- domain: "ai_ethics" | "medical" | "environmental" | "social" | "economic" | "privacy"

Return ONLY valid JSON array:
[
  {{
    "id": "ethical_expanded_{n:03d}",
    "type": "ethical_dilemma",
    "domain": "...",
    "scenario": "...",
    "option_a": {{"choice": "...", "justification": "...", "frameworks": ["utilitarian"]}},
    "option_b": {{"choice": "...", "justification": "...", "frameworks": ["deontological"]}},
    "why_unresolvable": "..."
  }}
]"""
    },
    "perspective_conflicts": {
        "file": "perspective_conflicts/expanded_perspectives.json",
        "count": 20,
        "system": """You are a dataset engineer for the GodelAI C-S-P continual learning framework.
Generate perspective conflict scenarios where multiple stakeholders have fundamentally different
but internally consistent worldviews on the same issue. These test the model's ability to
hold multiple valid perspectives simultaneously without collapsing into one. Focus on:
AI governance, climate policy, economic systems, education reform, healthcare access, immigration.""",
        "user": """Generate {count} perspective conflict scenarios. Each must have 3-4 distinct
stakeholder perspectives that are all internally consistent and evidence-based.

Return ONLY valid JSON array:
[
  {{
    "id": "perspective_expanded_{n:03d}",
    "type": "perspective_conflict",
    "domain": "...",
    "issue": "...",
    "perspectives": [
      {{
        "stakeholder": "...",
        "position": "...",
        "reasoning": "...",
        "evidence": "..."
      }}
    ],
    "synthesis_challenge": "Why these perspectives cannot be easily reconciled (1 sentence)"
  }}
]"""
    },
    "temporal_conflicts": {
        "file": "temporal_conflicts/expanded_temporal.json",
        "count": 20,
        "system": """You are a dataset engineer for the GodelAI C-S-P continual learning framework.
Generate temporal conflict scenarios where scientific or social consensus has changed significantly
over time — where what was "true" at time T1 is now considered false or incomplete at time T2.
These test the model's ability to update beliefs without catastrophic forgetting of the reasoning
process. Focus on: medical discoveries, physics revisions, nutritional science, psychological
theories, economic models, historical reinterpretations.""",
        "user": """Generate {count} temporal conflict scenarios showing how understanding evolved.

Return ONLY valid JSON array:
[
  {{
    "id": "temporal_expanded_{n:03d}",
    "type": "temporal_conflict",
    "domain": "...",
    "title": "...",
    "timeline": [
      {{
        "period": "1950-1980",
        "belief": "...",
        "status": "accepted",
        "authority": "..."
      }},
      {{
        "period": "1980-2010",
        "belief": "...",
        "status": "transitional",
        "authority": "..."
      }},
      {{
        "period": "2010-present",
        "belief": "...",
        "status": "current",
        "authority": "..."
      }}
    ],
    "lesson": "What this evolution teaches about knowledge and certainty (1 sentence)"
  }}
]"""
    }
}


def generate_dataset(category: str, config: dict) -> list:
    """Generate a conflict dataset category using the LLM."""
    print(f"\n[L] Generating {config['count']} {category} items...")

    prompt_user = config["user"].format(count=config["count"], n=1)

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": config["system"]},
                {"role": "user", "content": prompt_user}
            ],
            temperature=0.8,
            max_tokens=8000,
        )

        content = response.choices[0].message.content.strip()

        # Clean up potential markdown code blocks
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])

        data = json.loads(content)
        print(f"  ✅ Generated {len(data)} items for {category}")
        return data

    except json.JSONDecodeError as e:
        print(f"  ❌ JSON parse error for {category}: {e}")
        print(f"  Raw content (first 500 chars): {content[:500]}")
        return []
    except Exception as e:
        print(f"  ❌ Generation error for {category}: {e}")
        return []


def save_dataset(data: list, filepath: Path, category: str, title: str):
    """Save dataset in GodelAI standard format."""
    output = {
        "$schema": "https://godelai.dev/schemas/conflict-dataset-v1.json",
        "title": title,
        "version": "2.0",
        "created_by": "L (GodelAI CEO) — MACP v2.2 via LLM generation",
        "created_at": datetime.now().isoformat(),
        "description": f"Expanded {category} conflict dataset for C-S-P T-Score activation (target: T=0.3-0.5)",
        "count": len(data),
        "data": data
    }

    filepath = DATASET_DIR / filepath
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  💾 Saved {len(data)} items to {filepath}")
    return filepath


def main():
    print("=" * 70)
    print("GodelAI Conflict Dataset Generator v2.0")
    print("Author: L (GodelAI CEO) — MACP v2.2 'Identity'")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    all_results = {}

    for category, config in GENERATION_PROMPTS.items():
        data = generate_dataset(category, config)

        if data:
            filepath = save_dataset(
                data,
                Path(config["file"]),
                category,
                f"GodelAI {category.replace('_', ' ').title()} — Expanded v2.0"
            )
            all_results[category] = {
                "count": len(data),
                "file": str(filepath)
            }
        else:
            all_results[category] = {"count": 0, "error": "generation failed"}

    # Summary
    print("\n" + "=" * 70)
    print("GENERATION SUMMARY")
    print("=" * 70)
    total = 0
    for cat, result in all_results.items():
        count = result.get("count", 0)
        total += count
        status = "✅" if count > 0 else "❌"
        print(f"  {status} {cat}: {count} items")

    print(f"\n  Total conflict items generated: {total}")
    print(f"  Previous total: 22 items")
    print(f"  Expansion factor: {total/22:.1f}x" if total > 0 else "")

    # Save summary
    summary_path = DATASET_DIR / "generation_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_items": total,
            "previous_total": 22,
            "results": all_results
        }, f, indent=2)

    print(f"\n  Summary saved to: {summary_path}")
    return all_results


if __name__ == "__main__":
    main()
