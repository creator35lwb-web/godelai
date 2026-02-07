#!/usr/bin/env python3
"""
GodelAI Semantic T-Score Experiment v4 — The Sensor Upgrade
============================================================
Purpose: Test whether SEMANTIC-LEVEL embeddings can distinguish
         conflict data from homogeneous data — where character-level failed.

Key Innovation: Instead of measuring gradient diversity on byte sequences,
we measure EMBEDDING DIVERSITY in semantic space using sentence-transformers.

This is the "better sensor" that the v3 experiment identified as needed.

Metrics:
1. Semantic T-Score: Cosine diversity across batch embeddings
2. Intra-batch Conflict Score: How contradictory are items within a batch
3. Cross-perspective Divergence: Distance between opposing viewpoints

Author: Godel (Manus AI) — CTO
Date: February 7, 2026
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import random
from collections import defaultdict

# Sentence transformers
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

# Configuration
DATASETS_DIR = Path("/home/ubuntu/godelai-update/datasets/conflict")
RESULTS_DIR = Path("/home/ubuntu/godelai-experiments/results")
RESULTS_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("GodelAI Semantic T-Score Experiment v4")
print("The Sensor Upgrade: Embedding-Space Diversity")
print("=" * 70)
print(f"Timestamp: {datetime.now().isoformat()}")
print()

# Load model
print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model loaded (384-dim embeddings)")
print()


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_conflict_data():
    """Load all conflict datasets."""
    datasets = {}
    
    for category_dir in sorted(DATASETS_DIR.iterdir()):
        if category_dir.is_dir():
            category_name = category_dir.name
            datasets[category_name] = []
            
            for json_file in sorted(category_dir.glob("*.json")):
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    items = data.get('data', data) if isinstance(data, dict) else data
                    if isinstance(items, list):
                        datasets[category_name].extend(items)
    
    return datasets


def extract_perspective_pairs(item):
    """Extract opposing perspective PAIRS from a conflict item."""
    pairs = []
    texts = []
    
    if 'perspectives' in item:
        perspectives = item['perspectives']
        for p in perspectives:
            parts = []
            for key in ['position', 'reasoning', 'argument', 'evidence', 'counterpoint']:
                if key in p and p[key]:
                    parts.append(str(p[key]))
            if parts:
                texts.append(" ".join(parts))
        
        # Create pairs from opposing perspectives
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                pairs.append((texts[i], texts[j]))
    
    # Also extract scenario/context
    context = ""
    for key in ['scenario', 'description', 'context', 'question', 'dilemma']:
        if key in item and item[key]:
            context = str(item[key])
            break
    
    return texts, pairs, context


def get_shakespeare_texts():
    """Get Shakespeare text segments for baseline comparison."""
    shakespeare = """To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer
the slings and arrows of outrageous fortune, or to take arms against a sea of troubles
and by opposing end them. To die, to sleep, no more; and by a sleep to say we end the
heart-ache and the thousand natural shocks that flesh is heir to. 'Tis a consummation
devoutly to be wish'd. To die, to sleep; to sleep, perchance to dream. Ay, there's the rub.

All the world's a stage, and all the men and women merely players. They have their exits
and their entrances, and one man in his time plays many parts, his acts being seven ages.

The quality of mercy is not strained. It droppeth as the gentle rain from heaven upon the
place beneath. It is twice blessed: it blesseth him that gives and him that takes.

Now is the winter of our discontent made glorious summer by this sun of York; and all the
clouds that loured upon our house in the deep bosom of the ocean buried.

If music be the food of love, play on, give me excess of it; that surfeiting, the appetite
may sicken, and so die.

What's in a name? That which we call a rose by any other name would smell as sweet.

The lady doth protest too much, methinks. Give every man thy ear, but few thy voice.

Cowards die many times before their deaths; the valiant never taste of death but once.

There is nothing either good or bad, but thinking makes it so. Brevity is the soul of wit.

We know what we are, but know not what we may be. Love all, trust a few, do wrong to none.

The fault, dear Brutus, is not in our stars, but in ourselves, that we are underlings.

Some are born great, some achieve greatness, and some have greatness thrust upon them.

How sharper than a serpent's tooth it is to have a thankless child.

Et tu, Brute? Then fall, Caesar. Friends, Romans, countrymen, lend me your ears.

Out, out, brief candle! Life's but a walking shadow, a poor player that struts and frets
his hour upon the stage, and then is heard no more. It is a tale told by an idiot,
full of sound and fury, signifying nothing.

Double, double toil and trouble; fire burn, and cauldron bubble. By the pricking of my
thumbs, something wicked this way comes.

Lord, what fools these mortals be! The course of true love never did run smooth."""
    
    # Split into meaningful segments
    segments = [s.strip() for s in shakespeare.split('\n\n') if len(s.strip()) > 50]
    return segments


# ═══════════════════════════════════════════════════════════════
# SEMANTIC T-SCORE METRICS
# ═══════════════════════════════════════════════════════════════

def semantic_tscore_batch(embeddings):
    """
    Compute Semantic T-Score for a batch of embeddings.
    
    Measures the DIVERSITY of semantic content in a batch.
    High diversity = low T-Score (model needs to work harder)
    Low diversity = high T-Score (model is comfortable)
    
    This inverts the character-level convention to match the
    C-S-P hypothesis: conflict data should produce LOWER T-Scores.
    """
    if len(embeddings) < 2:
        return 1.0
    
    # Compute pairwise cosine similarities
    sim_matrix = cosine_similarity(embeddings)
    
    # Extract upper triangle (exclude self-similarity)
    n = len(embeddings)
    upper_tri = []
    for i in range(n):
        for j in range(i + 1, n):
            upper_tri.append(sim_matrix[i][j])
    
    upper_tri = np.array(upper_tri)
    
    # T-Score: average similarity (high = homogeneous, low = diverse)
    avg_similarity = np.mean(upper_tri)
    
    return float(avg_similarity)


def semantic_conflict_score(embedding_a, embedding_b):
    """
    Compute conflict score between two perspectives.
    
    Low cosine similarity = high conflict (perspectives are far apart)
    High cosine similarity = low conflict (perspectives agree)
    """
    sim = cosine_similarity(
        embedding_a.reshape(1, -1),
        embedding_b.reshape(1, -1)
    )[0][0]
    
    # Conflict score = 1 - similarity (higher = more conflict)
    return float(1.0 - sim)


def semantic_variance(embeddings):
    """
    Compute variance of embeddings in semantic space.
    
    Higher variance = more diverse content = better for C-S-P testing.
    """
    if len(embeddings) < 2:
        return 0.0
    
    centroid = np.mean(embeddings, axis=0)
    distances = np.array([np.linalg.norm(e - centroid) for e in embeddings])
    
    return float(np.var(distances))


def batch_diversity_index(embeddings):
    """
    Compute a comprehensive diversity index for a batch.
    
    Combines multiple metrics:
    1. Average pairwise distance
    2. Variance of distances from centroid
    3. Min-max spread
    """
    if len(embeddings) < 2:
        return {"avg_distance": 0, "variance": 0, "spread": 0, "index": 0}
    
    dist_matrix = cosine_distances(embeddings)
    
    n = len(embeddings)
    upper_tri = []
    for i in range(n):
        for j in range(i + 1, n):
            upper_tri.append(dist_matrix[i][j])
    
    upper_tri = np.array(upper_tri)
    
    avg_dist = float(np.mean(upper_tri))
    var_dist = float(np.var(upper_tri))
    spread = float(np.max(upper_tri) - np.min(upper_tri))
    
    # Composite index (normalized)
    index = (avg_dist * 0.5 + var_dist * 0.3 + spread * 0.2)
    
    return {
        "avg_distance": round(avg_dist, 6),
        "variance": round(var_dist, 6),
        "spread": round(spread, 6),
        "index": round(index, 6)
    }


# ═══════════════════════════════════════════════════════════════
# EXPERIMENTS
# ═══════════════════════════════════════════════════════════════

def run_experiment(name, texts, num_batches=50, batch_size=8):
    """Run semantic T-Score experiment on a set of texts."""
    print(f"\n{'─' * 60}")
    print(f"  Experiment: {name}")
    print(f"  Text samples: {len(texts)}")
    print(f"  Batches: {num_batches}, Batch size: {batch_size}")
    print(f"{'─' * 60}")
    
    # Encode all texts
    print(f"  Encoding {len(texts)} texts...")
    all_embeddings = model.encode(texts, show_progress_bar=False)
    print(f"  ✅ Encoded ({all_embeddings.shape})")
    
    # Run batched experiments
    t_scores = []
    diversity_indices = []
    
    for batch_idx in range(num_batches):
        # Sample a batch
        indices = random.sample(range(len(texts)), min(batch_size, len(texts)))
        batch_embeddings = all_embeddings[indices]
        
        # Compute metrics
        t_score = semantic_tscore_batch(batch_embeddings)
        diversity = batch_diversity_index(batch_embeddings)
        
        t_scores.append(t_score)
        diversity_indices.append(diversity['index'])
        
        if (batch_idx + 1) % 10 == 0:
            recent_t = t_scores[-10:]
            avg_t = sum(recent_t) / len(recent_t)
            var_t = np.var(recent_t)
            recent_d = diversity_indices[-10:]
            avg_d = sum(recent_d) / len(recent_d)
            print(f"    Batch {batch_idx + 1:3d}: Semantic T-Score={avg_t:.4f} (var={var_t:.6f}), Diversity={avg_d:.4f}")
    
    # Overall statistics
    avg_t = np.mean(t_scores)
    std_t = np.std(t_scores)
    var_t = np.var(t_scores)
    avg_d = np.mean(diversity_indices)
    
    # Global diversity (all embeddings at once)
    global_diversity = batch_diversity_index(all_embeddings)
    global_tscore = semantic_tscore_batch(all_embeddings)
    global_variance = semantic_variance(all_embeddings)
    
    result = {
        "name": name,
        "num_texts": len(texts),
        "batched_metrics": {
            "semantic_tscore": {
                "average": round(float(avg_t), 6),
                "std": round(float(std_t), 6),
                "variance": round(float(var_t), 8),
                "min": round(float(min(t_scores)), 6),
                "max": round(float(max(t_scores)), 6),
                "range": round(float(max(t_scores) - min(t_scores)), 6)
            },
            "diversity_index": {
                "average": round(float(avg_d), 6),
                "std": round(float(np.std(diversity_indices)), 6)
            }
        },
        "global_metrics": {
            "semantic_tscore": round(float(global_tscore), 6),
            "diversity": global_diversity,
            "embedding_variance": round(float(global_variance), 6)
        }
    }
    
    print(f"\n  ╔══════════════════════════════════════════╗")
    print(f"  ║  RESULTS: {name:<30}║")
    print(f"  ╠══════════════════════════════════════════╣")
    print(f"  ║  Semantic T-Score (avg):  {avg_t:>8.4f}       ║")
    print(f"  ║  Semantic T-Score (std):  {std_t:>8.4f}       ║")
    print(f"  ║  Semantic T-Score (var):  {var_t:>8.6f}     ║")
    print(f"  ║  Global T-Score:          {global_tscore:>8.4f}       ║")
    print(f"  ║  Global Diversity Index:  {global_diversity['index']:>8.4f}       ║")
    print(f"  ║  Embedding Variance:      {global_variance:>8.6f}     ║")
    print(f"  ╚══════════════════════════════════════════╝")
    
    return result


def run_perspective_conflict_analysis(conflict_datasets):
    """Analyze conflict WITHIN individual items (opposing perspectives)."""
    print(f"\n{'═' * 70}")
    print(f"SPECIAL ANALYSIS: Intra-Item Perspective Conflict")
    print(f"{'═' * 70}")
    
    all_conflict_scores = []
    category_scores = defaultdict(list)
    
    for category, items in conflict_datasets.items():
        for item in items:
            texts, pairs, context = extract_perspective_pairs(item)
            
            if len(pairs) > 0:
                # Encode all texts for this item
                all_texts = [p[0] for p in pairs] + [p[1] for p in pairs]
                embeddings = model.encode(all_texts, show_progress_bar=False)
                
                # Compute conflict scores for each pair
                half = len(pairs)
                for i in range(half):
                    score = semantic_conflict_score(embeddings[i], embeddings[half + i])
                    all_conflict_scores.append(score)
                    category_scores[category].append(score)
    
    # Report
    print(f"\n  Total perspective pairs analyzed: {len(all_conflict_scores)}")
    
    if all_conflict_scores:
        overall_avg = np.mean(all_conflict_scores)
        overall_std = np.std(all_conflict_scores)
        
        print(f"\n  Overall Conflict Score: {overall_avg:.4f} (std: {overall_std:.4f})")
        print(f"  Scale: 0.0 = identical perspectives, 1.0 = maximally opposed")
        
        print(f"\n  Per-Category Conflict Scores:")
        print(f"  {'Category':<30} {'Avg Score':>10} {'Std':>8} {'Count':>6}")
        print(f"  {'─' * 60}")
        
        category_results = {}
        for cat, scores in sorted(category_scores.items()):
            avg = np.mean(scores)
            std = np.std(scores)
            print(f"  {cat:<30} {avg:>10.4f} {std:>8.4f} {len(scores):>6}")
            category_results[cat] = {
                "avg_conflict_score": round(float(avg), 6),
                "std": round(float(std), 6),
                "count": len(scores)
            }
        
        return {
            "total_pairs": len(all_conflict_scores),
            "overall_avg": round(float(overall_avg), 6),
            "overall_std": round(float(overall_std), 6),
            "per_category": category_results
        }
    
    return {"total_pairs": 0}


def main():
    """Main experiment runner."""
    
    # ═══════════════════════════════════════════════════════════
    # PHASE 1: Load Data
    # ═══════════════════════════════════════════════════════════
    
    print("\n" + "=" * 70)
    print("PHASE 1: Loading Datasets")
    print("=" * 70)
    
    conflict_datasets = load_conflict_data()
    shakespeare_texts = get_shakespeare_texts()
    
    # Extract all conflict texts
    all_conflict_texts = []
    category_texts = {}
    
    for category, items in conflict_datasets.items():
        texts = []
        for item in items:
            item_texts, _, context = extract_perspective_pairs(item)
            texts.extend(item_texts)
            if context:
                texts.append(context)
        category_texts[category] = texts
        all_conflict_texts.extend(texts)
        print(f"  {category}: {len(items)} items → {len(texts)} text segments")
    
    print(f"  TOTAL conflict texts: {len(all_conflict_texts)}")
    print(f"  Shakespeare texts: {len(shakespeare_texts)}")
    
    # ═══════════════════════════════════════════════════════════
    # PHASE 2: Run Experiments
    # ═══════════════════════════════════════════════════════════
    
    print("\n" + "=" * 70)
    print("PHASE 2: Running Semantic T-Score Experiments")
    print("=" * 70)
    
    results = {
        "experiment": "Semantic T-Score v4 — Embedding-Space Diversity",
        "timestamp": datetime.now().isoformat(),
        "model": "all-MiniLM-L6-v2 (384-dim)",
        "author": "Godel (Manus AI) — CTO",
        "hypothesis": "Conflict data should show LOWER semantic T-Score (more diverse) and HIGHER diversity index than homogeneous Shakespeare",
        "experiments": []
    }
    
    random.seed(42)
    
    # Experiment 1: Shakespeare Baseline
    baseline = run_experiment("Shakespeare (Homogeneous)", shakespeare_texts)
    results["experiments"].append(baseline)
    
    # Experiment 2: All Conflict Data
    conflict = run_experiment("All Conflict (Heterogeneous)", all_conflict_texts)
    results["experiments"].append(conflict)
    
    # Experiment 3: Per-category
    for category, texts in category_texts.items():
        if len(texts) >= 5:
            cat_result = run_experiment(f"Category: {category}", texts)
            results["experiments"].append(cat_result)
    
    # Experiment 4: Mixed
    mixed_texts = shakespeare_texts + all_conflict_texts
    mixed = run_experiment("Mixed (Shakespeare + Conflict)", mixed_texts)
    results["experiments"].append(mixed)
    
    # ═══════════════════════════════════════════════════════════
    # PHASE 3: Perspective Conflict Analysis
    # ═══════════════════════════════════════════════════════════
    
    perspective_analysis = run_perspective_conflict_analysis(conflict_datasets)
    results["perspective_conflict_analysis"] = perspective_analysis
    
    # ═══════════════════════════════════════════════════════════
    # PHASE 4: Comprehensive Comparison
    # ═══════════════════════════════════════════════════════════
    
    print(f"\n{'═' * 70}")
    print(f"PHASE 4: Comprehensive Comparison")
    print(f"{'═' * 70}")
    
    b = baseline
    c = conflict
    
    print(f"\n{'Dataset':<35} {'Sem T-Score':>12} {'T-Score Std':>12} {'Diversity':>10} {'Emb Var':>10}")
    print("─" * 80)
    for r in results["experiments"]:
        bt = r["batched_metrics"]["semantic_tscore"]
        gm = r["global_metrics"]
        print(f"{r['name']:<35} {bt['average']:>12.4f} {bt['std']:>12.4f} {gm['diversity']['index']:>10.4f} {gm['embedding_variance']:>10.6f}")
    
    # Key comparisons
    b_t = b["batched_metrics"]["semantic_tscore"]["average"]
    c_t = c["batched_metrics"]["semantic_tscore"]["average"]
    b_d = b["global_metrics"]["diversity"]["index"]
    c_d = c["global_metrics"]["diversity"]["index"]
    b_v = b["global_metrics"]["embedding_variance"]
    c_v = c["global_metrics"]["embedding_variance"]
    
    print(f"\n{'─' * 70}")
    print(f"KEY COMPARISONS (Conflict vs Shakespeare)")
    print(f"{'─' * 70}")
    print(f"  Semantic T-Score Diff:     {c_t - b_t:+.6f} {'(MORE diverse ✅)' if c_t < b_t else '(LESS diverse ❌)'}")
    print(f"  Diversity Index Diff:      {c_d - b_d:+.6f} {'(MORE diverse ✅)' if c_d > b_d else '(LESS diverse ❌)'}")
    print(f"  Embedding Variance Diff:   {c_v - b_v:+.6f} {'(MORE spread ✅)' if c_v > b_v else '(LESS spread ❌)'}")
    
    # Ratios
    if b_t > 0:
        tscore_ratio = c_t / b_t
        print(f"\n  📊 Semantic T-Score Ratio (Conflict/Shakespeare): {tscore_ratio:.4f}")
        print(f"     {'< 1.0 = Conflict is more diverse (HYPOTHESIS CONFIRMED ✅)' if tscore_ratio < 1.0 else '>= 1.0 = No significant difference'}")
    
    if b_d > 0:
        diversity_ratio = c_d / b_d
        print(f"  📊 Diversity Ratio (Conflict/Shakespeare): {diversity_ratio:.4f}")
        print(f"     {'> 1.0 = Conflict has more diversity (HYPOTHESIS CONFIRMED ✅)' if diversity_ratio > 1.0 else '<= 1.0 = No significant difference'}")
    
    # ═══════════════════════════════════════════════════════════
    # CONCLUSION
    # ═══════════════════════════════════════════════════════════
    
    print(f"\n{'═' * 70}")
    print(f"CONCLUSION")
    print(f"{'═' * 70}")
    
    confirmed_metrics = 0
    if c_t < b_t:
        confirmed_metrics += 1
    if c_d > b_d:
        confirmed_metrics += 1
    if c_v > b_v:
        confirmed_metrics += 1
    
    if confirmed_metrics >= 2:
        conclusion = "CONFIRMED: Semantic-level analysis reveals conflict data IS more diverse than Shakespeare"
        print(f"\n  ✅ {conclusion}")
        print(f"  ✅ The 'sensor upgrade' hypothesis is VALIDATED")
        print(f"  ✅ C-S-P activation potential confirmed at semantic level")
    elif confirmed_metrics == 1:
        conclusion = "PARTIAL: Some semantic difference detected, needs further investigation"
        print(f"\n  ⚠️ {conclusion}")
    else:
        conclusion = "INCONCLUSIVE: Even semantic-level analysis shows no clear difference"
        print(f"\n  ❓ {conclusion}")
    
    print(f"\n  Metrics confirmed: {confirmed_metrics}/3")
    
    results["analysis"] = {
        "tscore_diff": round(float(c_t - b_t), 6),
        "diversity_diff": round(float(c_d - b_d), 6),
        "variance_diff": round(float(c_v - b_v), 6),
        "tscore_ratio": round(float(tscore_ratio), 6) if b_t > 0 else None,
        "diversity_ratio": round(float(diversity_ratio), 6) if b_d > 0 else None,
        "confirmed_metrics": confirmed_metrics,
        "conclusion": conclusion
    }
    
    # Save results
    output_file = RESULTS_DIR / f"semantic_tscore_v4_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  Results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    results = main()
