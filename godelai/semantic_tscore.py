"""
GodelAI Semantic T-Score — Meaning-Level Diversity Measurement

Uses sentence-transformers to compute diversity of text embeddings,
applying the same T-Score formula used for gradient diversity but
at the semantic level.

Hypothesis: Higher semantic tension (greater embedding diversity)
correlates with lower gradient T-Score and stronger C-S-P activation.

Author: Claude Code (Opus 4.6) — Lead Engineer
Date: February 7, 2026
Protocol: MACP v2.0
"""

import numpy as np
from typing import List, Dict, Optional, Tuple


class SemanticTScore:
    """
    Compute T-Score at the semantic level using sentence embeddings.

    Mirrors the gradient-diversity T-Score formula:
        T = 1 - (||sum(e_i)||^2 / sum(||e_i||^2)) / N

    Where e_i are sentence embeddings instead of gradients.

    - T near 0: All texts say the same thing (low diversity)
    - T near 1: Texts are maximally diverse in meaning
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize with a sentence-transformer model.

        Args:
            model_name: HuggingFace model ID for sentence-transformers.
                        Default 'all-MiniLM-L6-v2' is fast and effective.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers required. Install with: "
                "pip install sentence-transformers"
            )
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts into embeddings.

        Args:
            texts: List of text strings.

        Returns:
            np.ndarray of shape [N, embedding_dim]
        """
        return self.model.encode(texts, convert_to_numpy=True)

    def compute_tscore(self, texts: List[str]) -> float:
        """
        Compute semantic T-Score for a batch of texts.

        Uses the same formula as GodelAgent.measure_gradient_diversity()
        but applied to sentence embeddings instead of gradients.

        Args:
            texts: List of text strings (batch).

        Returns:
            float: Semantic T-Score in [0, 1].
                   0 = all texts semantically identical
                   1 = maximally diverse meanings
        """
        if len(texts) < 2:
            return 0.5  # Cannot measure diversity with < 2 texts

        embeddings = self.embed(texts)
        return self._tscore_from_embeddings(embeddings)

    def _tscore_from_embeddings(self, embeddings: np.ndarray) -> float:
        """
        Core T-Score computation from embedding matrix.

        Formula (mirrors gradient T-Score):
            sum_emb_norm = ||sum(e_i)||^2
            sum_norm_emb = sum(||e_i||^2)
            ratio = sum_emb_norm / (sum_norm_emb + eps)
            T = 1 - ratio / N

        Args:
            embeddings: np.ndarray of shape [N, dim]

        Returns:
            float: T-Score in [0, 1]
        """
        n = embeddings.shape[0]
        eps = 1e-8

        # ||sum(e_i)||^2 — how much embeddings align
        sum_emb = np.sum(embeddings, axis=0)
        sum_emb_norm = np.dot(sum_emb, sum_emb)

        # sum(||e_i||^2) — total individual magnitude
        sum_norm_emb = np.sum(np.linalg.norm(embeddings, axis=1) ** 2)

        # Diversity ratio (same formula as gradient T-Score)
        ratio = sum_emb_norm / (sum_norm_emb + eps)

        # Linear normalization: T = 1 - ratio/N
        t_score = 1.0 - min(1.0, max(0.0, ratio / n))

        return float(t_score)

    def compute_pairwise_tension(self, text_a: str, text_b: str) -> float:
        """
        Compute semantic tension between two texts using cosine distance.

        Args:
            text_a: First text (e.g., position_a claim)
            text_b: Second text (e.g., position_b claim)

        Returns:
            float: Cosine distance in [0, 2].
                   0 = identical meaning
                   1 = orthogonal
                   2 = opposite meaning
        """
        embeddings = self.embed([text_a, text_b])
        cosine_sim = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]) + 1e-8
        )
        return float(1.0 - cosine_sim)

    def analyze_conflict_sample(self, sample: dict) -> Dict[str, float]:
        """
        Analyze a single conflict data sample for semantic tension.

        Expects the schema from docs/CLAUDE_CODE_CONFLICT_DATA_SCALING_GUIDE.md:
        - position_a.claim / position_b.claim
        - training_text

        Also handles the old schema:
        - perspectives[].position / perspectives[].reasoning
        - fact_a.statement / fact_b.statement

        Args:
            sample: A conflict data dict.

        Returns:
            dict with semantic metrics:
                - pairwise_tension: cosine distance between positions
                - batch_tscore: T-Score across all text fragments
                - num_fragments: how many text pieces were analyzed
        """
        texts = []

        # New schema (position_a / position_b)
        if 'position_a' in sample and 'position_b' in sample:
            pa = sample['position_a']
            pb = sample['position_b']
            claim_a = pa.get('claim', '')
            claim_b = pb.get('claim', '')
            if claim_a:
                texts.append(claim_a)
            if claim_b:
                texts.append(claim_b)
            if pa.get('reasoning'):
                texts.append(pa['reasoning'])
            if pb.get('reasoning'):
                texts.append(pb['reasoning'])

        # Old schema (perspectives array)
        elif 'perspectives' in sample:
            for p in sample['perspectives']:
                if p.get('position'):
                    texts.append(p['position'])
                if p.get('reasoning'):
                    texts.append(p['reasoning'])

        # Old schema (fact_a / fact_b)
        elif 'fact_a' in sample and 'fact_b' in sample:
            claim_a = sample['fact_a'].get('statement', '')
            claim_b = sample['fact_b'].get('statement', '')
            if claim_a:
                texts.append(claim_a)
            if claim_b:
                texts.append(claim_b)

        # Training text
        if sample.get('training_text'):
            texts.append(sample['training_text'])
        elif sample.get('training_prompt'):
            texts.append(sample['training_prompt'])

        if len(texts) < 2:
            return {
                'pairwise_tension': 0.0,
                'batch_tscore': 0.5,
                'num_fragments': len(texts)
            }

        # Pairwise tension between the two main positions
        pairwise = self.compute_pairwise_tension(texts[0], texts[1])

        # Batch T-Score across all fragments
        batch_tscore = self.compute_tscore(texts)

        return {
            'pairwise_tension': pairwise,
            'batch_tscore': batch_tscore,
            'num_fragments': len(texts)
        }

    def analyze_dataset(self, samples: List[dict]) -> Dict[str, float]:
        """
        Analyze an entire conflict dataset for semantic tension distribution.

        Args:
            samples: List of conflict data dicts.

        Returns:
            dict: Aggregate statistics on semantic tension.
        """
        tensions = []
        tscores = []

        for sample in samples:
            result = self.analyze_conflict_sample(sample)
            tensions.append(result['pairwise_tension'])
            tscores.append(result['batch_tscore'])

        if not tensions:
            return {'count': 0}

        return {
            'count': len(tensions),
            'avg_pairwise_tension': float(np.mean(tensions)),
            'std_pairwise_tension': float(np.std(tensions)),
            'min_pairwise_tension': float(np.min(tensions)),
            'max_pairwise_tension': float(np.max(tensions)),
            'avg_semantic_tscore': float(np.mean(tscores)),
            'std_semantic_tscore': float(np.std(tscores)),
            'min_semantic_tscore': float(np.min(tscores)),
            'max_semantic_tscore': float(np.max(tscores)),
        }

    def compare_with_gradient_tscore(
        self,
        semantic_tensions: List[float],
        gradient_tscores: List[float]
    ) -> Dict[str, float]:
        """
        Compute correlation between semantic tension and gradient T-Score.

        Hypothesis: Higher semantic tension -> Lower gradient T-Score
                    -> More C-S-P activation

        Args:
            semantic_tensions: List of pairwise tension values per sample.
            gradient_tscores: List of gradient T-Score values per sample.

        Returns:
            dict with correlation metrics.
        """
        from scipy import stats

        if len(semantic_tensions) != len(gradient_tscores):
            raise ValueError("Lists must be same length")

        if len(semantic_tensions) < 3:
            return {'error': 'Need at least 3 data points for correlation'}

        pearson_r, pearson_p = stats.pearsonr(semantic_tensions, gradient_tscores)
        spearman_r, spearman_p = stats.spearmanr(semantic_tensions, gradient_tscores)

        return {
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'spearman_r': float(spearman_r),
            'spearman_p': float(spearman_p),
            'n_samples': len(semantic_tensions),
            'hypothesis_supported': pearson_r < -0.2 and pearson_p < 0.05
        }
