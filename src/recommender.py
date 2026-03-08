# src/recommender.py
from dataclasses import dataclass
from typing import List, Dict, Optional
from collections import defaultdict
import pickle
import numpy as np
import pandas as pd


@dataclass
class RecItem:
    item_number: int
    score: float


def _assign_age_group(age: int) -> str:
    if age < 18:   return "0-17"
    elif age < 25: return "18-24"
    elif age < 35: return "25-34"
    elif age < 45: return "35-44"
    elif age < 55: return "45-54"
    else:          return "55+"


class Recommender:
    def __init__(self, bundle_path: str):
        with open(bundle_path, "rb") as f:
            bundle = pickle.load(f)

        self.user_item_matrix    = bundle["user_item_matrix"]
        self.user_similarity_df  = bundle["user_similarity_df"]
        self.user_demographics   = bundle["user_demographics"]
        self.train_user_purchases = bundle["train_user_purchases"]
        self.item_meta           = bundle["item_meta"]
        self.popular_by_concern  = bundle["popular_by_concern"]

        # item -> concern id lookup
        self.item_to_concern = dict(
            zip(self.item_meta["item_number"], self.item_meta["skin_concern_cat_id"])
        )

    # ── Public helpers ────────────────────────────────────────────────────────

    def get_all_users(self) -> List[str]:
        return sorted(self.user_demographics["encrypted_user"].astype(str).tolist())

    def get_user_demographics(self, user_id: str) -> Optional[Dict]:
        row = self.user_demographics[
            self.user_demographics["encrypted_user"].astype(str) == str(user_id)
        ]
        if len(row) == 0:
            return None
        return {
            "gender":    row["gender"].iloc[0],
            "age_group": row["age_group"].iloc[0],
        }

    # ── New user: E2-A cohort CF ──────────────────────────────────────────────

    def recommend_new_user(
        self,
        gender: str,
        birth_year: int,
        selected_concern_ids: List[int],
        top_k: int,
        current_year: int = 2025,
    ) -> List[RecItem]:
        """
        E2-A cold-start cohort CF for a brand-new user.

        1. Find all training users matching gender + age_group.
        2. Build pairwise cosine similarity among those neighbours only;
           weight each neighbour by their avg similarity to the rest of the
           cohort (self-similarity zeroed out).
        3. Score items: sum(weight × quantity) across all neighbours.
        4. Filter results to selected_concern_ids from the image.
        5. Exclude items the cohort neighbours already purchased in training
           (already_purchased is empty for a genuinely new user, so nothing
           is excluded from candidates — consistent with E2-A in evaluation).
        """
        age_group = _assign_age_group(current_year - int(birth_year))

        # ── 1. Demographic cohort ─────────────────────────────────────────────
        df = self.user_demographics
        mask = (df["gender"] == gender) & (df["age_group"] == age_group)
        candidates = df[mask]["encrypted_user"].astype(str).tolist()
        candidates = [
            u for u in candidates
            if u in self.user_item_matrix.index
            and u in self.user_similarity_df.index
        ]

        if not candidates:
            return []

        # ── 2. Cohort representativeness weights ──────────────────────────────
        group_sim = self.user_similarity_df.loc[candidates, candidates].copy()
        np.fill_diagonal(group_sim.values, 0)
        avg_sim = group_sim.mean(axis=1)   # Series: user -> weight

        # ── 3. Score items ────────────────────────────────────────────────────
        selected_concern_ids = set(selected_concern_ids)
        item_scores: dict = defaultdict(float)

        for u in candidates:
            weight = avg_sim[u]
            for item, qty in self.user_item_matrix.loc[u].items():
                if qty <= 0:
                    continue
                # ── 4. Filter by image concerns ───────────────────────────────
                if self.item_to_concern.get(item) not in selected_concern_ids:
                    continue
                item_scores[item] += weight * float(qty)

        ranked = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [RecItem(item_number=int(i), score=float(s)) for i, s in ranked]

    # ── Existing user: concern-based (E3 style) ───────────────────────────────

    def recommend_by_concern(
        self,
        user_id: str,
        selected_concern_ids: List[int],
        top_k: int,
        allow_repeats: bool,
        top_neighbors: int = 50,
        min_sim: float = 0.05,
        repurchase_boost: float = 1.2,
    ) -> List[RecItem]:
        """
        Existing user, concern-based (image concerns as filter).

        Uses the user's own similarity row (E3 style):
        - Demographic neighbours ranked by cosine similarity to this user.
        - Items filtered to selected_concern_ids.
        - allow_repeats=True  → previously bought items included + boost applied.
        - allow_repeats=False → previously bought items excluded, no boost.
        """
        selected_concern_ids = set(selected_concern_ids)
        if not selected_concern_ids:
            return []

        if user_id not in self.user_similarity_df.index:
            return []

        demo = self.get_user_demographics(user_id)
        if demo is None:
            return []

        candidates = self._get_candidate_users(demo["gender"], demo["age_group"])
        candidates = [
            u for u in candidates
            if u != user_id and u in self.user_similarity_df.columns
        ]
        if not candidates:
            return []

        sims = (
            self.user_similarity_df.loc[user_id, candidates]
            .sort_values(ascending=False)
        )
        sims = sims[sims >= min_sim].head(top_neighbors)
        if sims.empty:
            return []

        already = set() if allow_repeats else self.train_user_purchases.get(user_id, set())
        item_scores: dict = defaultdict(float)

        for sim_user, w in sims.items():
            if sim_user not in self.user_item_matrix.index:
                continue
            for item, qty in self.user_item_matrix.loc[sim_user].items():
                if qty <= 0:
                    continue
                if self.item_to_concern.get(item) not in selected_concern_ids:
                    continue
                if item in already:
                    continue
                item_scores[item] += float(w) * float(qty)

        # Repurchase boost (E3)
        if allow_repeats:
            for item in self.train_user_purchases.get(user_id, set()):
                if item in item_scores:
                    item_scores[item] *= repurchase_boost

        ranked = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [RecItem(item_number=int(i), score=float(s)) for i, s in ranked]

    # ── Existing user: pure similarity (E3, no concern filter) ───────────────

    def recommend_by_user_similarity(
        self,
        user_id: str,
        top_k: int,
        allow_repeats: bool,
        top_neighbors: int = 50,
        min_sim: float = 0.05,
        repurchase_boost: float = 1.2,
    ) -> List[RecItem]:
        """
        Existing user, pure purchase-pattern similarity (E3, no concern filter).

        All items from similar users — no concern filtering.
        allow_repeats + boost mirrors E3 exactly.
        """
        if user_id not in self.user_similarity_df.index:
            return []

        demo = self.get_user_demographics(user_id)
        if demo is None:
            return []

        candidates = self._get_candidate_users(demo["gender"], demo["age_group"])
        candidates = [
            u for u in candidates
            if u != user_id and u in self.user_similarity_df.columns
        ]
        if not candidates:
            return []

        sims = (
            self.user_similarity_df.loc[user_id, candidates]
            .sort_values(ascending=False)
        )
        sims = sims[sims >= min_sim].head(top_neighbors)
        if sims.empty:
            return []

        already = set() if allow_repeats else self.train_user_purchases.get(user_id, set())
        item_scores: dict = defaultdict(float)

        for sim_user, w in sims.items():
            if sim_user not in self.user_item_matrix.index:
                continue
            for item, qty in self.user_item_matrix.loc[sim_user].items():
                if qty <= 0:
                    continue
                if item in already:
                    continue
                item_scores[item] += float(w) * float(qty)

        # Repurchase boost (E3)
        if allow_repeats:
            for item in self.train_user_purchases.get(user_id, set()):
                if item in item_scores:
                    item_scores[item] *= repurchase_boost

        ranked = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [RecItem(item_number=int(i), score=float(s)) for i, s in ranked]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_candidate_users(self, gender: str, age_group: str) -> List[str]:
        df = self.user_demographics
        mask = (df["gender"] == gender) & (df["age_group"] == age_group)
        return df[mask]["encrypted_user"].astype(str).tolist()
