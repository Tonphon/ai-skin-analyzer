# src/recommender.py
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union
from collections import defaultdict
import pickle
import numpy as np
import pandas as pd

@dataclass
class RecItem:
    item_number: int
    score: float
    reason: str

def _assign_age_group(age: int) -> str:
    if age < 18: return "0-17"
    elif age < 25: return "18-24"
    elif age < 35: return "25-34"
    elif age < 45: return "35-44"
    elif age < 55: return "45-54"
    else: return "55+"

class Recommender:
    def __init__(self, bundle_path: str):
        with open(bundle_path, "rb") as f:
            bundle = pickle.load(f)

        self.user_item_matrix = bundle["user_item_matrix"]
        self.user_similarity_df = bundle["user_similarity_df"]
        self.user_demographics = bundle["user_demographics"]
        self.train_user_purchases = bundle["train_user_purchases"]
        self.item_meta = bundle["item_meta"]
        self.popular_by_concern = bundle["popular_by_concern"]

        # quick map item -> concern id
        self.item_to_concern = dict(zip(self.item_meta["item_number"], self.item_meta["skin_concern_cat_id"]))

    def get_all_users(self) -> List[str]:
        """Return list of all user IDs in the system."""
        return sorted(self.user_demographics["encrypted_user"].astype(str).tolist())

    def get_user_demographics(self, user_id: str) -> Optional[Dict]:
        """Fetch demographics for a specific user."""
        row = self.user_demographics[self.user_demographics["encrypted_user"].astype(str) == str(user_id)]
        if len(row) == 0:
            return None
        return {
            "gender": row["gender"].iloc[0],
            "age_group": row["age_group"].iloc[0],
        }

    def _get_candidate_users(self, gender: str, age_group: str) -> List[str]:
        """Get users matching gender and age_group."""
        df = self.user_demographics
        df2 = df[(df["gender"] == gender) & (df["age_group"] == age_group)]
        return df2["encrypted_user"].astype(str).tolist()

    def recommend_by_concern(
        self,
        selected_concern_ids: List[int],
        top_k: int,
        allow_repeats: bool,
        user_id: Optional[str] = None,
        gender: Optional[str] = None,
        birth_year: Optional[int] = None,
        current_year: int = 2025,
        top_neighbors: int = 50,
        min_sim: float = 0.05,
        repurchase_boost: float = 1.2,
    ) -> List[RecItem]:
        """
        B1: Concern-based CF recommendation.
        Uses concern IDs from image analysis + demographic matching.
        """
        selected_concern_ids = sorted(list(set(selected_concern_ids)))
        if not selected_concern_ids:
            return []

        # If existing user with history → CF
        if user_id is not None and user_id in self.user_similarity_df.index:
            # Get demo group
            if gender is None or birth_year is None:
                row = self.user_demographics[self.user_demographics["encrypted_user"].astype(str) == str(user_id)]
                if len(row):
                    gender = row["gender"].iloc[0]
                    age_group = row["age_group"].iloc[0]
                else:
                    age_group = "25-34"
            else:
                age = current_year - int(birth_year)
                age_group = _assign_age_group(age)

            candidates = self._get_candidate_users(gender, age_group)
            candidates = [u for u in candidates if u != user_id and u in self.user_similarity_df.columns]

            if not candidates:
                return self._fallback_popular(selected_concern_ids, top_k)

            sims = self.user_similarity_df.loc[user_id, candidates].sort_values(ascending=False)
            sims = sims[sims >= min_sim].head(top_neighbors)

            if sims.empty:
                return self._fallback_popular(selected_concern_ids, top_k)

            already = set() if allow_repeats else self.train_user_purchases.get(user_id, set())
            item_scores = defaultdict(float)

            for sim_user, w in sims.items():
                if sim_user not in self.user_item_matrix.index:
                    continue
                vec = self.user_item_matrix.loc[sim_user]
                for item, qty in vec.items():
                    if qty <= 0:
                        continue
                    # Filter by concern
                    cid = self.item_to_concern.get(item)
                    if cid not in selected_concern_ids:
                        continue
                    if item in already:
                        continue
                    item_scores[item] += float(w) * float(qty)

            # Repurchase boost
            if allow_repeats and user_id in self.train_user_purchases:
                own = self.train_user_purchases[user_id]
                for item in list(own):
                    if item in item_scores:
                        item_scores[item] *= repurchase_boost

            ranked = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            return [
                RecItem(item_number=int(i), score=float(s), reason="CF: similar users with same concerns")
                for i, s in ranked
            ]

        # New user or no history → fallback
        return self._fallback_popular(selected_concern_ids, top_k)

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
        B2: User-user similarity recommendation.
        Ignores skin concerns; purely based on purchase patterns.
        Only for existing users.
        """
        if user_id not in self.user_similarity_df.index:
            return []

        # Get user demographics
        demo = self.get_user_demographics(user_id)
        if demo is None:
            return []

        gender = demo["gender"]
        age_group = demo["age_group"]

        # Get candidate users (same gender + age group)
        candidates = self._get_candidate_users(gender, age_group)
        candidates = [u for u in candidates if u != user_id and u in self.user_similarity_df.columns]

        if not candidates:
            return []

        # Get similarity scores
        sims = self.user_similarity_df.loc[user_id, candidates].sort_values(ascending=False)
        sims = sims[sims >= min_sim].head(top_neighbors)

        if sims.empty:
            return []

        already = set() if allow_repeats else self.train_user_purchases.get(user_id, set())
        item_scores = defaultdict(float)

        # Aggregate items from similar users
        for sim_user, w in sims.items():
            if sim_user not in self.user_item_matrix.index:
                continue
            vec = self.user_item_matrix.loc[sim_user]
            for item, qty in vec.items():
                if qty <= 0:
                    continue
                if item in already:
                    continue
                # No concern filtering here
                item_scores[item] += float(w) * float(qty)

        # Repurchase boost
        if allow_repeats and user_id in self.train_user_purchases:
            own = self.train_user_purchases[user_id]
            for item in list(own):
                if item in item_scores:
                    item_scores[item] *= repurchase_boost

        ranked = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            RecItem(item_number=int(i), score=float(s), reason="Similar users also purchased")
            for i, s in ranked
        ]

    def _fallback_popular(self, concern_ids: List[int], top_k: int) -> List[RecItem]:
        """
        Fallback: interleave popular items across concerns with normalized scores.
        Supports both new format [(item, qty), ...] and old format [item, ...].
        """
        # Build pools with scores for each concern
        pools_with_scores = []
        
        for cid in concern_ids:
            items_data = self.popular_by_concern.get(cid, [])
            if not items_data:
                pools_with_scores.append([])
                continue
            
            # Check format: tuple (item, qty) or plain item number
            first_item = items_data[0]
            if isinstance(first_item, (tuple, list)) and len(first_item) == 2:
                # NEW FORMAT: [(item_number, qty), ...]
                items = [int(item) for item, qty in items_data]
                quantities = [float(qty) for item, qty in items_data]
                
                # Normalize scores: score = qty / max_qty in this concern
                max_qty = max(quantities) if quantities else 1.0
                scores = [qty / max_qty for qty in quantities]
                
                pool = [
                    (items[i], scores[i], f"Popular in concern {cid} (qty={quantities[i]:.0f})")
                    for i in range(len(items))
                ]
            else:
                # OLD FORMAT (backward compatibility): [item_number, ...]
                items = [int(item) for item in items_data]
                # Use rank-based score: 1.0 for rank 1, decreasing linearly
                scores = [1.0 - (i / len(items)) for i in range(len(items))]
                pool = [
                    (items[i], scores[i], f"Popular in concern {cid} (rank {i+1})")
                    for i in range(len(items))
                ]
            
            pools_with_scores.append(pool)
        
        # Interleave across concerns
        out = []
        seen = set()
        ptrs = [0] * len(pools_with_scores)
        
        while len(out) < top_k and any(ptrs[i] < len(pools_with_scores[i]) for i in range(len(pools_with_scores))):
            for i in range(len(pools_with_scores)):
                if ptrs[i] >= len(pools_with_scores[i]):
                    continue
                
                item, score, reason = pools_with_scores[i][ptrs[i]]
                ptrs[i] += 1
                
                if item in seen:
                    continue
                seen.add(item)
                
                out.append(RecItem(item_number=item, score=score, reason=reason))
                
                if len(out) >= top_k:
                    break
        
        return out