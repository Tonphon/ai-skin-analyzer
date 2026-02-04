# src/recommender.py
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
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
    if age < 25: return "18-24"
    elif age < 35: return "25-34"
    elif age < 45: return "35-44"
    elif age < 55: return "45-54"
    else: return "55+"

class Recommender:
    def __init__(self, bundle_path: str):
        with open(bundle_path, "rb") as f:
            bundle = pickle.load(f)

        # expected keys inside bundle
        self.user_item_matrix = bundle["user_item_matrix"]     # pd.DataFrame (users x items)
        self.user_similarity_df = bundle["user_similarity_df"] # pd.DataFrame (users x users)
        self.user_demographics = bundle["user_demographics"]   # pd.DataFrame: encrypted_user, gender, age_group, (optional) primary concern
        self.train_user_purchases = bundle["train_user_purchases"]  # dict user -> set(items)
        self.item_meta = bundle["item_meta"]                   # pd.DataFrame with item_number, skin_concern_cat_id, name/brand(optional)
        self.popular_by_concern = bundle["popular_by_concern"] # dict concern_id -> list[item_number]

        # quick map item -> concern id
        self.item_to_concern = dict(zip(self.item_meta["item_number"], self.item_meta["skin_concern_cat_id"]))

    def _get_candidate_users(self, gender: str, age_group: str) -> List[str]:
        df = self.user_demographics
        df2 = df[(df["gender"] == gender) & (df["age_group"] == age_group)]
        return df2["encrypted_user"].astype(str).tolist()

    def recommend(
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
    ) -> List[RecItem]:

        selected_concern_ids = sorted(list(set(selected_concern_ids)))
        if not selected_concern_ids:
            return []

        # If we have an existing user with history -> CF
        if user_id is not None and user_id in self.user_similarity_df.index:
            # determine demo group (gender/age_group)
            if gender is None or birth_year is None:
                # try fetch from demographics table
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
                # aggregate items from similar users
                if sim_user not in self.user_item_matrix.index:
                    continue
                vec = self.user_item_matrix.loc[sim_user]
                for item, qty in vec.items():
                    if qty <= 0:
                        continue
                    # concern filter comes from IMAGE prediction
                    cid = self.item_to_concern.get(item)
                    if cid not in selected_concern_ids:
                        continue
                    if item in already:
                        continue
                    item_scores[item] += float(w) * float(qty)

            # optional repurchase boost
            if allow_repeats and user_id in self.train_user_purchases:
                own = self.train_user_purchases[user_id]
                for item in list(own):
                    if item in item_scores:
                        item_scores[item] *= 1.2

            ranked = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            return [
                RecItem(item_number=int(i), score=float(s), reason="CF: similar users bought it")
                for i, s in ranked
            ]

        # Otherwise -> cold-start fallback
        return self._fallback_popular(selected_concern_ids, top_k)

    def _fallback_popular(self, concern_ids: List[int], top_k: int) -> List[RecItem]:
        # interleave popular lists across concerns so multi-concern results feel balanced
        pools = [self.popular_by_concern.get(cid, []) for cid in concern_ids]
        out = []
        seen = set()
        ptrs = [0] * len(pools)

        while len(out) < top_k and any(ptrs[i] < len(pools[i]) for i in range(len(pools))):
            for i in range(len(pools)):
                if ptrs[i] >= len(pools[i]): 
                    continue
                item = pools[i][ptrs[i]]
                ptrs[i] += 1
                if item in seen:
                    continue
                seen.add(item)
                out.append(RecItem(item_number=int(item), score=0.0, reason=f"Fallback: popular in concern {concern_ids[i]}"))
                if len(out) >= top_k:
                    break

        return out
