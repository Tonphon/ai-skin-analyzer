# scripts/build_cf_bundle.py
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# --------- Config (edit if your paths differ) ----------
SALES_PATH = "data/sales_fact_skincare_user_features_encrypted_item_number.csv"
ITEM_MASTER_PATH = "data/item_master_with_skin_concern_cat.csv"
OUT_PATH = "artifacts/cf_bundle.pkl"

TRAIN_MONTHS = ["2025-07", "2025-08"]
VALID_CONCERNS = [1, 2, 3, 4, 5]
TOP_POP_PER_CONCERN = 200

CURRENT_YEAR_FOR_AGE = 2025
# ------------------------------------------------------


def assign_age_group(age: int) -> str:
    """Updated age groups: 0-17, 18-24, 25-34, 35-44, 45-54, 55+"""
    if age < 18:
        return "0-17"
    elif age < 25:
        return "18-24"
    elif age < 35:
        return "25-34"
    elif age < 45:
        return "35-44"
    elif age < 55:
        return "45-54"
    else:
        return "55+"


def normalize_gender(x):
    """Map Thai gender to simple M/F to match Streamlit options."""
    if pd.isna(x):
        return None
    x = str(x).strip()
    if x in ["หญิง", "female", "Female", "F"]:
        return "F"
    if x in ["ชาย", "male", "Male", "M"]:
        return "M"
    return x


def main():
    os.makedirs("artifacts", exist_ok=True)

    # ---- Load data ----
    sales = pd.read_csv(SALES_PATH)
    item_master = pd.read_csv(ITEM_MASTER_PATH)

    # sales_date -> month_id
    sales["sales_date"] = pd.to_datetime(sales["sales_date"], errors="coerce")
    sales["month_id"] = sales["sales_date"].dt.to_period("M").astype(str)

    # Merge concern category
    keep_cols_item = ["item_number", "skin_concern_cat_id", "skin_concern_cat_name"]
    item_master2 = item_master[keep_cols_item].drop_duplicates("item_number")

    df = sales.merge(item_master2, on="item_number", how="left")

    # Keep only valid concerns 1..5
    df = df[df["skin_concern_cat_id"].isin(VALID_CONCERNS)].copy()

    # Keep train months only
    train = df[df["month_id"].isin(TRAIN_MONTHS)].copy()

    # Basic cleanup
    train["gender"] = train["gender"].apply(normalize_gender)
    train["birth_year"] = pd.to_numeric(train["birth_year"], errors="coerce")
    train["sales_quantity"] = pd.to_numeric(train["sales_quantity"], errors="coerce").fillna(0).astype(float)

    # ---- Build user demographics ----
    demo = (
        train[["encrypted_user", "gender", "birth_year"]]
        .dropna(subset=["encrypted_user"])
        .drop_duplicates("encrypted_user")
        .copy()
    )
    demo["age"] = (CURRENT_YEAR_FOR_AGE - demo["birth_year"]).astype("Int64")
    demo["age_group"] = demo["age"].fillna(30).astype(int).apply(assign_age_group)

    user_demographics = demo[["encrypted_user", "gender", "age_group"]].copy()

    # ---- Primary concern from train history ----
    user_concern_qty = (
        train.groupby(["encrypted_user", "skin_concern_cat_id"])["sales_quantity"]
        .sum()
        .reset_index()
    )
    idx = user_concern_qty.groupby("encrypted_user")["sales_quantity"].idxmax()
    primary = user_concern_qty.loc[idx].rename(columns={"skin_concern_cat_id": "primary_concern"})
    user_demographics = user_demographics.merge(primary[["encrypted_user", "primary_concern"]],
                                                on="encrypted_user", how="left")

    # ---- Build user-item matrix ----
    user_item_matrix = (
        train.groupby(["encrypted_user", "item_number"])["sales_quantity"]
        .sum()
        .unstack(fill_value=0)
    )

    # ---- User-user cosine similarity ----
    sim = cosine_similarity(user_item_matrix.values)
    user_similarity_df = pd.DataFrame(sim, index=user_item_matrix.index, columns=user_item_matrix.index)

    # ---- Train purchases set ----
    train_user_purchases = (
        train.groupby("encrypted_user")["item_number"]
        .apply(lambda x: set(map(int, x.tolist())))
        .to_dict()
    )

    # ---- Popular-by-concern fallback (NEW: store tuples with quantities) ----
    pop = (
        train.groupby(["skin_concern_cat_id", "item_number"])["sales_quantity"]
        .sum()
        .reset_index()
        .sort_values(["skin_concern_cat_id", "sales_quantity"], ascending=[True, False])
    )

    popular_by_concern = {}
    for cid in VALID_CONCERNS:
        concern_items = pop[pop["skin_concern_cat_id"] == cid].head(TOP_POP_PER_CONCERN)
        # Store as list of (item_number, sales_quantity) tuples
        popular_by_concern[cid] = [
            (int(row["item_number"]), float(row["sales_quantity"]))
            for _, row in concern_items.iterrows()
        ]

    # ---- Item meta ----
    item_meta = item_master.copy()

    bundle = {
        "user_item_matrix": user_item_matrix,
        "user_similarity_df": user_similarity_df,
        "user_demographics": user_demographics,
        "train_user_purchases": train_user_purchases,
        "item_meta": item_meta,
        "popular_by_concern": popular_by_concern,
    }

    with open(OUT_PATH, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("✅ Saved:", OUT_PATH)
    print("Users:", user_item_matrix.shape[0], "Items:", user_item_matrix.shape[1])
    print("Train rows:", len(train))
    
    # Print sample popular items for verification
    print("\nSample popular items by concern:")
    for cid in VALID_CONCERNS[:2]:  # Show first 2 concerns
        if cid in popular_by_concern:
            print(f"  Concern {cid}: {len(popular_by_concern[cid])} items, "
                  f"top item qty = {popular_by_concern[cid][0][1] if popular_by_concern[cid] else 0:.1f}")


if __name__ == "__main__":
    main()