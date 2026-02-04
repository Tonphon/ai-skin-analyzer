import streamlit as st
import pandas as pd
from PIL import Image

from src.classifier import SkinConcernClassifier
from src.recommender import Recommender
from src.config import CONCERN_ID_TO_NAME

st.set_page_config(page_title="Skin Analyzer + Recommender", layout="wide")

@st.cache_resource
def load_classifier():
    return SkinConcernClassifier("models/best_model.pth", device="cpu")

@st.cache_resource
def load_recommender():
    return Recommender("artifacts/cf_bundle.pkl")

@st.cache_data
def load_item_meta():
    return pd.read_csv("data/item_master_with_skin_concern_cat.csv")

st.title("Skin Concern Classification + Product Recommendation (Demo)")

clf = load_classifier()
rec = load_recommender()
item_meta = load_item_meta()

# ---------- Sidebar: user info ----------
st.sidebar.header("User Info")
mode = st.sidebar.radio("User type", ["Existing user (has userID)", "New user (no history)"])

user_id = None
gender = st.sidebar.selectbox("Gender", ["F", "M"], index=0)
birth_year = st.sidebar.number_input("Birth year", min_value=1940, max_value=2026, value=2003, step=1)

allow_repeats = st.sidebar.toggle("Allow repeat recommendations", value=True)
top_k = st.sidebar.slider("Top-K", min_value=1, max_value=20, value=10)

if mode.startswith("Existing"):
    user_id = st.sidebar.text_input("User ID (encrypted_user)")

# ---------- Main: image input ----------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1) Input")
    cam = st.camera_input("Capture face (optional)")
    up = st.file_uploader("Or upload a photo", type=["jpg", "jpeg", "png"])

    img = None
    if cam is not None:
        img = Image.open(cam)
    elif up is not None:
        img = Image.open(up)

    if img:
        st.image(img, caption="Input image", use_container_width=True)

analyze_btn = st.button("Analyze → Recommend", type="primary", disabled=(img is None))

with col2:
    st.subheader("2) Output")

    if analyze_btn and img is not None:
        pred = clf.predict(img)

        # Show label scores
        st.markdown("### Classification scores")
        score_df = (
            pd.DataFrame(pred.label_scores.items(), columns=["label", "score"])
            .sort_values("score", ascending=False)
        )
        st.dataframe(score_df, use_container_width=True)

        # predicted concern IDs
        predicted_concerns = pred.concern_ids
        if not predicted_concerns:
            st.warning("No concerns mapped from model output. Check LABEL_TO_CONCERN_ID in config.py.")
            st.stop()

        st.markdown("### Selected skin concerns (from image)")
        st.write([f"{cid} - {CONCERN_ID_TO_NAME.get(cid, str(cid))}" for cid in predicted_concerns])

        # allow user to override / add concerns
        st.markdown("### (Optional) Adjust concerns")
        all_cids = sorted(CONCERN_ID_TO_NAME.keys())
        chosen = st.multiselect(
            "Use these concern IDs for recommendation:",
            options=all_cids,
            default=predicted_concerns
        )
        chosen = sorted(list(set(chosen)))

        # Recommend
        recs = rec.recommend(
            selected_concern_ids=chosen,
            top_k=top_k,
            allow_repeats=allow_repeats,
            user_id=user_id if user_id else None,
            gender=gender,
            birth_year=int(birth_year),
        )

        st.markdown("### 3) Recommendations")

        if not recs:
            st.info("No recommendations found. (Cold-start or strict filters) Try adding concerns or disabling demographic filter in code.")
            st.stop()

        # Join with item meta for display
        rec_df = pd.DataFrame([{"item_number": r.item_number, "score": r.score, "reason": r.reason} for r in recs])
        show = rec_df.merge(item_meta, on="item_number", how="left")

        # Display as simple table
        st.dataframe(
            show[["item_number", "skin_concern_cat_name", "score", "reason"] + [c for c in show.columns if c not in ["item_number","skin_concern_cat_name","score","reason"]][:0]],
            use_container_width=True
        )

        # Quick export
        st.download_button(
            "Download results (CSV)",
            data=show.to_csv(index=False).encode("utf-8"),
            file_name="recommendations.csv",
            mime="text/csv"
        )
