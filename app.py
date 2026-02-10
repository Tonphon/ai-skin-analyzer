import streamlit as st
import pandas as pd
from PIL import Image

from src.classifier import SkinConcernClassifier
from src.recommender import Recommender
from src.config import CONCERN_ID_TO_NAME

st.set_page_config(page_title="Skin Analyzer + Recommender", layout="wide")

@st.cache_resource
def load_classifier():
    # classifier will read labels from models/labels.json automatically
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

# Show model labels (from labels.json)
with st.expander("Model target classes (from models/labels.json)", expanded=False):
    st.write(clf.class_names)

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
        st.image(img, caption="Input image", width='stretch')

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
        st.dataframe(score_df, width='stretch')

        # predicted concern IDs (may be empty until you update LABEL_TO_CONCERN_ID)
        predicted_concerns = pred.concern_ids or []

        if not predicted_concerns:
            st.warning(
                "No concerns mapped from model output yet (LABEL_TO_CONCERN_ID in src/config.py doesn't match your new labels). "
                "You can still manually pick concern IDs below."
            )

        st.markdown("### Selected skin concerns (from image)")
        if predicted_concerns:
            st.write([f"{cid} - {CONCERN_ID_TO_NAME.get(cid, str(cid))}" for cid in predicted_concerns])
        else:
            st.write("[]")

        # allow user to override / add concerns
        st.markdown("### (Optional) Adjust concerns")
        all_cids = sorted(CONCERN_ID_TO_NAME.keys())
        chosen = st.multiselect(
            "Use these concern IDs for recommendation:",
            options=all_cids,
            default=predicted_concerns
        )
        chosen = sorted(list(set(chosen)))

        if not chosen:
            st.info("Pick at least 1 concern ID to generate recommendations.")
            st.stop()

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
            st.info("No recommendations found. Try adding concerns or relaxing filters.")
            st.stop()

        # Join with item meta for display
        rec_df = pd.DataFrame([{"item_number": r.item_number, "score": r.score, "reason": r.reason} for r in recs])
        show = rec_df.merge(item_meta, on="item_number", how="left")

        st.dataframe(
            show[["item_number", "skin_concern_cat_name", "score", "reason"]],
            width='stretch'
        )

        st.download_button(
            "Download results (CSV)",
            data=show.to_csv(index=False).encode("utf-8"),
            file_name="recommendations.csv",
            mime="text/csv"
        )
