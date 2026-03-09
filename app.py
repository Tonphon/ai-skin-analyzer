import streamlit as st
import pandas as pd
from PIL import Image

from src.classifier import SkinConcernClassifier
from src.recommender import Recommender
from src.config import CONCERN_ID_TO_NAME

st.set_page_config(page_title="Skin Analyzer + Recommender", layout="wide")


@st.cache_resource
def load_classifier():
    return SkinConcernClassifier("models/best_model.pth", device="cpu", arch="efficientnet_v2_s")


@st.cache_resource
def load_recommender():
    return Recommender("artifacts/cf_bundle.pkl")


@st.cache_data
def load_item_meta():
    return pd.read_csv("data/item_master_with_skin_concern_cat.csv")


def _format_recs(recs, item_meta: pd.DataFrame) -> pd.DataFrame:
    """Convert list of RecItem to a display-ready DataFrame."""
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame([{"item_number": r.item_number, "score": round(r.score, 4)} for r in recs])
    df = df.merge(item_meta[["item_number", "item_desc", "skin_concern_cat_name"]], on="item_number", how="left")
    return df[["item_number", "item_desc", "skin_concern_cat_name", "score"]]


# ── Load resources ────────────────────────────────────────────────────────────
st.title("Skin Concern Classification + Product Recommendation")

clf       = load_classifier()
rec       = load_recommender()
item_meta = load_item_meta()

with st.expander("Model target classes (from models/labels.json)", expanded=False):
    st.write(clf.class_names)

# ── Sidebar: user info ────────────────────────────────────────────────────────
st.sidebar.header("User Info")

if "user_mode" not in st.session_state:
    st.session_state.user_mode = "New user (no history)"

mode = st.sidebar.radio(
    "User type",
    ["Existing user (has userID)", "New user (no history)"],
    key="user_mode",
)

user_id    = None
gender     = None
birth_year = None

if mode == "Existing user (has userID)":
    available_users = rec.get_all_users()
    if available_users:
        user_id  = st.sidebar.selectbox("Select User ID", options=available_users)
        user_demo = rec.get_user_demographics(user_id)
        if user_demo:
            st.sidebar.info(f"**Gender:** {user_demo['gender']}")
            st.sidebar.info(f"**Age Group:** {user_demo['age_group']}")
            gender = user_demo["gender"]
            age_group = user_demo["age_group"]
            birth_year = {
                "18-24": 2025 - 21,
                "25-34": 2025 - 30,
                "35-44": 2025 - 40,
                "45-54": 2025 - 50,
            }.get(age_group, 2025 - 60)
        else:
            st.sidebar.warning("User demographics not found.")
            gender, birth_year = "F", 2003
    else:
        st.sidebar.warning("No existing users found in the system.")

else:  # New user
    st.sidebar.subheader("Enter your details")
    gender     = st.sidebar.selectbox("Gender", ["F", "M"])
    birth_year = st.sidebar.number_input("Birth year", min_value=1940, max_value=2026, value=2003, step=1)

# Common settings
allow_repeats     = st.sidebar.toggle("Allow repeat recommendations", value=True)
top_k             = st.sidebar.slider("Top-K recommendations", min_value=1, max_value=20, value=10)
repurchase_boost  = st.sidebar.slider("Repurchase boost (existing users)", min_value=1.0, max_value=2.0, value=1.2, step=0.1)

# ── Main: image input ─────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1) Input")
    cam = st.camera_input("Capture face (optional)")
    up  = st.file_uploader("Or upload a photo", type=["jpg", "jpeg", "png"])

    img = None
    if cam is not None:
        img = Image.open(cam)
    elif up is not None:
        img = Image.open(up)

    if img:
        st.image(img, caption="Input image", width="stretch")

analyze_btn = st.button("Analyze → Recommend", type="primary", disabled=(img is None))

# ── Output ────────────────────────────────────────────────────────────────────
with col2:
    st.subheader("2) Output")

    if analyze_btn and img is not None:

        # ── Classify ─────────────────────────────────────────────────────────
        pred = clf.predict(img)

        st.markdown("### Classification scores")
        score_df = (
            pd.DataFrame(pred.label_scores.items(), columns=["label", "score"])
            .sort_values("score", ascending=False)
        )
        st.dataframe(score_df, width='stretch')

        predicted_concerns = pred.concern_ids or []

        if not predicted_concerns:
            st.warning(
                "No concerns mapped from model output yet "
                "(LABEL_TO_CONCERN_ID in src/config.py doesn't match your labels). "
                "You can still pick concern IDs manually below."
            )

        st.markdown("### Detected skin concerns")
        if predicted_concerns:
            st.write([f"{cid} – {CONCERN_ID_TO_NAME.get(cid, str(cid))}" for cid in predicted_concerns])
        else:
            st.write("None detected")

        st.markdown("### (Optional) Adjust concerns")
        all_cids = sorted(CONCERN_ID_TO_NAME.keys())
        chosen = st.multiselect(
            "Concern IDs used for recommendation:",
            options=all_cids,
            default=predicted_concerns,
            format_func=lambda x: f"{x} – {CONCERN_ID_TO_NAME.get(x, str(x))}",
        )
        chosen = sorted(set(chosen))

        if not chosen:
            st.info("Pick at least 1 concern ID to generate recommendations.")
            st.stop()

        # ── Recommendations ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 3) Recommendations")

        # ── Section 1: Concern-based ──────────────────────────────────────────
        st.markdown("### Recommend for your skin concern")
        st.caption("Products matching your analyzed skin concerns")

        if mode == "New user (no history)":
            # E2-A cohort CF, filtered to image concerns
            recs_concern = rec.recommend_new_user(
                gender=gender,
                birth_year=int(birth_year),
                selected_concern_ids=chosen,
                top_k=top_k,
            )
        else:
            # Existing user: E3-style CF, filtered to image concerns
            recs_concern = rec.recommend_by_concern(
                user_id=user_id,
                selected_concern_ids=chosen,
                top_k=top_k,
                allow_repeats=allow_repeats,
                repurchase_boost=repurchase_boost,
            )

        if not recs_concern:
            st.info("No recommendations found. Try adjusting concerns or relaxing filters.")
        else:
            st.dataframe(_format_recs(recs_concern, item_meta), width='stretch')

        st.markdown("---")

        # ── Section 2: User-similarity based ─────────────────────────────────
        st.markdown("### Users similar to you liked these products")
        st.caption("Based on purchase patterns of similar users")

        if mode == "New user (no history)":
            st.info("This section is available once you have purchase history.")
        else:
            if user_id is None:
                st.warning("User ID not available.")
            else:
                # E3: all items, user's own similarity row, repeat boost
                recs_similar = rec.recommend_by_user_similarity(
                    user_id=user_id,
                    top_k=top_k,
                    allow_repeats=allow_repeats,
                    repurchase_boost=repurchase_boost,
                )

                if not recs_similar:
                    st.info("No similar users found or no recommendations available.")
                else:
                    st.dataframe(_format_recs(recs_similar, item_meta), width='stretch')
