import streamlit as st
import pandas as pd
from PIL import Image

from src.classifier import SkinConcernClassifier
from src.recommender import Recommender
from src.config import CONCERN_ID_TO_NAME

st.set_page_config(page_title="Skin Analyzer + Recommender", layout="wide")

@st.cache_resource
def load_classifier():
    return SkinConcernClassifier("models/convnext_224.pth", device="cpu", arch="convnextv2_tiny")

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

# Initialize session state for mode if not exists
if "user_mode" not in st.session_state:
    st.session_state.user_mode = "New user (no history)"

# Mode selection
mode = st.sidebar.radio(
    "User type",
    ["Existing user (has userID)", "New user (no history)"],
    key="user_mode"
)

user_id = None
gender = None
birth_year = None

if mode == "Existing user (has userID)":
    # Get list of available users
    available_users = rec.get_all_users()
    
    if available_users:
        user_id = st.sidebar.selectbox("Select User ID", options=available_users)
        
        # Fetch user demographics from stored data
        user_demo = rec.get_user_demographics(user_id)
        
        if user_demo is not None:
            st.sidebar.info(f"**Gender:** {user_demo['gender']}")
            st.sidebar.info(f"**Age Group:** {user_demo['age_group']}")
            gender = user_demo['gender']
            # Derive approximate birth_year from age_group (use midpoint for calculations)
            age_group = user_demo['age_group']
            if age_group == "18-24":
                birth_year = 2025 - 21
            elif age_group == "25-34":
                birth_year = 2025 - 30
            elif age_group == "35-44":
                birth_year = 2025 - 40
            elif age_group == "45-54":
                birth_year = 2025 - 50
            else:  # 55+
                birth_year = 2025 - 60
        else:
            st.sidebar.warning("User demographics not found. Using defaults.")
            gender = "F"
            birth_year = 2003
    else:
        st.sidebar.warning("No existing users found in the system.")
        user_id = None

else:  # New user
    st.sidebar.subheader("Enter your details")
    gender = st.sidebar.selectbox("Gender", ["F", "M"], index=0)
    birth_year = st.sidebar.number_input("Birth year", min_value=1940, max_value=2026, value=2003, step=1)

# Common settings
allow_repeats = st.sidebar.toggle("Allow repeat recommendations", value=True)
top_k = st.sidebar.slider("Top-K recommendations", min_value=1, max_value=20, value=10)
repurchase_boost = st.sidebar.slider("Repurchase boost (for existing users)", min_value=1.0, max_value=2.0, value=1.2, step=0.1)

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
            default=predicted_concerns,
            format_func=lambda x: f"{x} - {CONCERN_ID_TO_NAME.get(x, str(x))}"
        )
        chosen = sorted(list(set(chosen)))

        if not chosen:
            st.info("Pick at least 1 concern ID to generate recommendations.")
            st.stop()

        # ---------- Recommendations: 2 sections ----------
        st.markdown("---")
        st.markdown("## 3) Recommendations")

        # === B1: Concern-based CF ===
        st.markdown("### Recommend for your skin concern")
        st.caption("Products matching your analyzed skin concerns")
        
        recs_concern = rec.recommend_by_concern(
            selected_concern_ids=chosen,
            top_k=top_k,
            allow_repeats=allow_repeats,
            user_id=user_id if user_id else None,
            gender=gender,
            birth_year=int(birth_year),
            repurchase_boost=repurchase_boost,
        )

        if not recs_concern:
            st.info("No recommendations found. Try adding concerns or relaxing filters.")
        else:
            rec_df = pd.DataFrame([
                {"item_number": r.item_number, "score": r.score, "reason": r.reason} 
                for r in recs_concern
            ])
            show = rec_df.merge(item_meta, on="item_number", how="left")
            st.dataframe(
                show[["item_number", "skin_concern_cat_name", "score", "reason"]],
                use_container_width=True
            )
            st.download_button(
                "Download concern-based results (CSV)",
                data=show.to_csv(index=False).encode("utf-8"),
                file_name="recommendations_concern.csv",
                mime="text/csv"
            )

        st.markdown("---")

        # === B2: User-user similarity ===
        st.markdown("### Users similar to you liked these products")
        st.caption("Based on purchase patterns of similar users")

        if mode == "New user (no history)":
            st.info("This section is available after you have purchase history.")
        else:
            # Existing user → use user-user similarity
            if user_id is None:
                st.warning("User ID not found.")
            else:
                recs_similar = rec.recommend_by_user_similarity(
                    user_id=user_id,
                    top_k=top_k,
                    allow_repeats=allow_repeats,
                    repurchase_boost=repurchase_boost,
                )

                if not recs_similar:
                    st.info("No similar users found or no recommendations available.")
                else:
                    rec_df2 = pd.DataFrame([
                        {"item_number": r.item_number, "score": r.score, "reason": r.reason} 
                        for r in recs_similar
                    ])
                    show2 = rec_df2.merge(item_meta, on="item_number", how="left")
                    st.dataframe(
                        show2[["item_number", "skin_concern_cat_name", "score", "reason"]],
                        use_container_width=True
                    )
                    st.download_button(
                        "Download similarity-based results (CSV)",
                        data=show2.to_csv(index=False).encode("utf-8"),
                        file_name="recommendations_similarity.csv",
                        mime="text/csv"
                    )