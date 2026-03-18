"""
pip install streamlit pillow mediapipe==0.10.9 opencv-python-headless numpy pandas
streamlit run app.py
"""
import io
import numpy as np
import cv2
import mediapipe as mp
import streamlit as st
import pandas as pd
from PIL import Image

from src.classifier import SkinConcernClassifier
from src.recommender import Recommender
from src.config import CONCERN_ID_TO_NAME

# ── MediaPipe setup ───────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh

st.set_page_config(page_title="Skin Analyzer + Recommender", layout="wide")

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.progress-wrap { display: flex; gap: 10px; justify-content: center; margin-bottom: 24px; }
.step-chip { display: flex; align-items: center; gap: 8px; padding: 6px 16px; border-radius: 999px;
             font-size: 13px; font-weight: 600; border: 1.5px solid #333; color: #555; background: #111; }
.step-chip.active { border-color: #00ff99; color: #00ff99; background: #001a0f; }
.step-chip.done   { border-color: #1a4a30; color: #2a6a44; background: #0d1f18; }
.guide-card { background: #111; border: 1px solid #222; border-radius: 12px; padding: 20px;
              margin-bottom: 20px; display: flex; align-items: center; gap: 20px; }
.guide-text h3 { margin: 0 0 6px; font-size: 17px; color: #00ff99; }
.guide-text p  { margin: 0; font-size: 13px; color: #888; line-height: 1.6; }
.bar-wrap { background: #111; border: 1px solid #222; border-radius: 10px; padding: 14px 18px; margin-bottom: 10px; }
.bar-label { font-size: 12px; color: #666; margin-bottom: 6px; display: flex; justify-content: space-between; }
.bar-track { position: relative; height: 8px; background: #222; border-radius: 4px; overflow: visible; margin-bottom: 4px; }
.bar-target { position: absolute; top: -4px; width: 3px; height: 16px; background: #00ff99; border-radius: 2px; opacity: 0.6; }
.bar-cursor { position: absolute; top: -5px; width: 5px; height: 18px; background: #fff; border-radius: 2px; transform: translateX(-50%); }
.ok-box   { background: #001a0f; border: 1px solid #1a4a30; border-radius: 8px; padding: 10px 14px; color: #00ff99; font-size: 13px; margin-bottom: 10px; }
.warn-box { background: #1a1200; border: 1px solid #665500; border-radius: 8px; padding: 10px 14px; color: #ffcc00; font-size: 13px; margin-bottom: 10px; }
.err-box  { background: #1a0a0a; border: 1px solid #4a1a1a; border-radius: 8px; padding: 10px 14px; color: #ff6666; font-size: 13px; margin-bottom: 10px; }
.result-card { background: #0d1f18; border: 1px solid #1a4a30; border-radius: 12px; padding: 28px; text-align: center; margin-top: 16px; }
</style>
""", unsafe_allow_html=True)


# ── Face analysis helpers (from implement.py) ─────────────────────────────────

def analyze_face(image_rgb: np.ndarray) -> dict | None:
    """Returns yaw (degrees) and coverage (% of image area), or None if no face detected."""
    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5,
    ) as face_mesh:
        results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return None

    lm = results.multi_face_landmarks[0].landmark
    h, w = image_rgb.shape[:2]

    # Yaw via solvePnP
    model_pts = np.array([
        (0.0,    0.0,    0.0),
        (0.0,   -63.6,  -12.5),
        (-43.3,  32.7,  -26.0),
        (43.3,   32.7,  -26.0),
        (-28.9, -28.9,  -24.1),
        (28.9,  -28.9,  -24.1),
    ], dtype=np.float64)
    idxs = [1, 152, 226, 446, 57, 287]
    img_pts = np.array([(lm[i].x * w, lm[i].y * h) for i in idxs], dtype=np.float64)
    focal = w
    cam = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float64)
    _, rvec, _ = cv2.solvePnP(model_pts, img_pts, cam, np.zeros((4, 1)),
                               flags=cv2.SOLVEPNP_ITERATIVE)
    rmat, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    yaw = angles[1]

    # Coverage via bounding box of all landmarks
    xs = [l.x for l in lm]
    ys = [l.y for l in lm]
    box_w = (max(xs) - min(xs)) * w
    box_h = (max(ys) - min(ys)) * h
    coverage = (box_w * box_h) / (w * h) * 100

    return {"yaw": yaw, "coverage": coverage}


def check_pose(yaw: float, step: int) -> tuple[bool, str]:
    targets = {0: (0, 20), 1: (-45, 25), 2: (45, 25)}
    target, tol = targets[step]
    if abs(yaw - target) <= tol:
        return True, ""
    if step == 0:
        msg = f"Turn a little to the right ({yaw:.0f}°)" if yaw < target - tol else f"Turn a little to the left ({yaw:.0f}°)"
        return False, msg + " — try facing more straight ahead"
    elif step == 1:
        msg = f"Not enough ({yaw:.0f}°) — turn your head further left" if yaw > target + tol else f"Too far ({yaw:.0f}°) — turn back slightly"
        return False, msg
    else:
        msg = f"Not enough ({yaw:.0f}°) — turn your head further right" if yaw < target - tol else f"Too far ({yaw:.0f}°) — turn back slightly"
        return False, msg


def check_coverage(cov: float) -> tuple[bool, str]:
    if cov < 5:
        return False, f"Move closer to the camera — face covers {cov:.0f}% of frame (target: 10-30%)"
    elif cov > 40:
        return False, f"Move further from the camera — face covers {cov:.0f}% of frame (target: 10-30%)"
    return True, ""


# ── Cached resource loaders ───────────────────────────────────────────────────

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

# ── Session state ─────────────────────────────────────────────────────────────
if "step"             not in st.session_state: st.session_state.step             = 0
if "captures"         not in st.session_state: st.session_state.captures         = [None, None, None]
if "cam_key"          not in st.session_state: st.session_state.cam_key          = 0
if "pred_scores"      not in st.session_state: st.session_state.pred_scores      = [None, None, None]
if "chosen_concerns"  not in st.session_state: st.session_state.chosen_concerns  = []

STEPS = [
    {"label": "Face Forward", "title": "Face the camera straight on",
     "desc": "Look directly into the camera. Center your face in the frame so your forehead and chin are both visible.", "target_yaw": 0},
    {"label": "Turn Left",    "title": "Turn your face to the left",
     "desc": "Turn your head approximately 45° to your left so your cheek and ear are clearly visible.", "target_yaw": -45},
    {"label": "Turn Right",   "title": "Turn your face to the right",
     "desc": "Turn your head approximately 45° to your right so your cheek and ear are clearly visible.", "target_yaw": 45},
]

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
        user_id   = st.sidebar.selectbox("Select User ID", options=available_users)
        user_demo = rec.get_user_demographics(user_id)
        if user_demo:
            st.sidebar.info(f"**Gender:** {user_demo['gender']}")
            st.sidebar.info(f"**Age Group:** {user_demo['age_group']}")
            gender    = user_demo["gender"]
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
allow_repeats    = st.sidebar.toggle("Allow repeat recommendations", value=True)
top_k            = st.sidebar.slider("Top-K recommendations", min_value=1, max_value=20, value=10)
repurchase_boost = st.sidebar.slider("Repurchase boost (existing users)", min_value=1.0, max_value=2.0, value=1.2, step=0.1)

# ── Step progress chips ───────────────────────────────────────────────────────
chips = '<div class="progress-wrap">'
for i, s in enumerate(STEPS):
    if i < st.session_state.step or st.session_state.step == 3:
        cls, num = "done", "+"
    elif i == st.session_state.step:
        cls, num = "active", str(i + 1)
    else:
        cls, num = "", str(i + 1)
    chips += f'<div class="step-chip {cls}"><span>{num}</span>{s["label"]}</div>'
chips += "</div>"
st.markdown(chips, unsafe_allow_html=True)

# ── Thumbnails row ────────────────────────────────────────────────────────────
thumb_cols = st.columns(3)
for i, col in enumerate(thumb_cols):
    with col:
        if st.session_state.captures[i] is not None:
            st.image(Image.open(io.BytesIO(st.session_state.captures[i])),
                     caption=STEPS[i]["label"], use_container_width=True)
            # Show classification scores for this photo if available
            scores = st.session_state.pred_scores[i]
            if scores is not None:
                score_df = (
                    pd.DataFrame(scores.items(), columns=["label", "score"])
                    .sort_values("score", ascending=False)
                    .reset_index(drop=True)
                )
                score_df["score"] = score_df["score"].apply(lambda x: f"{x:.4f}")
                st.caption(f"Classification scores: Photo {i + 1}")
                st.dataframe(score_df, use_container_width=True, hide_index=True)
        else:
            st.markdown(
                f'<div style="width:100%;height:88px;border-radius:8px;border:1.5px dashed #2a2a2a;'
                f'background:#111;display:flex;align-items:center;justify-content:center;'
                f'font-size:13px;color:#444;">{STEPS[i]["label"]}</div>',
                unsafe_allow_html=True,
            )

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ── All 3 captures done: run analysis + recommendations ───────────────────────
if st.session_state.step == 3:
    st.markdown("""
    <div class="result-card">
        <h3 style="color:#00ff99;margin:0 0 6px;">All 3 photos captured!</h3>
        <p style="color:#888;font-size:13px;margin:0;">Ready to analyze your skin concerns.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        analyze_btn = st.button("Analyze and Recommend", type="primary",
                                disabled=all(s is not None for s in st.session_state.pred_scores))
    with col_btn2:
        if st.button("Retake Photos"):
            st.session_state.step            = 0
            st.session_state.captures        = [None, None, None]
            st.session_state.cam_key         = 0
            st.session_state.pred_scores     = [None, None, None]
            st.session_state.chosen_concerns = []
            st.rerun()

    if analyze_btn:
        # ── Classify all 3 images, store scores, union concern IDs ──────────
        all_concern_ids = set()
        with st.spinner("Analyzing skin concerns across all 3 photos..."):
            for i, raw_bytes in enumerate(st.session_state.captures):
                img_pil = Image.open(io.BytesIO(raw_bytes))
                pred    = clf.predict(img_pil)
                st.session_state.pred_scores[i] = pred.label_scores
                if pred.concern_ids:
                    all_concern_ids.update(pred.concern_ids)
        st.session_state.chosen_concerns = sorted(all_concern_ids)
        st.rerun()

    # ── Show results if scores are already computed ───────────────────────
    if all(s is not None for s in st.session_state.pred_scores):
        chosen = st.session_state.get("chosen_concerns", [])

        st.markdown("### Detected skin concerns")
        if chosen:
            st.write([f"{cid} - {CONCERN_ID_TO_NAME.get(cid, str(cid))}" for cid in chosen])
        else:
            st.warning(
                "No concerns were detected from any of the 3 photos. "
                "Please retake your photos and ensure your face is clearly visible."
            )
            st.stop()

        # ── Recommendations ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## Recommendations")

        # ── Section 1: Concern-based ──────────────────────────────────────────
        st.markdown("### Products for your skin concerns")
        st.caption("Products matching your analyzed skin concerns")

        if mode == "New user (no history)":
            recs_concern = rec.recommend_new_user(
                gender=gender,
                birth_year=int(birth_year),
                selected_concern_ids=chosen,
                top_k=top_k,
            )
        else:
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
            st.dataframe(_format_recs(recs_concern, item_meta), use_container_width=True)

        st.markdown("---")

        # ── Section 2: User-similarity based ─────────────────────────────────
        st.markdown("### Users similar to you also liked")
        st.caption("Based on purchase patterns of similar users")

        if mode == "New user (no history)":
            st.info("This section is available once you have purchase history.")
        else:
            if user_id is None:
                st.warning("User ID not available.")
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
                    st.dataframe(_format_recs(recs_similar, item_meta), use_container_width=True)

# ── Active capture step ───────────────────────────────────────────────────────
else:
    step = st.session_state.step
    info = STEPS[step]

    st.markdown(f"""
    <div class="guide-card">
        <div class="guide-text">
            <h3>Step {step + 1} of {len(STEPS)}: {info['title']}</h3>
            <p>{info['desc']}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    img_file = st.camera_input(
        f"Step {step + 1}/{len(STEPS)} — {info['label']}",
        key=f"cam_{step}_{st.session_state.cam_key}",
    )

    # # Upload photo (disabled — uncomment to re-enable)
    # up = st.file_uploader("Or upload a photo", type=["jpg", "jpeg", "png"])
    # if up is not None:
    #     img_file = up

    if img_file is not None:
        img_pil = Image.open(img_file).convert("RGB")
        img_np  = np.array(img_pil)

        with st.spinner("Analyzing pose..."):
            result = analyze_face(img_np)

        if result is None:
            st.markdown('<div class="err-box">No face detected — please center your face in the frame and try again.</div>',
                        unsafe_allow_html=True)
        else:
            yaw        = result["yaw"]
            coverage   = result["coverage"]
            target_yaw = info["target_yaw"]

            pose_ok,  pose_msg = check_pose(yaw, step)
            cov_ok,   cov_msg  = check_coverage(coverage)
            clamp = lambda v, lo, hi: max(lo, min(hi, v))

            # ── Angle bar ─────────────────────────────────────────────────────
            pct_cursor = (clamp(yaw, -90, 90) + 90) / 180 * 100
            pct_target = (clamp(target_yaw, -90, 90) + 90) / 180 * 100
            bar_col    = "#00ff99" if pose_ok else "#ffcc00"
            st.markdown(f"""
            <div class="bar-wrap">
                <div class="bar-label">
                    <span>Left</span>
                    <span style="color:#aaa">Angle: <b style="color:{bar_col}">{yaw:.1f}</b>
                    &nbsp;|&nbsp; Target: <b style="color:#00ff99">{target_yaw}</b></span>
                    <span>Right</span>
                </div>
                <div class="bar-track">
                    <div class="bar-target" style="left:{pct_target:.1f}%"></div>
                    <div class="bar-cursor" style="left:{pct_cursor:.1f}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Coverage bar ──────────────────────────────────────────────────
            pct_cov     = clamp(coverage, 0, 100)
            pct_cov_lo  = 10
            pct_cov_hi  = 30
            cov_bar_col = "#00ff99" if cov_ok else "#ffcc00"
            st.markdown(f"""
            <div class="bar-wrap">
                <div class="bar-label">
                    <span>Far</span>
                    <span style="color:#aaa">Distance: <b style="color:{cov_bar_col}">{coverage:.0f}%</b>
                    &nbsp;|&nbsp; Target: <b style="color:#00ff99">10-30%</b></span>
                    <span>Close</span>
                </div>
                <div class="bar-track">
                    <div style="position:absolute;top:0;left:{pct_cov_lo}%;width:{pct_cov_hi - pct_cov_lo}%;
                         height:100%;background:rgba(0,255,153,0.15);border-radius:2px;"></div>
                    <div class="bar-cursor" style="left:{pct_cov:.1f}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Feedback messages ─────────────────────────────────────────────
            if pose_ok and cov_ok:
                st.markdown('<div class="ok-box">Pose and distance look great — ready to use this photo!</div>',
                            unsafe_allow_html=True)
            else:
                if not pose_ok:
                    st.markdown(f'<div class="warn-box">{pose_msg}</div>', unsafe_allow_html=True)
                if not cov_ok:
                    st.markdown(f'<div class="warn-box">{cov_msg}</div>', unsafe_allow_html=True)

        # ── Accept / retake buttons ───────────────────────────────────────────
        buf = io.BytesIO()
        img_pil.save(buf, format="JPEG", quality=92)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Use this photo"):
                st.session_state.captures[step] = buf.getvalue()
                st.session_state.step    += 1
                st.session_state.cam_key += 1
                st.rerun()
        with c2:
            if st.button("Retake"):
                st.session_state.cam_key += 1
                st.rerun()