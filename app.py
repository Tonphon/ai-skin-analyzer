"""
Skin Analyzer + Recommender — combined app
Face alignment uses dlib (shape_predictor_68_face_landmarks.dat).
Concern descriptions are shown after analysis.

Requirements:
    pip install streamlit pillow opencv-python-headless dlib numpy pandas scikit-learn timm torch torchvision

Run:
    streamlit run app.py
"""
import io
import os
import types
import numpy as np
import cv2
import dlib
import streamlit as st
import pandas as pd
from PIL import Image

from src.classifier import SkinConcernClassifier
from src.recommender import Recommender
from src.config import CONCERN_ID_TO_NAME, CLASS_THRESHOLDS
from src.concern_descriptions import get_all_descriptions_for_concerns

# ── dlib setup ────────────────────────────────────────────────────────────────
_detector = dlib.get_frontal_face_detector()
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_predictor_path = os.path.join(_BASE_DIR, "shape_predictor_68_face_landmarks.dat")
if not os.path.exists(_predictor_path):
    st.error(
        "Missing `shape_predictor_68_face_landmarks.dat` in the project root.\n\n"
        "Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n"
        "Extract and place the .dat file next to app.py."
    )
    st.stop()
_predictor = dlib.shape_predictor(_predictor_path)

# ── Oval / ellipse config ─────────────────────────────────────────────────────
_CENTER_RATIO   = (0.5, 0.5)
_AXES_RATIO     = (0.20, 0.45)
_MAX_ANGLE_DEG  = 5       # max in-plane eye roll for front step
_MIN_SIZE_RATIO = 0.98    # face must fill ≥98% of oval axes
_FRONT_TURN_MAX = 0.10    # near-center for front photo
_SIDE_TURN_MIN  = 0.3    # minimum normalized turn for side photo
_SIDE_TURN_MAX  = 0.55    # maximum normalized turn for side photo
_MAX_SIDE_ROLL  = 12      # allow a bit more tilt on side shots
_STABLE_FRAMES  = 10

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
.concern-badge { display: inline-block; padding: 4px 12px; border-radius: 16px; font-size: 12px; font-weight: 600;
                 margin-right: 8px; margin-bottom: 8px; }
.concern-badge.whitening  { background: #fff3e0; color: #e65100; }
.concern-badge.anti-aging { background: #f3e5f5; color: #6a1b9a; }
.concern-badge.acne       { background: #e8f5e9; color: #2e7d32; }
.concern-badge.eye-care   { background: #e3f2fd; color: #1565c0; }
.concern-badge.sensitive  { background: #fce4ec; color: #c2185b; }
</style>
""", unsafe_allow_html=True)


# ── Face analysis helpers (dlib) ──────────────────────────────────────────────

def _get_eye_roll(shape_np: np.ndarray) -> float:
    """In-plane roll angle from dlib eye landmarks (indices 36-47)."""
    left_eye_center  = np.mean(shape_np[36:42], axis=0)
    right_eye_center = np.mean(shape_np[42:48], axis=0)
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    return float(np.degrees(np.arctan2(dy, dx)))


def _estimate_turn(shape_np: np.ndarray, face) -> float:
    """Approximate left/right turn from nose position inside the face box.

    This is a simple 2D heuristic:
    - take the nose tip x-position (landmark 30)
    - compare it with the horizontal center of the detected face box
    - normalize by half the face width

    Output is unitless, not true degrees.
    Negative ~= subject turned left, positive ~= subject turned right.
    """
    nose_x = float(shape_np[30, 0])
    face_center_x = (face.left() + face.right()) / 2.0
    face_half_w = max((face.right() - face.left()) / 2.0, 1.0)
    return float((nose_x - face_center_x) / face_half_w)


def _evaluate_step_alignment(step: int, in_ellipse: bool, size_ok: bool, roll: float, turn: float) -> tuple[bool, str]:
    base_ok = in_ellipse and size_ok

    if step == 0:
        turn_ok = abs(turn) <= _FRONT_TURN_MAX
        angle_ok = abs(roll) <= _MAX_ANGLE_DEG
        if base_ok and turn_ok and angle_ok:
            return True, "Face aligned ✅ Hold still..."
        if not in_ellipse:
            return False, "Move your face into the oval guide"
        if not size_ok:
            return False, "Move closer ❌ Face is too small"
        if not turn_ok:
            if turn < 0:
                return False, "Turn slightly back to center from the left"
            return False, "Turn slightly back to center from the right"
        if roll > 0:
            return False, "Head tilted right ❌ Straighten up"
        return False, "Head tilted left ❌ Straighten up"

    if step == 1:  # subject's left
        turn_ok = (-_SIDE_TURN_MAX <= turn <= -_SIDE_TURN_MIN)
        angle_ok = abs(roll) <= _MAX_SIDE_ROLL
        if base_ok and turn_ok and angle_ok:
            return True, "Left angle looks good ✅ Hold still..."
        if not in_ellipse:
            return False, "Keep your face inside the oval while turning left"
        if not size_ok:
            return False, "Move closer ❌ Face is too small"
        if turn > -_SIDE_TURN_MIN:
            return False, "Turn more to your left"
        if turn < -_SIDE_TURN_MAX:
            return False, "Turn a little back toward the center"
        return False, "Keep your head more upright"

    turn_ok = (_SIDE_TURN_MIN <= turn <= _SIDE_TURN_MAX)
    angle_ok = abs(roll) <= _MAX_SIDE_ROLL
    if base_ok and turn_ok and angle_ok:
        return True, "Right angle looks good ✅ Hold still..."
    if not in_ellipse:
        return False, "Keep your face inside the oval while turning right"
    if not size_ok:
        return False, "Move closer ❌ Face is too small"
    if turn < _SIDE_TURN_MIN:
        return False, "Turn more to your right"
    if turn > _SIDE_TURN_MAX:
        return False, "Turn a little back toward the center"
    return False, "Keep your head more upright"


def analyze_face(image_rgb: np.ndarray, step: int) -> dict:
    """Detect face with dlib and evaluate alignment for the current step."""
    image_rgb = np.ascontiguousarray(image_rgb)
    if image_rgb.dtype != np.uint8:
        if np.issubdtype(image_rgb.dtype, np.floating):
            if image_rgb.size and float(np.nanmax(image_rgb)) <= 1.0:
                image_rgb = (image_rgb * 255).clip(0, 255).astype(np.uint8)
            else:
                image_rgb = np.nan_to_num(image_rgb, nan=0.0).clip(0, 255).astype(np.uint8)
        else:
            image_rgb = np.nan_to_num(image_rgb, nan=0.0).clip(0, 255).astype(np.uint8)
    else:
        image_rgb = image_rgb.copy()

    if image_rgb.ndim == 2:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
    elif image_rgb.ndim == 3 and image_rgb.shape[2] == 4:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)
    elif image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Unsupported image shape for face analysis: {image_rgb.shape}")

    image_rgb = np.ascontiguousarray(image_rgb, dtype=np.uint8)

    h, w = image_rgb.shape[:2]
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    gray = np.ascontiguousarray(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY), dtype=np.uint8)

    ell_cx = int(w * _CENTER_RATIO[0])
    ell_cy = int(h * _CENTER_RATIO[1])
    ell_ax = int(w * _AXES_RATIO[0])
    ell_ay = int(h * _AXES_RATIO[1])

    faces = _detector(gray)
    if len(faces) == 0:
        cv2.ellipse(img_bgr, (ell_cx, ell_cy), (ell_ax, ell_ay), 0, 0, 360, (255, 255, 255), 4)
        return {
            "roll": 0.0,
            "turn": 0.0,
            "coverage": 0.0,
            "in_ellipse": False,
            "size_ok": False,
            "face_ok": False,
            "status_text": "No face detected",
            "annotated_img": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
        }

    face = max(faces, key=lambda f: (f.right() - f.left()) * (f.bottom() - f.top()))
    shape = _predictor(gray, face)
    shape_np = np.array([[p.x, p.y] for p in shape.parts()])

    cx = float(np.mean(shape_np[:, 0]))
    cy = float(np.mean(shape_np[:, 1]))
    face_w = face.right() - face.left()
    face_h = face.bottom() - face.top()

    in_ellipse = ((((cx - ell_cx) ** 2) / (ell_ax ** 2)) + (((cy - ell_cy) ** 2) / (ell_ay ** 2))) <= 1
    size_ok = (face_w >= ell_ax * _MIN_SIZE_RATIO) and (face_h >= ell_ay * _MIN_SIZE_RATIO)
    coverage = (face_w * face_h) / (w * h) * 100
    roll = _get_eye_roll(shape_np)
    turn = _estimate_turn(shape_np, face)

    face_ok, status_text = _evaluate_step_alignment(step, in_ellipse, size_ok, roll, turn)
    oval_color = (0, 255, 0) if face_ok else (0, 0, 255)
    cv2.ellipse(img_bgr, (ell_cx, ell_cy), (ell_ax, ell_ay), 0, 0, 360, oval_color, 4)

    return {
        "roll": roll,
        "turn": turn,
        "coverage": coverage,
        "in_ellipse": in_ellipse,
        "size_ok": size_ok,
        "face_ok": face_ok,
        "status_text": status_text,
        "annotated_img": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
    }


# ── Cached resource loaders ───────────────────────────────────────────────────

@st.cache_resource
def load_classifier():
    return SkinConcernClassifier("models/best_model.pth", device="cpu", arch="efficientnet_v2_s")


@st.cache_resource
def load_recommender():
    try:
        return Recommender("artifacts/cf_bundle.pkl")
    except Exception as e:
        st.error(f"Failed to load artifacts/cf_bundle.pkl: {type(e).__name__}: {e}")
        st.stop()


@st.cache_data
def load_item_meta():
    return pd.read_csv("data/item_master_with_skin_concern_cat.csv")


def _format_recs(recs, item_meta: pd.DataFrame) -> pd.DataFrame:
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

# ── Fix: read-only numpy array in recommender's fill_diagonal ─────────────────
# recommender.py calls np.fill_diagonal(group_sim.values, 0) but pandas can
# return a read-only view. Patch np.fill_diagonal to copy before writing.
_orig_fill_diagonal = np.fill_diagonal

def _safe_fill_diagonal(a, val, wrap=False):
    if isinstance(a, np.ndarray) and not a.flags.writeable:
        # Can't fix in-place — caller needs a writable copy, but we can't
        # reassign their local var. Best we can do: make a writable copy,
        # fill it, then copy data back via flat iterator trick.
        writeable = a.copy()
        _orig_fill_diagonal(writeable, val, wrap)
        a.setflags(write=True)
        a[:] = writeable
    else:
        _orig_fill_diagonal(a, val, wrap)

np.fill_diagonal = _safe_fill_diagonal

# ── Session state ─────────────────────────────────────────────────────────────
if "step"            not in st.session_state: st.session_state.step            = 0
if "captures"        not in st.session_state: st.session_state.captures        = [None, None, None]
if "cam_key"         not in st.session_state: st.session_state.cam_key         = 0
if "pred_scores"     not in st.session_state: st.session_state.pred_scores     = [None, None, None]
if "chosen_concerns" not in st.session_state: st.session_state.chosen_concerns = []
if "last_live_frame"   not in st.session_state: st.session_state.last_live_frame   = None
if "last_capture_rgb"  not in st.session_state: st.session_state.last_capture_rgb  = None
if "stable_ok_count"   not in st.session_state: st.session_state.stable_ok_count   = 0

STEPS = [
    {"label": "Face Forward", "title": "Face the camera straight on",
     "desc": "Look directly into the camera. Center your face in the oval guide so your forehead and chin are both visible.", "target_yaw": 0},
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
                     caption=STEPS[i]["label"], width="stretch")
            scores = st.session_state.pred_scores[i]
            if scores is not None:
                score_df = (
                    pd.DataFrame(scores.items(), columns=["label", "score"])
                    .sort_values("score", ascending=False)
                    .reset_index(drop=True)
                )
                score_df["score"] = score_df["score"].apply(lambda x: f"{x:.4f}")
                st.caption(f"Classification scores: Photo {i + 1}")
                st.dataframe(score_df, width="stretch", hide_index=True)
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
            st.session_state.last_capture_rgb = None
            st.rerun()

    if analyze_btn:
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

        st.markdown("### 🔍 ผลการวิเคราะห์ปัญหาผิวหน้า")

        if chosen:
            all_positive_labels = set()
            for scores in st.session_state.pred_scores:
                if scores:
                    for label, score in scores.items():
                        threshold = CLASS_THRESHOLDS.get(label, 0.5)
                        if score >= threshold:
                            all_positive_labels.add(label)

            if all_positive_labels:
                descriptions = get_all_descriptions_for_concerns(list(all_positive_labels))

                for label in sorted(all_positive_labels):
                    desc = descriptions[label]
                    with st.expander(f"✨ {desc['title']}", expanded=True):
                        st.markdown(f"**คำอธิบาย:** {desc['description']}")
                        st.markdown(f"**คำแนะนำ:** {desc['tips']}")
                        detected_in = []
                        for i, scores in enumerate(st.session_state.pred_scores):
                            if scores and label in scores:
                                threshold = CLASS_THRESHOLDS.get(label, 0.5)
                                if scores[label] >= threshold:
                                    detected_in.append(f"ภาพที่ {i+1} ({scores[label]:.2%})")
                        if detected_in:
                            st.caption(f"🎯 ตรวจพบใน: {', '.join(detected_in)}")

                st.markdown("---")
                st.info(f"💡 **สรุป:** ตรวจพบปัญหาผิวทั้งหมด {len(all_positive_labels)} ประเภท จาก {len(chosen)} กลุ่มปัญหาหลัก")
            else:
                st.info("ไม่พบปัญหาผิวที่เด่นชัดในภาพของคุณ ผิวของคุณดูดีอยู่แล้ว! 😊")
        else:
            st.warning(
                "⚠️ ไม่สามารถตรวจจับปัญหาผิวได้จากภาพทั้ง 3 ภาพ "
                "กรุณาถ่ายภาพใหม่และตรวจสอบให้แน่ใจว่าใบหน้าของคุณชัดเจน"
            )
            st.stop()

        # ── Recommendations ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## Recommendations")

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
            st.dataframe(_format_recs(recs_concern, item_meta), width="stretch")

        st.markdown("---")

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
                    st.dataframe(_format_recs(recs_similar, item_meta), width="stretch")

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

    st.caption("Live webcam guide: follow the oval color and text below. The app auto-captures after the frame stays valid briefly.")
    run = st.checkbox("Start Webcam", value=True, key=f"run_cam_{step}")

    preview = st.empty()
    status_box = st.empty()

    ctrl1, ctrl2 = st.columns(2)
    with ctrl1:
        if st.button("Retake current step"):
            st.session_state.captures[step] = None
            st.session_state.pred_scores[step] = None
            st.session_state.last_live_frame = None
            st.session_state.stable_ok_count = 0
            st.session_state.last_capture_rgb = None
            st.rerun()
    with ctrl2:
        if st.button("Reset all photos"):
            st.session_state.step = 0
            st.session_state.captures = [None, None, None]
            st.session_state.cam_key = 0
            st.session_state.pred_scores = [None, None, None]
            st.session_state.chosen_concerns = []
            st.session_state.last_live_frame = None
            st.session_state.stable_ok_count = 0
            st.session_state.last_capture_rgb = None
            st.rerun()

    if run:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if not cap.isOpened():
            st.error("Cannot access webcam")
        else:
            try:
                while run:
                    ret, frame = cap.read()
                    if not ret:
                        status_box.markdown('<div class="err-box">❌ Cannot access webcam frame.</div>', unsafe_allow_html=True)
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = analyze_face(frame_rgb, step)
                    preview.image(result["annotated_img"], width="stretch")
                    st.session_state.last_live_frame = result["annotated_img"]
                    st.session_state.last_capture_rgb = frame_rgb.copy()

                    if result["face_ok"]:
                        st.session_state.stable_ok_count += 1
                        status_box.markdown(
                            f'<div class="ok-box">✅ {result["status_text"]} ({st.session_state.stable_ok_count}/{_STABLE_FRAMES})</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.session_state.stable_ok_count = 0
                        css = "err-box" if result["status_text"] == "No face detected" else "warn-box"
                        icon = "❌" if css == "err-box" else "⚠️"
                        status_box.markdown(
                            f'<div class="{css}">{icon} {result["status_text"]}</div>',
                            unsafe_allow_html=True,
                        )

                    if st.session_state.stable_ok_count >= _STABLE_FRAMES:
                        capture_rgb = st.session_state.last_capture_rgb if st.session_state.last_capture_rgb is not None else frame_rgb
                        capture_pil = Image.fromarray(capture_rgb)
                        buf = io.BytesIO()
                        capture_pil.save(buf, format="PNG")
                        st.session_state.captures[step] = buf.getvalue()
                        st.session_state.step += 1
                        st.session_state.cam_key += 1
                        st.session_state.stable_ok_count = 0
                        break
            finally:
                cap.release()

            if st.session_state.step != step:
                st.rerun()
    else:
        if st.session_state.last_live_frame is not None:
            preview.image(st.session_state.last_live_frame, width="stretch")
        status_box.markdown('<div class="warn-box">⚠️ Webcam paused.</div>', unsafe_allow_html=True)
