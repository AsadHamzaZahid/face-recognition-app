import streamlit as st
import numpy as np
import cv2
import os
import pickle
from PIL import Image
from deepface import DeepFace

DB_FILE = "face_database.pkl"
MODEL = "VGG-Face"
THRESHOLD = 0.6

st.set_page_config(page_title="Face Recognition App", layout="wide")
st.title("Face Recognition App")


def load_db():

    if os.path.exists(DB_FILE):
        with open(DB_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def save_db(db):
    with open(DB_FILE, "wb") as f:
        pickle.dump(db, f)


def pil_to_rgb(pil_img):
    return np.array(pil_img.convert("RGB"))


def get_embedding(img_rgb):
    result = DeepFace.represent(
        img_rgb, model_name=MODEL, enforce_detection=True)
    return np.array(result[0]["embedding"])


def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def find_match(embedding, db):
    best_name = "Unknown"
    best_dist = THRESHOLD
    for name, embeddings in db.items():
        dists = [cosine_distance(embedding, e) for e in embeddings]
        min_dist = min(dists)
        if min_dist < best_dist:
            best_dist = min_dist
            best_name = name
    return best_name, round(best_dist, 3)


def detect_faces(img_rgb):
    try:
        faces = DeepFace.extract_faces(img_rgb, enforce_detection=True)
        return faces
    except Exception:
        return []


def draw_box(img_rgb, facial_area, name, dist):
    img = img_rgb.copy()
    x = facial_area["x"]
    y = facial_area["y"]
    w = facial_area["w"]
    h = facial_area["h"]
    color = (0, 200, 80) if name != "Unknown" else (220, 0, 0)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    label = f"{name} ({dist})"
    cv2.rectangle(img, (x, y + h - 28), (x + w, y + h), color, cv2.FILLED)
    cv2.putText(img, label, (x + 6, y + h - 8),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1)
    return img


# ── Sidebar ────────────────────────────────────────────────────────────────
mode = st.sidebar.radio("Mode", ["Enroll a person", "Recognize faces"])
db = load_db()
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Database:** {len(db)} person(s)")
if db:
    st.sidebar.markdown("**Enrolled:** " + ", ".join(db.keys()))
if st.sidebar.button("Clear database"):
    save_db({})
    st.sidebar.success("Cleared!")
    st.rerun()

# ── ENROLL ─────────────────────────────────────────────────────────────────
if mode == "Enroll a person":
    st.header("Enroll a person")
    name = st.text_input("Person's name", placeholder="e.g. Ahmed Khan")
    photo = st.file_uploader(
        "Upload a clear front-facing photo", type=["jpg", "jpeg", "png"])

    if photo and name:
        img_rgb = pil_to_rgb(Image.open(photo))
        st.image(img_rgb, caption="Uploaded photo", width=300)

        if st.button("Encode & save"):
            with st.spinner("Encoding face..."):
                try:
                    embedding = get_embedding(img_rgb)
                    db.setdefault(name, []).append(embedding)
                    save_db(db)
                    st.success(
                        f"Saved {name}! ({len(db[name])} photo(s) enrolled)")
                    st.info(
                        "Tip: enroll 2-3 photos per person for better accuracy.")
                except Exception as e:
                    st.error(f"No face detected or error: {e}")

# ── RECOGNIZE ──────────────────────────────────────────────────────────────
else:
    st.header("Recognize faces")
    if not db:
        st.warning("No people enrolled yet. Go to 'Enroll a person' first.")
    else:
        photo = st.file_uploader("Upload image to scan", type=[
                                 "jpg", "jpeg", "png"])
        if photo:
            img_rgb = pil_to_rgb(Image.open(photo))
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original")
                st.image(img_rgb, use_container_width=True)

            with st.spinner("Detecting and recognizing faces..."):
                faces = detect_faces(img_rgb)

            if not faces:
                st.error("No faces detected.")
            else:
                annotated = img_rgb.copy()
                results = []
                for face in faces:
                    area = face["facial_area"]
                    x, y, w, h = area["x"], area["y"], area["w"], area["h"]
                    face_crop = img_rgb[y:y + h, x:x + w]
                    try:
                        emb = get_embedding(face_crop)
                        name, dist = find_match(emb, db)
                    except Exception:
                        name, dist = "Unknown", 1.0
                    results.append((name, dist))
                    annotated = draw_box(annotated, area, name, dist)

                with col2:
                    st.subheader("Result")
                    st.image(annotated, use_container_width=True)

                st.markdown("### Matches")
                for i, (name, dist) in enumerate(results):
                    icon = "✅" if name != "Unknown" else "❌"
                    color = "green" if name != "Unknown" else "red"
                    st.markdown(
                        f"{icon} **Face {i+1}** → "
                        f"<span style='color:{color}'>{name}</span> "
                        f"(distance: `{dist}`)",
                        unsafe_allow_html=True
                    )
