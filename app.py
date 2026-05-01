import os
import re
import json
import base64
import tempfile
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
import librosa.effects
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import streamlit as st
import requests
from pathlib import Path
import streamlit.components.v1 as components
from urllib.parse import urlparse

# ============================================================
# SHARED CONSTANTS
# ============================================================
SR          = 22050
N_MELS      = 128
N_FFT       = 2048
HOP_LENGTH  = 512
CHUNK_SEC   = 4.0
IMG_SIZE    = (224, 224)
TOP_DB      = 30          # Keep same as preprocessing
MEL_DB_MIN  = -80.0       # Keep same as preprocessing
MEL_DB_MAX  =   0.0       # Keep same as preprocessing

# ============================================================
# APP CONFIG
# ============================================================
MODEL_PATH  = "model_output/best_final.keras"
CONFIG_PATH = "model_output/model_config.json"
CONFIDENCE_THRESHOLD = 0.65

INAT_TAXON_IDS = {
    "Asian Koel":                   "343700",
    "Indian Peafowl":               "4849",
    "Coppersmith Barbet":           "343683",
    "Jungle Babbler":               "554379",
    "Purple-rumped Sunbird":        "343833",
    "Greater Racket-tailed Drongo": "343737",
    "Indian Pitta":                 "343778",
    "White-throated Kingfisher":    "343690",
    "Red-wattled Lapwing":          "343811",
    "Malabar Whistling Thrush":     "343862",
}

WIKI_TITLES = {
    "Asian Koel":                   "Asian koel",
    "Indian Peafowl":               "Indian peafowl",
    "Coppersmith Barbet":           "Coppersmith barbet",
    "Jungle Babbler":               "Jungle babbler",
    "Purple-rumped Sunbird":        "Purple-rumped sunbird",
    "Greater Racket-tailed Drongo": "Greater racket-tailed drongo",
    "Indian Pitta":                 "Indian pitta",
    "White-throated Kingfisher":    "White-throated kingfisher",
    "Red-wattled Lapwing":          "Red-wattled lapwing",
    "Malabar Whistling Thrush":     "Malabar whistling thrush",
}

# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model_and_config():
    # Use fallback model if the best checkpoint is not present.
    model_path = MODEL_PATH
    if not Path(model_path).exists():
        model_path = "model_output/bird_classifier_final.keras"
    model = tf.keras.models.load_model(model_path)
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    return model, config["label_map"]


# ============================================================
# WIKIPEDIA DATA
# ============================================================

@st.cache_data(show_spinner=False)
def get_wiki_data(bird_name: str) -> dict:
    title  = WIKI_TITLES.get(bird_name, bird_name).replace(" ", "_")
    result = {"summary": "", "sci_name": "", "thumb_url": "", "page_url": ""}
    try:
        r = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
            timeout=10,
            headers={"User-Agent": "AvianAudioClassifier/2.0"}
        )
        if r.status_code == 200:
            data      = r.json()
            raw       = data.get("extract", "")
            sentences = [s.strip() for s in raw.split(".") if s.strip()]
            result["summary"] = ". ".join(sentences[:3]) + "."
            m = re.search(r'\(([A-Z][a-z]+ [a-z]+(?:\s[a-z]+)?)\)', raw)
            if m:
                result["sci_name"] = m.group(1)
            thumb = data.get("thumbnail", {})
            url   = thumb.get("source", "")
            if url:
                result["thumb_url"] = re.sub(r'/\d+px-', '/800px-', url)
            result["page_url"] = (
                data.get("content_urls", {}).get("desktop", {}).get("page", "")
            )
    except Exception:
        pass
    return result

# ============================================================
# CURATED PHOTO GALLERY
# ============================================================
BIRD_GALLERY = {
    "Indian Peafowl": [
        "https://imgs.search.brave.com/Vm3pGtkmho2O-s99Rve3LcRHyzAdhh-sFwN7d1lLWNY/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9pLnBpbmltZy5jb20vb3JpZ2luYWxzL2M4LzViLzBlL2M4NWIwZTBiMzRhZDdiZmI1YTdiZjhhMDAxZDYyNzE3LmpwZw"
    ],
    "Asian Koel": [
        "https://imgs.search.brave.com/U3dzutdLK0VqrCD1AJTzYQbh3PwFzZdmDA3nbZqpl2A/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9jZG4uZG93bmxvYWQuYW1zLmJpcmRzLmNvcm5lbGwuZWR1L2FwaS92Mi9hc3NldC8xMjYzOTI4MTEvMTIwMA"
    ],
    "Coppersmith Barbet": [
        "https://imgs.search.brave.com/BQvvUCLnidKkwfO9Bi7LOda4vDhoLtQD_C1H6fACCDg/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9pLnBpbmltZy5jb20vb3JpZ2luYWxzL2M3L2FkL2ZlL2M3YWRmZTI0OGM4NDg0ZWM0ZjY5NmRjNTRiYmNkNDZlLmpwZw"
    ],
    "Jungle Babbler": [
        "https://imgs.search.brave.com/QM06-JBIPdabm8swD1JwfWf_6VHkz1ybflLbboNBMNM/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9jZG4ucGl4YWJheS5jb20vcGhvdG8vMjAyMi8xMC8xMi8xMS81OS9iaXJkLTc1MTY0MTZfNjQwLmpwZw"
    ],
    "Purple-rumped Sunbird": [
        "https://imgs.search.brave.com/uDYq_2R6YoJY8CuDbx4UioH9h9I60KM6KUjKG3a1Mm0/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9tZWRpYS5nZXR0eWltYWdlcy5jb20vaWQvNDYxNTgyOTgzL3Bob3RvL3B1cnBsZS1ydW1wZWQtc3VuYmlyZC5qcGc_cz02MTJ4NjEyJnc9MCZrPTIwJmM9N3psOTlHNDBuVDZQc2Q3RDRFVkpJY2xpeThNT2t1dW5kSVdKbDBkM01ZMD0"
    ],
    "Greater Racket-tailed Drongo": [
        "https://imgs.search.brave.com/ml7CZBxx13c3U-FPdgTAthDjRw52uB3OLZ5SUT0sTiY/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly90NC5mdGNkbi5uZXQvanBnLzE1Lzg1LzU4LzY3LzM2MF9GXzE1ODU1ODY3NzFfV3k4bTlCUHg1Qm9aMnRLSEhvQ096NmNhRW96TWdSNm4uanBn"
    ],
    "Indian Pitta": [
        "https://imgs.search.brave.com/_HLc0zABcTlKvT8l2LyaurJNgXYxm198tE38V36y0js/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly91cGxvYWQud2lraW1lZGlhLm9yZy93aWtpcGVkaWEvY29tbW9ucy80LzQxL1BpdHRhX21vbHVjY2Vuc2lzXy1fS2FlbmdfS3JhY2hhbi5qcGc"
    ],
    "White-throated Kingfisher": [
        "https://imgs.search.brave.com/hb-HzG0Cl7hk9jRC4ZifuePxBx5grgvU37enSvQa-5s/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9zdGF0aWMudGhhaW5hdGlvbmFscGFya3MuY29tL2ltZy9zcGVjaWVzL3dpa2kvMjAxMy8wNy8wOS83MTc5L2hhbGN5b24tc215cm5lbnNpcy1rZXJhbGEtaW5kaWEtOC0xLXctMTUwMC5qcGc"
    ],
    "Red-wattled Lapwing": [
        "https://imgs.search.brave.com/zF3-uAJw2cih0_zaCtUId68ms_KnMaYgZi5BAZjVrw0/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly93d3cubmF0dXJlaW5mb2N1cy5pbi9tZWRpYS8xOTIwLTI1NjBweC1yZWQtd2F0dGxlZC1sYXB3aW5nLmpwZw"
    ],
    "Malabar Whistling Thrush": [
        "https://imgs.search.brave.com/wsaNRjE0hnN8iN7FG_bGhfJIM0pVCuaMH1YNI0AiyLw/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9jZG4uZG93bmxvYWQuYW1zLmJpcmRzLmNvcm5lbGwuZWR1L2FwaS92Mi9hc3NldC82NDA0MjQ2MDMvMTIwMA"
    ],
}

import hashlib

PHOTO_CACHE_DIR = Path("photo_cache")
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


def _resolve_source_image_url(url: str) -> str:
    """
    Brave image result URLs often embed the real source URL as a Base64 token.
    Decode that token so we fetch the direct image instead of the proxy wrapper.
    """
    if "imgs.search.brave.com" not in url:
        return url

    token = url.rstrip("/").split("/")[-1]
    padding = "=" * (-len(token) % 4)
    try:
        decoded = base64.urlsafe_b64decode(token + padding).decode("utf-8")
        if decoded.startswith(("http://", "https://")):
            return decoded
    except Exception:
        pass
    return url


def _infer_image_suffix(url: str, content_type: str = "") -> str:
    # Try file extension from URL first.
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in VALID_IMAGE_EXTS:
        return suffix

    # Otherwise infer from response content type.
    if "png" in content_type:
        return ".png"
    if "webp" in content_type:
        return ".webp"
    if "gif" in content_type:
        return ".gif"
    return ".jpg"

def get_cached_photo(url: str) -> str | None:
    """Download image once, cache locally, return local path."""
    PHOTO_CACHE_DIR.mkdir(exist_ok=True)

    source_url = _resolve_source_image_url(url)
    url_hash = hashlib.md5(source_url.encode()).hexdigest()

    # Reuse cached file if it already exists.
    existing = sorted(PHOTO_CACHE_DIR.glob(f"{url_hash}.*"))
    if existing:
        return str(existing[0])

    try:
        r = requests.get(source_url, timeout=12, allow_redirects=True, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        if r.status_code == 200 and r.content:
            content_type = (r.headers.get("Content-Type") or "").lower()
            if content_type and "image" not in content_type:
                return None

            ext = _infer_image_suffix(source_url, content_type)
            local_path = PHOTO_CACHE_DIR / f"{url_hash}{ext}"
            local_path.write_bytes(r.content)
            return str(local_path)
    except Exception:
        pass
    return None


@st.cache_data(show_spinner=False, ttl=60 * 60)
def get_inat_observation_points(taxon_id: str, per_page: int = 200) -> list[list[float]]:
    """Return [lat, lon] points for recent research-grade iNaturalist sightings."""
    if not taxon_id:
        return []

    try:
        r = requests.get(
            "https://api.inaturalist.org/v1/observations",
            params={
                "taxon_id": taxon_id,
                "quality_grade": "research",
                "verifiable": "true",
                "geo": "true",
                "order_by": "created_at",
                "order": "desc",
                "per_page": per_page,
            },
            timeout=12,
            headers={"User-Agent": "AvianAudioClassifier/2.0"},
        )
        if r.status_code != 200:
            return []

        points: list[list[float]] = []
        for obs in r.json().get("results", []):
            coords = (obs.get("geojson") or {}).get("coordinates")
            if not isinstance(coords, list) or len(coords) != 2:
                continue

            lon, lat = coords[0], coords[1]
            if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                points.append([float(lat), float(lon)])
        return points
    except Exception:
        return []

# ============================================================
# SPECTROGRAM
# ============================================================

def audio_to_melspec_array(y: np.ndarray, sr: int) -> np.ndarray:
    mel    = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = np.clip(mel_db, MEL_DB_MIN, MEL_DB_MAX)

    # Normalize using fixed dB bounds.
    mel_n  = (mel_db - MEL_DB_MIN) / (MEL_DB_MAX - MEL_DB_MIN) 

    mel_rgba = matplotlib.colormaps["viridis"](mel_n)  # â†’ (H, W, 4)
    mel_rgb  = (mel_rgba[:, :, :3] * 255).astype(np.uint8)
    mel_rgb  = np.flipud(mel_rgb)  # low freq at bottom

    # Resize and keep pixel scale expected by training.
    resized = tf.image.resize(mel_rgb, IMG_SIZE).numpy().astype(np.float32)
    return resized


# ============================================================
# INFERENCE
# ============================================================

def predict_from_audio(audio_path: str, model, label_map: dict):
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=TOP_DB)   # matches preprocess.py

    chunk_samples = int(CHUNK_SEC * SR)

    # Use 50% overlap to capture calls near chunk boundaries.
    hop    = chunk_samples // 2
    chunks = [
        y[s: s + chunk_samples]
        for s in range(0, len(y) - chunk_samples + 1, hop)
        if len(y[s: s + chunk_samples]) == chunk_samples
    ]

    # Pad short clips to one full chunk.
    if not chunks:
        if len(y) >= SR:
            pad           = np.zeros(chunk_samples, dtype=np.float32)
            pad[:len(y)]  = y
            chunks        = [pad]
        else:
            return None

    # Limit chunks for faster inference.
    if len(chunks) > 24:
        idx    = np.linspace(0, len(chunks) - 1, 24, dtype=int)
        chunks = [chunks[i] for i in idx]

    specs     = np.stack([audio_to_melspec_array(c, SR) for c in chunks])
    probs     = model.predict(specs, verbose=0)

    # Take per-class maximum confidence across chunks.
    max_probs = probs.max(axis=0) 
    # Re-normalize for probability display.
    avg_probs = max_probs / max_probs.sum()

    top_idx    = int(np.argmax(avg_probs))
    confidence = float(avg_probs[top_idx])
    return label_map[str(top_idx)], confidence, avg_probs


# ============================================================
# PAGE CONFIG & CSS
# ============================================================
st.set_page_config(
    page_title="Avian Audio Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@700&family=Manrope:wght@400;500;600;700&display=swap');

:root {
    --blue-1: #f1f7ff;
    --blue-2: #e4efff;
    --blue-3: #d7e8ff;
    --ink-1: #123b73;
    --ink-2: #2a5f9f;
    --ink-3: #4d7cb8;
    --card-bg: rgba(255, 255, 255, 0.84);
    --card-border: #b7cff5;
}

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 15% 20%, rgba(166, 204, 255, 0.42) 0%, rgba(166, 204, 255, 0) 42%),
        radial-gradient(circle at 85% 0%, rgba(145, 186, 245, 0.35) 0%, rgba(145, 186, 245, 0) 50%),
        linear-gradient(180deg, var(--blue-1) 0%, var(--blue-2) 52%, var(--blue-3) 100%);
    color: var(--ink-1);
}

h1, h2, h3 {
    font-family: 'Merriweather', serif;
}

.main-header { 
    text-align: center; padding: 2.5rem 0 1rem; 
}
.main-header h1 {
    font-size: 3rem; color: var(--ink-1);
    letter-spacing: -0.5px; margin-bottom: 0.3rem; font-weight: 700;
}
.main-header .tagline { 
    color: var(--ink-2); font-size: 1.05rem; font-weight: 500;
    letter-spacing: 0.5px;
}
.main-header .meta    { 
    color: var(--ink-3); font-size: 0.86rem; margin-top: 0.35rem;
    letter-spacing: 0.3px;
}

.pred-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.93) 0%, rgba(236,245,255,0.9) 100%);
    border: 1.5px solid var(--card-border); border-radius: 20px;
    padding: 2.2rem 1.8rem; text-align: center;
    box-shadow: 0 8px 28px rgba(52, 100, 168, 0.14);
}
.pred-card h2 {
    color: var(--ink-1); font-size: 2.1rem; margin: 0 0 0.25rem;
    font-family: 'Merriweather', serif; font-weight: 700;
}
.pred-card .sci  { color: #4e79ad; font-style: italic; font-size: 1rem; font-weight: 500; }
.pred-card .conf { font-size: 1.5rem; font-weight: 700; color: #245ea8; }

.about-card {
    background: var(--card-bg);
    border: 1.5px solid var(--card-border); border-radius: 16px;
    padding: 1.5rem 1.8rem; margin-top: 1rem;
    box-shadow: 0 4px 18px rgba(52, 100, 168, 0.12);
}
.about-card p { color: #1f4b85; font-size: 0.95rem; line-height: 1.7; font-weight: 500; margin: 0; }
.about-card a {
    color: #245ea8; font-size: 0.88rem; text-decoration: none;
    display: inline-block; margin-top: 0.8rem; font-weight: 500;
    border-bottom: 1px solid #7ca8de; padding-bottom: 1px;
}

.warn-card {
    background: linear-gradient(135deg, #fff8dd 0%, #fff2be 100%);
    border: 1.5px solid #d8bf63; border-radius: 14px;
    padding: 1rem 1.5rem; color: #715f19; font-size: 0.95rem;
}

.section-head {
    font-family: 'Merriweather', serif; font-size: 1.35rem;
    font-weight: 700; color: var(--ink-1);
    border-bottom: 1.5px solid #9fbfe9; padding-bottom: 0.35rem;
    margin: 1.8rem 0 1rem; letter-spacing: 0.3px;
}

.photo-wrap {
    background: var(--card-bg);
    border: 1.5px solid var(--card-border); border-radius: 16px;
    padding: 0.7rem; box-shadow: 0 4px 20px rgba(52, 100, 168, 0.12);
}
.photo-label {
    text-align: center; font-size: 0.75rem; color: #4e79ad;
    font-weight: 500; margin-top: 0.5rem;
    letter-spacing: 2px; text-transform: uppercase;
}

.soft-hr { border: none; border-top: 1.5px solid #a8c5ed; margin: 1.5rem 0; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(235, 244, 255, 0.94) 0%, rgba(222, 236, 255, 0.96) 100%);
    border-right: 1px solid #b6cef1;
}

[data-testid="stSidebar"] h3 {
    color: var(--ink-1) !important;
    font-family: 'Merriweather', serif !important;
}

[data-testid="stSidebar"] p, [data-testid="stSidebar"] li {
    color: #2d629f !important;
}

.footer {
    text-align: center; color: #3e6ea8; font-size: 0.78rem;
    padding: 2rem 0 1rem; border-top: 1px solid #b6cef1;
    margin-top: 2.5rem; letter-spacing: 0.3px;
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.82); border-radius: 14px; padding: 0.5rem;
    border: 1px solid #a8c4ed;
}

[data-testid="stFileUploader"] label {
    color: #2d629f !important;
}

.stMarkdown, .stText {
    color: #2a5f9f !important;
}

[data-testid="stChart"] {
    background: rgba(255,255,255,0.72);
    border-radius: 12px;
    padding: 10px;
}

div[data-testid="stVerticalBlock"] {
    background: rgba(255,255,255,0.35);
    border-radius: 16px;
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### Supported Species")
    for s in INAT_TAXON_IDS:
        st.markdown(f"- {s}")
    st.markdown("---")
    st.markdown("### Model Info")
    st.markdown("- **Architecture:** EfficientNetB0")
    st.markdown("- **Input:** Mel-spectrogram 224 x 224")
    st.markdown("- **Chunk size:** 4 seconds")
    st.markdown("- **Training data:** Xeno-canto API")
    st.markdown("- **Augmentation:** Time-stretch, pitch-shift, Gaussian noise, gain, pre-emphasis")
    st.markdown("---")
    st.markdown("### Region")
    st.markdown("Birds native to **Goa, India** and the Western Ghats.")

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>Avian Audio Classifier</h1>
    <p class="tagline">Upload a bird call | Identify the species | Explore its world</p>
    <p class="meta">10 species | Goa, India | EfficientNetB0 | Western Ghats</p>
</div>
<hr class="soft-hr">
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
with st.spinner("Loading model..."):
    try:
        model, label_map = load_model_and_config()
    except Exception as e:
        st.error(f"Could not load model: {e}\n\nRun 3_train.py first.")
        st.stop()

# ============================================================
# UPLOAD & INFERENCE
# ============================================================
st.markdown('<p class="section-head">Upload Bird Call Audio</p>',
            unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Supported formats: MP3, WAV, OGG, FLAC",
    type=["mp3", "wav", "ogg", "flac"]
)

if uploaded:
    st.audio(uploaded)

    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    with st.spinner("Analysing audio..."):
        prediction_result = predict_from_audio(
            tmp_path, model, label_map
        )
    os.unlink(tmp_path)

    if prediction_result is None:
        st.error("Audio too short. Please upload at least 1 second of audio.")
        st.stop()

    bird_name, confidence, all_probs = prediction_result

    if confidence < CONFIDENCE_THRESHOLD:
        st.markdown(f"""
        <div class="warn-card">
            <b>Low confidence ({confidence*100:.1f}%)</b> - This recording may not
            match any supported species clearly. The closest match is shown below.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")

    with st.spinner("Fetching species information..."):
        wiki      = get_wiki_data(bird_name)
        taxon_id  = INAT_TAXON_IDS.get(bird_name, "")
        
        # Use curated photos, then fallback to Wikipedia thumbnail.
        photos = BIRD_GALLERY.get(bird_name, [])
        if not photos and wiki.get("thumb_url"):
            photos = [(wiki.get("thumb_url"), "General")]

    st.markdown('<hr class="soft-hr">', unsafe_allow_html=True)

    # Prediction and probability chart
    col_pred, col_chart = st.columns([1, 1.5], gap="large")

    with col_pred:
        st.markdown('<p class="section-head">Prediction</p>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class="pred-card">
            <h2>{bird_name}</h2>
            <p class="sci">{wiki.get("sci_name","")}</p>
            <p class="conf">Confidence &nbsp; {confidence*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

        if wiki.get("summary"):
            page_link = (
                f'<a href="{wiki["page_url"]}" target="_blank">Read full Wikipedia article</a>'
                if wiki.get("page_url") else ""
            )
            st.markdown(f"""
            <div class="about-card">
                <p>{wiki["summary"]}</p>
                {page_link}
            </div>
            """, unsafe_allow_html=True)

    with col_chart:
        st.markdown('<p class="section-head">All Class Probabilities</p>',
                    unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        fig.patch.set_facecolor("#eef5ff")
        ax.set_facecolor("#eef5ff")

        species_names = [label_map[str(i)] for i in range(len(label_map))]
        colors = ["#2f6fbe" if s == bird_name else "#9dbbe6"
                  for s in species_names]
        bars = ax.barh(species_names, all_probs * 100, color=colors,
                       height=0.55, edgecolor="#7297cd", linewidth=0.8)
        ax.axvline(x=CONFIDENCE_THRESHOLD * 100, color="#f0aa3c",
                   linestyle="--", linewidth=1.2,
                   label=f"{CONFIDENCE_THRESHOLD*100:.0f}% threshold")
        ax.set_xlabel("Confidence (%)", fontsize=9, color="#3e6ea8")
        ax.set_xlim(0, 108)
        ax.tick_params(colors="#3e6ea8", labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.xaxis.grid(True, color="#c6d9f5", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, framealpha=0, labelcolor="#4c77ae")
        for bar, prob in zip(bars, all_probs):
            ax.text(bar.get_width() + 0.8,
                    bar.get_y() + bar.get_height() / 2,
                    f"{prob*100:.1f}%", va="center", fontsize=8, color="#2b65ac")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown('<hr class="soft-hr">', unsafe_allow_html=True)

    # Species photos
    st.markdown('<p class="section-head">Species Photos</p>',
                unsafe_allow_html=True)
    if photos:
        pcols = st.columns(len(photos), gap="medium")
        for col, url in zip(pcols, photos):
            img_url = url[0] if isinstance(url, tuple) else url
            if not img_url:
                continue
            with col:
                local = get_cached_photo(img_url)   # Download once and cache locally
                if local:
                    st.image(local, width="stretch") # Serve cached local file in UI
                else:
                    st.warning("Image unavailable.")
    else:
        st.info("No photos available.")

    st.markdown('<hr class="soft-hr">', unsafe_allow_html=True)

    # Range map
    st.markdown('<p class="section-head">Where This Bird Is Found</p>',
                unsafe_allow_html=True)

    if taxon_id:
        inat_points = get_inat_observation_points(taxon_id)
        inat_points_json = json.dumps(inat_points)
        leaflet_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{ background:#edf5ff; font-family:'Jost',sans-serif; }}
    #map {{
      height:430px; width:100%; border-radius:16px;
      border:1.5px solid #b8d0f0;
      box-shadow:0 4px 20px rgba(60,120,190,0.12);
    }}
    .legend {{
      background:rgba(245,250,255,0.95); border:1px solid #b8d0f0;
      border-radius:10px; padding:8px 12px;
      font-size:11px; color:#2b5f9f; line-height:1.8;
    }}
    .dot {{
      display:inline-block; width:10px; height:10px;
      border-radius:50%; margin-right:5px; vertical-align:middle;
    }}
  </style>
</head>
<body>
<div id="map"></div>
<script>
  var map = L.map('map', {{ center:[22,82], zoom:4 }});
  L.tileLayer(
    'https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}{{r}}.png',
    {{ attribution:'&copy; OpenStreetMap &copy; CARTO', subdomains:'abcd', maxZoom:19 }}
  ).addTo(map);

  var points = {inat_points_json};
  var sightings = L.layerGroup().addTo(map);
  points.forEach(function(p) {{
    L.circleMarker([p[0], p[1]], {{
      radius:4,
      color:'#2f6fbe',
      fillColor:'#5c96dc',
      fillOpacity:0.65,
      weight:1
    }}).addTo(sightings);
  }});

  if (points.length > 0) {{
    var bounds = L.latLngBounds(points.map(function(p) {{ return [p[0], p[1]]; }}));
    map.fitBounds(bounds.pad(0.18));
  }}

  var goaIcon = L.divIcon({{
    html:'<div style="width:14px;height:14px;background:#c84848;border-radius:50%;border:2.5px solid white;box-shadow:0 2px 6px rgba(0,0,0,0.3);"></div>',
    iconSize:[14,14], iconAnchor:[7,7], className:''
  }});
  L.marker([15.4909,73.8278],{{icon:goaIcon}})
   .addTo(map)
   .bindPopup('<b style="color:#1f4f8b;font-family:serif;">Goa, India</b>')
   .openPopup();

  var legend = L.control({{position:'bottomright'}});
  legend.onAdd = function() {{
    var div = L.DomUtil.create('div','legend');
    div.innerHTML =
      '<span class="dot" style="background:#2f6fbe;opacity:0.8;"></span> Research sightings (iNaturalist)<br>' +
      '<span class="dot" style="background:#c84848;"></span> Goa, India';
    return div;
  }};
  legend.addTo(map);
</script>
</body>
</html>"""
        components.html(leaflet_html, height=450, scrolling=False)
        if inat_points:
            st.caption(
                f"Blue dots = verified *{bird_name}* observations from iNaturalist &nbsp;|&nbsp; "
                f"[Full species page](https://www.inaturalist.org/taxa/{taxon_id})"
            )
        else:
            st.caption(
                f"No recent iNaturalist points returned for *{bird_name}*. "
                f"[Open species page](https://www.inaturalist.org/taxa/{taxon_id})"
            )
    else:
        st.info("Range map not available.")

    # Footer
    st.markdown("""
    <div class="footer">
        EfficientNetB0 | Xeno-canto recordings | Wikipedia photos | iNaturalist maps |
        10 species | Goa, India
    </div>
    """, unsafe_allow_html=True)
