import os
import io
import time
import glob
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont, ExifTags
import streamlit as st
from deepface import DeepFace
import cv2

# =============================
# ENHANCED CONFIG
# =============================
DATA_DIR = "data"
USERS_DIR = os.path.join(DATA_DIR, "users")
EMB_DIR = os.path.join(DATA_DIR, "embeddings")
USERS_CSV = os.path.join(DATA_DIR, "users.csv")
ATT_CSV = os.path.join(DATA_DIR, "attendance.csv")

# Enhanced model settings
MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "retinaface"
FALLBACK_DETECTORS = ["mtcnn", "opencv", "ssd"]
THRESHOLD = 0.40
MIN_SHOTS = 1
RECOMMENDED_SHOTS = 3

# New settings for enhanced detection
MAX_IMAGE_SIZE = 1200
MIN_FACE_SIZE = 40
ENABLE_FACE_ENHANCEMENT = True
ENABLE_DUPLICATE_REMOVAL = True

# =============================
# iOS THEME STYLING
# =============================
def apply_ios_theme():
    st.markdown("""
    <style>
    /* Import SF Pro Display font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global iOS styling */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* iOS Card styling */
    .ios-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Header styling */
    .ios-header {
        text-align: center;
        padding: 20px 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        margin-bottom: 30px;
        color: white;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .ios-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .ios-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 8px 0 0 0;
        font-weight: 400;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 8px;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        color: #666;
        font-weight: 500;
        padding: 12px 20px;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.9);
        border: 2px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        padding: 12px 16px;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.7) 100%);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Success/Error styling */
    .stSuccess {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border-radius: 12px;
        border: none;
    }
    
    .stError {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        border-radius: 12px;
        border: none;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #feca57 0%, #ff9ff3 100%);
        border-radius: 12px;
        border: none;
    }
    
    /* Image styling */
    .stImage > img {
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    /* Radio button styling */
    .stRadio > div {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        padding: 16px;
        backdrop-filter: blur(10px);
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        border: 2px dashed rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: #667eea;
        background: rgba(255, 255, 255, 1);
    }
    
    /* Camera input styling */
    .stCameraInput > div {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        border: 2px solid rgba(102, 126, 234, 0.2);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    </style>
    """, unsafe_allow_html=True)

def create_ios_header():
    st.markdown("""
    <div class="ios-header">
        <h1>📱 Smart Attendance</h1>
        <p>AI-Powered Face Recognition System</p>
    </div>
    """, unsafe_allow_html=True)

def create_metric_card(title, value, icon="📊"):
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 2rem; margin-bottom: 8px;">{icon}</div>
        <div style="font-size: 2rem; font-weight: 700; color: #667eea; margin-bottom: 4px;">{value}</div>
        <div style="font-size: 0.9rem; color: #666; font-weight: 500;">{title}</div>
    </div>
    """, unsafe_allow_html=True)

def create_section_header(title, subtitle="", icon=""):
    st.markdown(f"""
    <div style="text-align: center; padding: 20px 0;">
        <div style="font-size: 2rem; margin-bottom: 8px;">{icon}</div>
        <h2 style="color: #667eea; font-weight: 700; margin: 0;">{title}</h2>
        {f'<p style="color: #666; margin: 8px 0 0 0;">{subtitle}</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)

# =============================
# File helpers
# =============================
def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(USERS_DIR, exist_ok=True)
    os.makedirs(EMB_DIR, exist_ok=True)
    if not os.path.exists(USERS_CSV):
        pd.DataFrame(columns=["identifier", "name", "embedding_path", "image_path", "created_at"]).to_csv(USERS_CSV, index=False)
    if not os.path.exists(ATT_CSV):
        pd.DataFrame(columns=["timestamp", "identifier", "name", "status", "session_id"]).to_csv(ATT_CSV, index=False)

def load_users_df():
    ensure_dirs()
    return pd.read_csv(USERS_CSV)

def save_users_df(df):
    df.to_csv(USERS_CSV, index=False)

def load_att_df():
    ensure_dirs()
    return pd.read_csv(ATT_CSV)

def append_multiple_attendance(attendance_list, session_id):
    """Bulk append multiple attendance records"""
    if not attendance_list:
        return
    
    df = load_att_df()
    timestamp = datetime.utcnow().isoformat()
    
    rows = []
    for identifier, name in attendance_list:
        rows.append({
            "timestamp": timestamp,
            "identifier": identifier,
            "name": name,
            "status": "present",
            "session_id": session_id
        })
    
    new_df = pd.DataFrame(rows)
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(ATT_CSV, index=False)

# =============================
# ENHANCED IMAGE PROCESSING
# =============================
def fix_image_orientation(image):
    """Fix image orientation based on EXIF data"""
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        
        exif = image._getexif()
        if exif is not None:
            orientation_value = exif.get(orientation, 1)
            
            if orientation_value == 3:
                image = image.rotate(180, expand=True)
            elif orientation_value == 6:
                image = image.rotate(270, expand=True)
            elif orientation_value == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, TypeError):
        pass
    
    return image

def preprocess_image_for_detection(image_rgb):
    """Enhanced image preprocessing for better face detection"""
    pil_image = Image.fromarray(image_rgb)
    width, height = pil_image.size
    
    # Resize if image is too large
    if max(width, height) > MAX_IMAGE_SIZE:
        ratio = MAX_IMAGE_SIZE / max(width, height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        st.info(f"Resized image from {width}x{height} to {new_width}x{new_height} for better processing")
    
    processed_rgb = np.array(pil_image)
    
    # Enhance brightness if image appears dim
    if np.mean(processed_rgb) < 100:
        processed_rgb = np.clip(processed_rgb * 1.2 + 20, 0, 255).astype(np.uint8)
        st.info("Applied brightness enhancement to improve detection")
    
    return processed_rgb

def process_uploaded_image(uploaded_file):
    """Enhanced image processing with better error handling and orientation fixes"""
    try:
        # Read the uploaded file
        if hasattr(uploaded_file, 'getvalue'):
            # Camera input
            bytes_data = uploaded_file.getvalue()
            image = Image.open(io.BytesIO(bytes_data))
        else:
            # File uploader
            image = Image.open(uploaded_file)
        
        # Fix orientation based on EXIF data
        image = fix_image_orientation(image)
        
        # Convert to RGB
        image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Ensure proper format for DeepFace
        if img_array.dtype != np.uint8:
            img_array = img_array.astype(np.uint8)
        
        # Apply preprocessing
        img_array = preprocess_image_for_detection(img_array)
        
        # Validate image
        if img_array.size == 0:
            raise ValueError("Processed image is empty")
        
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            raise ValueError("Image must be RGB format")
            
        return img_array
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.error("Please try with a different image or check the file format")
        return None

def debug_image_info(image_rgb):
    """Display detailed image information for debugging"""
    st.markdown("### 🔍 Image Debug Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Width", f"{image_rgb.shape[1]}px")
        st.metric("Height", f"{image_rgb.shape[0]}px")
    
    with col2:
        st.metric("Channels", image_rgb.shape[2])
        st.metric("Data Type", str(image_rgb.dtype))
    
    with col3:
        st.metric("Min Pixel", int(np.min(image_rgb)))
        st.metric("Max Pixel", int(np.max(image_rgb)))
    
    # Show brightness histogram
    brightness = np.mean(image_rgb, axis=2)
    st.write(f"**Average Brightness**: {np.mean(brightness):.1f}")
    
    if np.mean(brightness) < 80:
        st.warning("⚠️ Image appears very dark - this may affect face detection")
    elif np.mean(brightness) > 200:
        st.warning("⚠️ Image appears very bright - this may affect face detection")
    
    # Show a small preview
    st.image(image_rgb, caption="Processed Image", width=300)

@st.cache_resource(show_spinner=False)
def warmup_models():
    """Load model/detector once for speed."""
    try:
        # Test with a small dummy image
        dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        DeepFace.represent(
            dummy_img,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False
        )
        st.success("🚀 AI Models loaded successfully")
    except Exception as e:
        st.warning(f"⚠️ Model warmup failed: {e}")
    return True

def compute_single_embedding(image_rgb, enforce=True):
    """Compute embedding for single face (used in registration)"""
    try:
        rep = DeepFace.represent(
            image_rgb,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=enforce
        )
        
        if isinstance(rep, list) and len(rep) > 0:
            emb = np.array(rep[0]["embedding"], dtype=np.float32)
        else:
            emb = np.array(rep["embedding"], dtype=np.float32)
            
        # Normalize
        n = np.linalg.norm(emb)
        if n > 0:
            emb = emb / n
        return emb.astype(np.float32)
    except Exception as e:
        st.error(f"Embedding computation failed: {e}")
        raise e

# =============================
# ENHANCED FACE DETECTION
# =============================
def compute_multiple_embeddings_enhanced(image_rgb):
    """Enhanced multi-face embedding computation with better error handling"""
    embeddings = []
    face_areas = []
    
    st.write(f"🔍 **Image Analysis**: {image_rgb.shape[1]}x{image_rgb.shape[0]} pixels")
    
    # Try different DeepFace detector backends
    detector_backends = ['retinaface', 'mtcnn', 'opencv', 'ssd']
    
    for detector in detector_backends:
        try:
            st.write(f"🔄 Trying {detector} detector...")
            
            representations = DeepFace.represent(
                image_rgb,
                model_name=MODEL_NAME,
                detector_backend=detector,
                enforce_detection=False
            )
            
            if representations and len(representations) > 0:
                st.success(f"✅ {detector} detected {len(representations)} face(s)")
                
                for i, rep in enumerate(representations):
                    try:
                        # Extract embedding
                        emb = np.array(rep["embedding"], dtype=np.float32)
                        
                        # Normalize embedding
                        norm = np.linalg.norm(emb)
                        if norm > 0:
                            emb = emb / norm
                        
                        embeddings.append(emb)
                        
                        # Extract face area with better error handling
                        face_area = rep.get("facial_area", {})
                        
                        # Ensure all required keys exist
                        required_keys = ['x', 'y', 'w', 'h']
                        if all(key in face_area for key in required_keys):
                            face_areas.append(face_area)
                        else:
                            # Create default face area if detection succeeded but area info is missing
                            face_areas.append({
                                'x': 0, 'y': 0, 
                                'w': image_rgb.shape[1] // 4, 
                                'h': image_rgb.shape[0] // 4
                            })
                        
                    except Exception as e:
                        st.warning(f"⚠️ Failed to process face {i+1} from {detector}: {str(e)}")
                        continue
                
                if len(embeddings) > 0:
                    st.success(f"🎯 Successfully processed {len(embeddings)} face(s)")
                    return embeddings, face_areas
                    
            else:
                st.write(f"❌ {detector}: No faces detected")
                
        except Exception as e:
            st.write(f"❌ {detector} failed: {str(e)}")
            continue
    
    # Enhanced OpenCV fallback
    st.write("🔄 Trying enhanced OpenCV fallback detection...")
    try:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization to improve contrast
        gray = cv2.equalizeHist(gray)
        
        # Try multiple cascade classifiers
        cascade_files = [
            'haarcascade_frontalface_default.xml',
            'haarcascade_frontalface_alt.xml',
            'haarcascade_frontalface_alt2.xml'
        ]
        
        detected_faces = []
        for cascade_file in cascade_files:
            try:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_file)
                
                # Try different scale factors
                for scale_factor in [1.1, 1.2, 1.3]:
                    faces = face_cascade.detectMultiScale(
                        gray, 
                        scaleFactor=scale_factor, 
                        minNeighbors=3,
                        minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                    
                    if len(faces) > 0:
                        st.write(f"✅ OpenCV ({cascade_file}, scale={scale_factor}): Found {len(faces)} face(s)")
                        detected_faces = faces
                        break
                
                if len(detected_faces) > 0:
                    break
                    
            except Exception as e:
                st.write(f"❌ {cascade_file} failed: {str(e)}")
                continue
        
        if len(detected_faces) == 0:
            st.error("❌ No faces detected with any method")
            return [], []
        
        # Process detected faces
        for i, (x, y, w, h) in enumerate(detected_faces):
            try:
                # Add padding around detected face
                padding = max(20, min(w, h) // 10)
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(image_rgb.shape[1], x + w + padding)
                y2 = min(image_rgb.shape[0], y + h + padding)
                
                face_img = image_rgb[y1:y2, x1:x2]
                
                if face_img.size == 0:
                    st.warning(f"⚠️ Face {i+1}: Empty face region, skipping")
                    continue
                
                # Resize face if too small
                if min(face_img.shape[:2]) < 50:
                    scale = 100 / min(face_img.shape[:2])
                    new_h, new_w = int(face_img.shape[0] * scale), int(face_img.shape[1] * scale)
                    face_img = cv2.resize(face_img, (new_w, new_h))
                
                # Get embedding for this face region
                rep = DeepFace.represent(
                    face_img,
                    model_name=MODEL_NAME,
                    detector_backend='opencv',
                    enforce_detection=False
                )
                
                if isinstance(rep, list) and len(rep) > 0:
                    emb = np.array(rep[0]["embedding"], dtype=np.float32)
                else:
                    emb = np.array(rep["embedding"], dtype=np.float32)
                
                # Normalize embedding
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                
                embeddings.append(emb)
                face_areas.append({
                    'x': int(x), 'y': int(y), 
                    'w': int(w), 'h': int(h)
                })
                
            except Exception as e:
                st.warning(f"⚠️ Failed to get embedding for OpenCV face {i+1}: {str(e)}")
                continue
        
        if len(embeddings) > 0:
            st.success(f"🎯 OpenCV method: Successfully processed {len(embeddings)} face(s)")
        
        return embeddings, face_areas
        
    except Exception as e:
        st.error(f"❌ All face detection methods failed: {str(e)}")
        return [], []

def cosine_distance_matrix(probe_emb, known_matrix):
    """Compute cosine distances between probe and all known embeddings"""
    if known_matrix.size == 0:
        return np.array([])
    sims = known_matrix @ probe_emb
    sims = np.clip(sims, -1.0, 1.0)
    return 1.0 - sims

@st.cache_data(show_spinner=False)
def load_all_embeddings():
    """Load all registered embeddings for fast batch matching"""
    users_df = load_users_df()
    ids, names, embs = [], [], []
    
    for _, row in users_df.iterrows():
        emb_path = row["embedding_path"]
        if not isinstance(emb_path, str) or not os.path.exists(emb_path):
            continue
        
        emb = np.load(emb_path).astype(np.float32)
        n = np.linalg.norm(emb)
        if n > 0:
            emb = emb / n
        
        embs.append(emb)
        ids.append(row["identifier"])
        names.append(row["name"])

    if len(embs) == 0:
        mat = np.zeros((0, 512), dtype=np.float32)
    else:
        max_dim = max(len(e) for e in embs)
        mat = np.vstack([e if len(e) == max_dim else np.pad(e, (0, max_dim - len(e))) for e in embs])

    return {
        "ids": np.array(ids, dtype=object),
        "names": np.array(names, dtype=object),
        "matrix": mat.astype(np.float32)
    }

def match_multiple_faces_enhanced(face_embeddings, known_data, threshold=None):
    """Enhanced face matching with better confidence scoring"""
    if threshold is None:
        threshold = THRESHOLD
    
    ids_arr, names_arr, emb_matrix = known_data["ids"], known_data["names"], known_data["matrix"]
    
    matched_students = []
    match_details = []
    
    for i, face_emb in enumerate(face_embeddings):
        if emb_matrix.shape[0] == 0:
            match_details.append({
                "face_id": i + 1,
                "student_id": "Unknown",
                "name": "No registered students",
                "confidence": 0.0,
                "distance": 1.0
            })
            continue
        
        # Compute distances to all known faces
        distances = cosine_distance_matrix(face_emb, emb_matrix)
        
        # Find best match
        best_idx = np.argmin(distances)
        best_dist = distances[best_idx]
        
        # Enhanced confidence calculation
        confidence = max(0.0, 1.0 - best_dist)
        
        # Apply threshold
        if best_dist <= threshold:
            student_id = ids_arr[best_idx]
            student_name = names_arr[best_idx]
            
            # Check for duplicates (same student detected multiple times)
            duplicate_found = False
            for existing_student_id, existing_name in matched_students:
                if existing_student_id == student_id:
                    duplicate_found = True
                    break
            
            if not duplicate_found:
                matched_students.append((student_id, student_name))
            
            match_details.append({
                "face_id": i + 1,
                "student_id": student_id,
                "name": student_name,
                "confidence": round(confidence, 3),
                "distance": round(best_dist, 3),
                "status": "Duplicate" if duplicate_found else "Matched"
            })
        else:
            # Find closest match for reference (even if above threshold)
            closest_student = names_arr[best_idx] if len(names_arr) > 0 else "None"
            
            match_details.append({
                "face_id": i + 1,
                "student_id": "Unknown",
                "name": f"Unknown (closest: {closest_student})",
                "confidence": round(confidence, 3),
                "distance": round(best_dist, 3),
                "status": "Below threshold"
            })
    
    return matched_students, match_details

def visualize_detected_faces(image_rgb, face_areas, match_details):
    """Draw bounding boxes and labels on detected faces"""
    img_pil = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    for i, (face_area, match) in enumerate(zip(face_areas, match_details)):
        if not face_area:
            continue
            
        # Extract coordinates
        x, y, w, h = face_area.get('x', 0), face_area.get('y', 0), face_area.get('w', 50), face_area.get('h', 50)
        
        # Choose color based on match status
        color = '#4CAF50' if match['student_id'] != "Unknown" else '#F44336'
        
        # Draw bounding box
        draw.rectangle([x, y, x + w, y + h], outline=color, width=4)
        
        # Prepare label
        if match['student_id'] != "Unknown":
            label = f"{match['name']}\n{match['confidence']:.2f}"
        else:
            label = f"Unknown\n{match['confidence']:.2f}"
        
        # Draw label background
        if font:
            bbox = draw.textbbox((x, y - 40), label, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x, y - 40), label, fill='white', font=font)
        else:
            draw.text((x, y - 30), label, fill=color)
    
    return np.array(img_pil)

def invalidate_embedding_cache():
    load_all_embeddings.clear()

# =============================
# STREAMLIT UI WITH iOS THEME
# =============================
st.set_page_config(
    page_title="Smart Attendance System", 
    layout="wide",
    page_icon="📱",
    initial_sidebar_state="collapsed"
)

# Apply iOS theme
apply_ios_theme()

# Create iOS-style header
create_ios_header()

# Initialize models and directories
warmup_models()
ensure_dirs()

if "reg_shots" not in st.session_state:
    st.session_state.reg_shots = []

# Create iOS-style tabs
tab_reg, tab_multi_mark, tab_logs, tab_manage = st.tabs([
    "👤 Register Student", 
    "📸 Class Attendance", 
    "📊 Attendance Logs", 
    "⚙️ Manage Data"
])

# =============================
# REGISTER STUDENT TAB
# =============================
with tab_reg:
    create_section_header("Student Registration", "Add new students to the system", "👤")
    
    st.markdown('<div class="ios-card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📝 Student Information")
        name = st.text_input("Student Full Name", placeholder="Enter full name")
        identifier = st.text_input("Student ID / Roll Number", placeholder="Enter student ID")
        
        st.markdown(f"### 📷 Photo Capture")
        st.info(f"📸 Capture **{RECOMMENDED_SHOTS}** clear face shots (minimum {MIN_SHOTS})")
        
        shot = st.camera_input("Registration Photo")
        
        col_add, col_clear = st.columns(2)
        with col_add:
            if st.button("➕ Add Shot", use_container_width=True):
                if shot:
                    rgb = process_uploaded_image(shot)
                    if rgb is not None:
                        st.session_state.reg_shots.append(rgb)
                        st.success(f"✅ Shot #{len(st.session_state.reg_shots)} added!")
                else:
                    st.warning("📷 Please capture a photo first.")
        
        with col_clear:
            if st.button("🗑️ Clear All Shots", use_container_width=True):
                st.session_state.reg_shots = []
                st.info("🧹 All shots cleared.")
    
    with col2:
        st.markdown("### 📸 Captured Photos")
        if len(st.session_state.reg_shots) > 0:
            st.success(f"**Photos Collected: {len(st.session_state.reg_shots)}**")
            for i, shot_img in enumerate(st.session_state.reg_shots[-3:], 1):
                st.image(shot_img, caption=f"Shot {len(st.session_state.reg_shots) - 3 + i}", width=200)
        else:
            st.info("📷 No photos captured yet")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Registration button
    if st.button("🎯 Register Student", type="primary", use_container_width=True):
        if not name or not identifier:
            st.error("❌ Please fill in both name and student ID.")
        elif len(st.session_state.reg_shots) < MIN_SHOTS:
            st.error(f"❌ Please capture at least {MIN_SHOTS} photo(s).")
        else:
            users_df = load_users_df()
            if (users_df["identifier"] == identifier).any():
                st.error("❌ Student ID already exists. Please use a different ID.")
            else:
                with st.spinner("🔄 Processing registration..."):
                    emb_list = []
                    ts = int(time.time())
                    
                    for i, rgb in enumerate(st.session_state.reg_shots, start=1):
                        img_path = os.path.join(USERS_DIR, f"{identifier}_{ts}_{i}.jpg")
                        Image.fromarray(rgb).save(img_path, "JPEG")
                        try:
                            emb = compute_single_embedding(rgb, enforce=True)
                            emb_list.append(emb)
                        except:
                            st.warning(f"⚠️ Shot #{i} skipped (no clear face detected).")
                    
                    if len(emb_list) == 0:
                        st.error("❌ No valid faces detected in any shot. Please try again with clearer photos.")
                    else:
                        avg_emb = np.mean(emb_list, axis=0)
                        n = np.linalg.norm(avg_emb)
                        if n > 0:
                            avg_emb = avg_emb / n
                        
                        emb_path = os.path.join(EMB_DIR, f"{identifier}.npy")
                        np.save(emb_path, avg_emb.astype(np.float32))
                        
                        new_row = pd.DataFrame([{
                            "identifier": identifier,
                            "name": name,
                            "embedding_path": emb_path,
                            "image_path": os.path.join(USERS_DIR, f"{identifier}_{ts}_1.jpg"),
                            "created_at": datetime.utcnow().isoformat()
                        }])
                        users_df = pd.concat([users_df, new_row], ignore_index=True)
                        save_users_df(users_df)
                        
                        st.session_state.reg_shots = []
                        invalidate_embedding_cache()
                        
                        st.success(f"🎉 Successfully registered {name} (ID: {identifier})")
                        st.balloons()

# =============================
# ENHANCED CLASS ATTENDANCE TAB
# =============================
with tab_multi_mark:
    create_section_header("Class Attendance", "Capture class photos for automatic attendance", "📸")
    
    cache = load_all_embeddings()
    ids_arr, names_arr, emb_matrix = cache["ids"], cache["names"], cache["matrix"]
    
    if emb_matrix.shape[0] == 0:
        st.warning("⚠️ No students registered yet. Please register students first.")
    else:
        # Show system status
        col1, col2, col3 = st.columns(3)
        with col1:
            create_metric_card("Registered Students", len(ids_arr), "👥")
        with col2:
            create_metric_card("AI Model", MODEL_NAME, "🤖")
        with col3:
            create_metric_card("Accuracy Threshold", f"{(1-THRESHOLD)*100:.0f}%", "🎯")
        
        st.markdown('<div class="ios-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📷 Capture Method")
            
            input_method = st.radio(
                "Choose input method:",
                ["📱 Live Camera", "📁 Upload Photo"],
                horizontal=True
            )
            
            # Add debug mode toggle
            debug_mode = st.checkbox("🔍 Enable Debug Mode (shows detailed detection info)")
            
            class_photo = None
            
            if input_method == "📱 Live Camera":
                class_photo = st.camera_input("Class Attendance Photo (Live)")
                st.caption("📸 For in-person classes - capture live photo")
            else:
                class_photo = st.file_uploader(
                    "Upload Class Photo",
                    type=['jpg', 'jpeg', 'png', 'webp', 'bmp'],
                    help="For online classes - upload screenshot from Zoom/Teams/Meet"
                )
                st.caption("💻 For online classes - upload screenshot from video calls")
                
                if class_photo:
                    st.info(f"📁 Uploaded: {class_photo.name} ({class_photo.size} bytes)")
            
            # Add enhanced detection settings
            st.markdown("### ⚙️ Detection Settings")
            
            # Allow user to adjust threshold for this session
            session_threshold = st.slider(
                "Recognition Threshold", 
                min_value=0.2, 
                max_value=0.8, 
                value=THRESHOLD,
                step=0.05,
                help="Lower = more lenient, Higher = more strict"
            )
            
            # Allow user to choose detection method priority
            detection_priority = st.selectbox(
                "Detection Method Priority",
                ["Auto (try all)", "RetinaFace first", "MTCNN first", "OpenCV first"],
                help="Choose which face detection method to prioritize"
            )
            
            if st.button("🔍 Detect & Match Faces", type="primary", use_container_width=True):
                if class_photo:
                    with st.spinner("🔄 Analyzing faces in the photo..."):
                        # Process image with enhanced function
                        rgb_image = process_uploaded_image(class_photo)
                        
                        if rgb_image is None:
                            st.error("❌ Failed to process the image. Please try again.")
                        else:
                            # Show debug information if enabled
                            if debug_mode:
                                debug_image_info(rgb_image)
                            
                            # Show processed image
                            st.image(rgb_image, caption="📸 Processed Image", width=300)
                            
                            # Extract all face embeddings with enhanced method
                            face_embeddings, face_areas = compute_multiple_embeddings_enhanced(rgb_image)
                            
                            if len(face_embeddings) == 0:
                                st.error("❌ No faces detected in the photo.")
                                st.markdown("""
                                **💡 Troubleshooting Tips:**
                                1. 💡 Ensure faces are clearly visible and well-lit
                                2. 📏 Faces should be at least 50x50 pixels in size
                                3. 🔄 Try enabling debug mode to see image details
                                4. 🎚️ Adjust the recognition threshold
                                5. 📸 Try a different photo with better quality
                                6. 🔧 Try different detection method priority
                                """)
                            else:
                                st.success(f"✅ Detected {len(face_embeddings)} faces in the photo")
                                
                                # Match faces against database with custom threshold
                                matched_students, match_details = match_multiple_faces_enhanced(
                                    face_embeddings, cache, session_threshold
                                )
                                
                                # Store results in session state
                                st.session_state['pending_attendance'] = matched_students
                                st.session_state['match_details'] = match_details
                                st.session_state['class_image'] = rgb_image
                                st.session_state['face_areas'] = face_areas
                                st.session_state['session_threshold'] = session_threshold
                else:
                    st.error("❌ Please capture or upload a photo first.")
        
        with col2:
            st.markdown("### 🎯 Detection Results")
            
            if 'match_details' in st.session_state:
                match_details = st.session_state['match_details']
                session_threshold = st.session_state.get('session_threshold', THRESHOLD)
                
                # Show threshold used
                st.info(f"🎚️ Using recognition threshold: {session_threshold:.2f}")
                
                # Show annotated image
                if 'class_image' in st.session_state and 'face_areas' in st.session_state:
                    annotated_img = visualize_detected_faces(
                        st.session_state['class_image'], 
                        st.session_state['face_areas'], 
                        match_details
                    )
                    st.image(annotated_img, caption="🎯 Detected Faces with Recognition", use_column_width=True)
                
                # Show detailed match results table with color coding
                df_results = pd.DataFrame(match_details)
                
                # Add color coding function for confidence
                def highlight_confidence(val):
                    if isinstance(val, (int, float)):
                        if val > 0.8:
                            return 'background-color: #d4edda'  # Green
                        elif val > 0.6:
                            return 'background-color: #fff3cd'  # Yellow
                        else:
                            return 'background-color: #f8d7da'  # Red
                    return ''
                
                # Style the dataframe if confidence column exists
                if 'confidence' in df_results.columns:
                    styled_df = df_results.style.applymap(highlight_confidence, subset=['confidence'])
                    st.dataframe(styled_df, use_container_width=True)
                else:
                    st.dataframe(df_results, use_container_width=True)
                
                # Show enhanced summary metrics
                recognized = len([m for m in match_details if m['student_id'] != 'Unknown'])
                unknown = len(match_details) - recognized
                high_conf = len([m for m in match_details if isinstance(m.get('confidence', 0), (int, float)) and m.get('confidence', 0) > 0.8])
                
                col_summary1, col_summary2, col_summary3, col_summary4 = st.columns(4)
                with col_summary1:
                    create_metric_card("Total Faces", len(match_details), "👥")
                with col_summary2:
                    create_metric_card("Recognized", recognized, "✅")
                with col_summary3:
                    create_metric_card("High Confidence", high_conf, "🎯")
                with col_summary4:
                    create_metric_card("Unknown", unknown, "❓")
                
                # Show confidence distribution in debug mode
                if debug_mode:
                    confidences = [m['confidence'] for m in match_details if isinstance(m['confidence'], (int, float))]
                    if confidences:
                        st.markdown("### 📊 Confidence Distribution")
                        conf_df = pd.DataFrame({'Confidence': confidences})
                        st.bar_chart(conf_df)
                
                # Detailed match information
                if recognized > 0:
                    st.markdown("### ✅ Recognized Students")
                    for detail in match_details:
                        if detail['student_id'] != "Unknown":
                            conf_val = detail.get('confidence', 0)
                            if isinstance(conf_val, (int, float)):
                                conf_color = "🟢" if conf_val > 0.8 else "🟡" if conf_val > 0.6 else "🟠"
                                st.write(f"{conf_color} **{detail['name']}** ({detail['student_id']}) - Confidence: {conf_val:.3f}")
                            else:
                                st.write(f"🔵 **{detail['name']}** ({detail['student_id']})")
                
                if unknown > 0:
                    st.markdown("### ❓ Unknown Faces")
                    for detail in match_details:
                        if detail['student_id'] == "Unknown":
                            dist_val = detail.get('distance', 1.0)
                            st.write(f"🔴 Face #{detail['face_id']} - Best match distance: {dist_val:.3f}")
                
                # Enhanced confirmation section
                if 'pending_attendance' in st.session_state and st.session_state['pending_attendance']:
                    session_id = f"class_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    st.markdown("---")
                    st.markdown("### ✅ Confirm Attendance")
                    
                    # Show current list with option to modify
                    st.write("**Students to mark as present:**")
                    
                    attendance_list = st.session_state['pending_attendance'].copy()
                    modified_list = []
                    
                    for i, (student_id, name) in enumerate(attendance_list):
                        col_name, col_remove = st.columns([4, 1])
                        with col_name:
                            st.write(f"• {name} ({student_id})")
                        with col_remove:
                            if st.button("Remove", key=f"remove_{i}"):
                                continue  # Skip this student
                        modified_list.append((student_id, name))
                    
                    # Update the list
                    st.session_state['pending_attendance'] = modified_list
                    
                    if len(modified_list) > 0:
                        st.success(f"Ready to mark {len(modified_list)} student(s) as present")
                        
                        if st.button("✅ Confirm & Mark Attendance", type="primary", use_container_width=True):
                            append_multiple_attendance(modified_list, session_id)
                            
                            # Clear session state
                            for key in ['pending_attendance', 'match_details', 'class_image', 'face_areas', 'session_threshold']:
                                if key in st.session_state:
                                    del st.session_state[key]
                            
                            st.success(f"🎉 Attendance marked for {len(modified_list)} students!")
                            st.balloons()
                            time.sleep(2)
                            st.rerun()
                    else:
                        st.warning("No students selected for attendance marking")
                
                # Add manual attendance option
                st.markdown("---")
                st.markdown("### ➕ Manual Attendance")
                with st.expander("Add student manually (if not detected)"):
                    manual_student = st.selectbox(
                        "Select student", 
                        options=[""] + [f"{name} ({id_})" for name, id_ in zip(names_arr, ids_arr)]
                    )
                    
                    if manual_student and st.button("Add Manual Attendance"):
                        # Parse student info
                        name = manual_student.split(" (")[0]
                        student_id = manual_student.split("(")[1].replace(")", "")
                        
                        # Add to pending list
                        if 'pending_attendance' not in st.session_state:
                            st.session_state['pending_attendance'] = []
                        
                        if (student_id, name) not in st.session_state['pending_attendance']:
                            st.session_state['pending_attendance'].append((student_id, name))
                            st.success(f"Added {name} to attendance list")
                            st.rerun()
                        else:
                            st.warning("Student already in attendance list")
        
        st.markdown('</div>', unsafe_allow_html=True)

# =============================
# ATTENDANCE LOGS TAB
# =============================
with tab_logs:
    create_section_header("Attendance Logs", "View and manage attendance records", "📊")
    
    logs = load_att_df()
    if logs.empty:
        st.info("📝 No attendance records yet.")
    else:
        st.markdown('<div class="ios-card">', unsafe_allow_html=True)
        
        # Add filters
        st.markdown("### 🔍 Filters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'session_id' in logs.columns:
                sessions = ['All'] + list(logs['session_id'].dropna().unique())
                selected_session = st.selectbox("📅 Filter by Session", sessions)
        
        with col2:
            if 'name' in logs.columns:
                students = ['All'] + list(logs['name'].dropna().unique())
                selected_student = st.selectbox("👤 Filter by Student", students)
        
        with col3:
            date_filter = st.date_input("📆 Filter by Date", value=None)
        
        # Apply filters
        filtered_logs = logs.copy()
        
        if 'selected_session' in locals() and selected_session != 'All':
            filtered_logs = filtered_logs[filtered_logs['session_id'] == selected_session]
        
        if 'selected_student' in locals() and selected_student != 'All':
            filtered_logs = filtered_logs[filtered_logs['name'] == selected_student]
        
        if date_filter:
            filtered_logs['date'] = pd.to_datetime(filtered_logs['timestamp']).dt.date
            filtered_logs = filtered_logs[filtered_logs['date'] == date_filter]
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display logs
        st.dataframe(
            filtered_logs.sort_values('timestamp', ascending=False),
            use_container_width=True
        )
        
        # Download button
        csv_data = filtered_logs.to_csv(index=False).encode()
        st.download_button(
            "📥 Download CSV",
            data=csv_data,
            file_name=f"attendance_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Statistics
        if not filtered_logs.empty:
            st.markdown("---")
            st.markdown("### 📈 Statistics")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                create_metric_card("Total Records", len(filtered_logs), "📝")
            with col2:
                create_metric_card("Unique Students", filtered_logs['name'].nunique(), "👥")
            with col3:
                if 'session_id' in filtered_logs.columns:
                    create_metric_card("Class Sessions", filtered_logs['session_id'].nunique(), "📅")

# =============================
# MANAGE DATA TAB
# =============================
with tab_manage:
    create_section_header("Manage System Data", "View and manage registered students", "⚙️")
    
    users_df = load_users_df()
    
    if users_df.empty:
        st.info("👥 No students registered yet.")
    else:
        st.markdown(f"### 👥 Registered Students ({len(users_df)})")
        
        # Display users in cards
        for _, user in users_df.iterrows():
            st.markdown('<div class="ios-card">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                if os.path.exists(user['image_path']):
                    img = Image.open(user['image_path'])
                    st.image(img, width=120)
                else:
                    st.write("📷 No image")
            
            with col2:
                st.markdown(f"**👤 {user['name']}**")
                st.write(f"🆔 ID: {user['identifier']}")
                st.write(f"📅 Registered: {user['created_at'][:10]}")
            
            with col3:
                if st.button(f"🗑️ Delete", key=f"del_{user['identifier']}", use_container_width=True):
                    # Delete files
                    for path in [user['embedding_path'], user['image_path']]:
                        if os.path.exists(path):
                            os.remove(path)
                    
                    # Update database
                    users_df = users_df[users_df['identifier'] != user['identifier']]
                    save_users_df(users_df)
                    invalidate_embedding_cache()
                    
                    st.success(f"✅ Deleted {user['name']}")
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ⚙️ System Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Embeddings", use_container_width=True):
            if not users_df.empty:
                with st.spinner("🔄 Recomputing embeddings..."):
                    cnt = 0
                    for _, user in users_df.iterrows():
                        img_paths = sorted(glob.glob(os.path.join(USERS_DIR, f"{user['identifier']}_*.jpg")))
                        emb_list = []
                        
                        for img_path in img_paths:
                            if os.path.exists(img_path):
                                rgb = np.array(Image.open(img_path).convert("RGB"))
                                try:
                                    emb = compute_single_embedding(rgb, enforce=True)
                                    emb_list.append(emb)
                                except:
                                    continue
                        
                        if emb_list:
                            avg_emb = np.mean(emb_list, axis=0)
                            n = np.linalg.norm(avg_emb)
                            if n > 0:
                                avg_emb = avg_emb / n
                            
                            emb_path = os.path.join(EMB_DIR, f"{user['identifier']}.npy")
                            np.save(emb_path, avg_emb.astype(np.float32))
                            cnt += 1
                    
                    invalidate_embedding_cache()
                    st.success(f"✅ Refreshed {cnt} embeddings")
            else:
                st.warning("⚠️ No users to refresh")
    
    with col2:
        if st.button("📊 System Stats", use_container_width=True):
            st.info(f"""
            **📈 System Statistics:**
            - 👥 Registered Students: {len(users_df)}
            - 📝 Total Attendance Records: {len(load_att_df())}
            - 🤖 AI Model: {MODEL_NAME}
            - 🎯 Recognition Threshold: {THRESHOLD}
            """)
    
    with col3:
        if st.button("🗑️ Delete All Data", use_container_width=True):
            if st.checkbox("⚠️ I understand this will delete everything"):
                import shutil
                try:
                    shutil.rmtree(DATA_DIR, ignore_errors=True)
                    ensure_dirs()
                    invalidate_embedding_cache()
                    st.success("✅ All data deleted successfully")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error deleting data: {e}")

# Footer with tips
st.markdown("---")
st.markdown("### 💡 Tips for Best Results")
st.markdown("""
- 💡 Use good lighting when taking class photos
- 👥 Ensure faces are clearly visible (not too far or blocked)  
- 📸 Register students with multiple clear face shots
- 💻 For online classes, take high-quality screenshots
- 🔄 If recognition is poor, try refreshing embeddings in Manage tab
- 🎚️ Adjust recognition threshold based on your needs (lower = more lenient)
- 🔍 Enable debug mode to troubleshoot detection issues
""")

# Add some spacing at the bottom
st.markdown("<br><br>", unsafe_allow_html=True)