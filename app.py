import os
import uuid
import secrets
import io
import base64
from datetime import timedelta
from functools import wraps
import re
import bcrypt
import pymongo



from dotenv import load_dotenv
load_dotenv()
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_session import Session
from werkzeug.utils import secure_filename

# ── Force CPU for TensorFlow ─────────────────────────────────────────────
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

try:
    import tensorflow as tf
    import numpy as np
    import h5py
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("[WARN] TensorFlow or dependencies not installed.")

# ── MongoDB Connection ──────────────────────────────────────────────────
mongo_client = pymongo.MongoClient(os.environ["MONGO_URI"])
db = mongo_client["brain_app"]
users = db["users"]

# ── Flask setup ─────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))

app.config.update(
    SESSION_TYPE="filesystem",
    SESSION_FILE_DIR="./flask_session",
    SESSION_PERMANENT=False,
    PERMANENT_SESSION_LIFETIME=timedelta(hours=2),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=False,
    MAX_CONTENT_LENGTH=50 * 1024 * 1024,
    UPLOAD_FOLDER=os.path.join("static", "uploads"),
)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs("./flask_session", exist_ok=True)
Session(app)

ALLOWED_EXTENSIONS = {"h5", "png", "jpg", "jpeg"}

# ── Helpers ─────────────────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed)

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# ── Load Models ──────────────────────────────────────────────────────
_seg_model = None
_grad_model = None

def get_models():
    global _seg_model, _grad_model
    if ML_AVAILABLE:
        if _seg_model is None:
            path = os.path.join("models", "best_brats_model.h5")
            if os.path.exists(path):
                _seg_model = tf.keras.models.load_model(path, compile=False)
        if _grad_model is None:
            path = os.path.join("models", "brats_grading_model.h5")
            if os.path.exists(path):
                _grad_model = tf.keras.models.load_model(path, compile=False)
    return _seg_model, _grad_model

# ── Clinical Mapping Logic ──────────────────────────────────────────
def get_clinical_insights(score):
    if score > 0.5:
        return {
            "stage": "WHO Grade IV (HGG)",
            "color": "#ef4444",
            "description": "Glioblastoma Multiforme (GBM). Highly aggressive and infiltrative.",
            "survival": "12–18 months (Median)",
            "treatment": "Standard Stupp Protocol: Surgical resection, followed by Radiotherapy + Temozolomide.",
            "markers": "IDH-wildtype, MGMT Promoter Methylation (Testing Recommended)",
            "insight": "Immediate neurosurgical consultation required. High mitotic activity detected.",
            "probs": [
                {"name": "Grade IV (HGG)", "prob": int(score * 100), "color": "#ef4444"},
                {"name": "Grade II (LGG)", "prob": int((1 - score) * 100), "color": "#f59e0b"}
            ]
        }
    else:
        return {
            "stage": "WHO Grade II (LGG)",
            "color": "#f59e0b",
            "description": "Low-Grade Glioma (Astrocytoma/Oligodendroglioma). Slow-growing but infiltrative.",
            "survival": "5–10+ years",
            "treatment": "Maximal safe resection. 'Watchful waiting' or chemotherapy depending on molecular markers.",
            "markers": "IDH1/2 Mutation likely, 1p/19q codeletion (Testing Recommended)",
            "insight": "Monitor for transformation to high-grade. Regular follow-up MRIs every 3-6 months.",
            "probs": [
                {"name": "Grade II (LGG)", "prob": int((1 - score) * 100), "color": "#10b981"},
                {"name": "Grade IV (HGG)", "prob": int(score * 100), "color": "#ef4444"}
            ]
        }

# ── Prediction ──────────────────────────────────────────────────────
def run_prediction(filepath):
    seg_model, grad_model = get_models()
    if seg_model is None or grad_model is None:
        raise RuntimeError("AI Models not fully loaded.")

    ext = filepath.rsplit(".", 1)[-1].lower()

    # --- Preprocessing ---
    if ext == "h5":
        with h5py.File(filepath, "r") as f:
            image = f["image"][:]
            ground_truth = f["mask"][:]

        image = image.astype(np.float32)
        ground_truth = (ground_truth > 0).astype(np.uint8)
    else:
        ground_truth = None
        # 1. Load and Resize
        img = Image.open(filepath).convert("L").resize((240, 240))
        arr = np.array(img).astype(np.float32)
        
        # 2. FIX: Min-Max Normalization for JPG/PNG
        # This aligns the pixel scale with the H5 MRI data
        if np.max(arr) > 0:
            arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)
            
        # 3. Stack to 4 channels
        image = np.stack([arr, arr, arr, arr], axis=-1)

    input_tensor = np.expand_dims(image, axis=0)

    # 1. Segmentation
    seg_pred = seg_model.predict(input_tensor, verbose=0)[0]
    
    # ADJUSTED THRESHOLD for broader detection sensitivity
    mask = (seg_pred > 0.3).astype(np.uint8) 
    tumor_fraction = float(mask.max(axis=-1).mean())
    
    # Calculate actual Dice score when ground truth is available
    dice_score = None

    if ground_truth is not None:
        predicted_mask = mask.max(axis=-1)

    # Make sure ground truth is 2D
        if ground_truth.ndim == 3:
            ground_truth = ground_truth.max(axis=-1)

        intersection = np.sum(predicted_mask * ground_truth)
        predicted_pixels = np.sum(predicted_mask)
        ground_truth_pixels = np.sum(ground_truth)

        if predicted_pixels == 0 and ground_truth_pixels == 0:
            dice_score = 1.0
        else:
            dice_score = (
            (2.0 * intersection) /
            (predicted_pixels + ground_truth_pixels)
        )   
    # 2. Grading (Logic for results display)
    if tumor_fraction > 0.00005: # Detection limit
        grade_score = float(grad_model.predict(input_tensor, verbose=0)[0][0])
        insights = get_clinical_insights(grade_score)
        result_text = "Tumor Detected"
    else:
        insights = {
            "stage": "N/A", "color": "#10b981", "description": "No significant tumor mass detected.",
            "survival": "Normal", "treatment": "N/A", "markers": "N/A", 
            "insight": "Routine screening recommended.", "probs": []
        }
        result_text = "No Tumor Detected"

    # --- Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor='#050810')
    axes[0].imshow(image[:, :, 0], cmap="gray")
    axes[0].set_title("Input MRI", color='white')
    axes[0].axis("off")

    axes[1].imshow(image[:, :, 0], cmap="gray")
    if tumor_fraction > 0:
        axes[1].imshow(mask.max(axis=-1), cmap="jet", alpha=0.5)
    axes[1].set_title("Prediction Overlay", color='white')
    axes[1].axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)

    return {
        "result": result_text,
        "tumor_fraction": round(tumor_fraction * 100, 2),
        "plot_b64": img_b64,
        "stage": insights["stage"],
        "color": insights["color"],
        "description": insights["description"],
        "survival": insights["survival"],
        "treatment": insights["treatment"],
        "markers": insights["markers"],
        "insight": insights["insight"],
        "grade_probs": insights["probs"],
        "staging_method": "ML Classifier",
        "dice": round(float(dice_score), 2) if dice_score is not None else None
    }

# ── Routes ─────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        try:
            # Check if username already exists
            if users.find_one({"username": username}):
                flash("Username already exists")
                return redirect(url_for("register"))

            # Hash password
            password_hash = hash_password(password)

            # Create user
            users.insert_one({
                "username": username,
                "email": email,
                "password_hash": password_hash
            })

            flash("Registered successfully")
            return redirect(url_for("login"))

        except Exception as e:
            print("REGISTRATION ERROR:", repr(e))
            flash("Registration failed")
            return redirect(url_for("register"))

    return render_template("register.html")
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username").lower()
        password = request.form.get("password")
        user = users.find_one({"username": username})
        if user and check_password(password, user["password_hash"]):
            session["user"] = username
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials")
    return render_template("login.html")

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", user=session["user"])

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    file = request.files.get("file")
    if file and allowed_file(file.filename):
        filename = secure_filename(str(uuid.uuid4()) + "_" + file.filename)
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(path)
        try:
            result = run_prediction(path)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)})
    return jsonify({"error": "Invalid file format"})

if __name__ == "__main__":
    if __name__ == "__main__":
        app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))