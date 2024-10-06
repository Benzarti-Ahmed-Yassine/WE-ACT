import numpy as np
import os
import cv2
from deepface import DeepFace
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_bcrypt import Bcrypt
from flask_wtf import FlaskForm
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from wtforms import StringField, PasswordField, EmailField, SubmitField
from wtforms.validators import DataRequired, Email
from datetime import timedelta
from scipy.stats import entropy
from flask_sqlalchemy import SQLAlchemy
import base64

# --- App Configuration ---
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = os.urandom(24)
app.permanent_session_lifetime = timedelta(minutes=5)
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Initialize security libraries
bcrypt = Bcrypt(app)
limiter = Limiter(app, key_func=get_remote_address)
db = SQLAlchemy(app)

# Define directories
COMPARISON_DIR = 'photo'
CAPTURE_DIR = 'static/captured'
os.makedirs(CAPTURE_DIR, exist_ok=True)

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- User Model ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))

    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)

# --- Forms ---
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = EmailField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Register')

# --- Utility Functions ---
def capture_image():
    """Capture an image from the camera with error handling and multiple attempts."""
    for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
        try:
            cap = cv2.VideoCapture(0 + backend)
            if not cap.isOpened():
                continue
            
            # Wait a bit for the camera to initialize
            for _ in range(5):
                ret, frame = cap.read()
                if ret and frame is not None:
                    cap.release()
                    return frame
            
            cap.release()
        except Exception as e:
            print(f"Failed attempt with backend {backend}: {str(e)}")
    
    raise RuntimeError("Unable to capture image from any camera backend")

def detect_ai_generated(image):
    try:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        noise = cv2.Laplacian(gray_img, cv2.CV_64F).var()
        hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
        hist_entropy = entropy(hist.flatten())
        edges = cv2.Canny(gray_img, 100, 200)
        edge_percentage = np.count_nonzero(edges) / edges.size

        score = (noise < 100) + (hist_entropy < 5) + (edge_percentage < 0.05)
        is_ai_generated = score >= 2
        confidence = (score / 3) * 100

        return {
            "is_ai_generated": is_ai_generated,
            "confidence": round(confidence, 2),
            "classification": "AI Generated" if is_ai_generated else "Real",
            "details": {
                "noise_level": round(noise, 2),
                "histogram_entropy": round(float(hist_entropy), 2),
                "edge_percentage": round(edge_percentage * 100, 2)
            }
        }
    except Exception as e:
        return {
            "is_ai_generated": None,
            "confidence": 0,
            "classification": "Error in detection",
            "error": str(e)
        }

def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

def compare_faces(captured_image, comparison_dir):
    results = []
    seen_files = set()
    faces_captured = detect_faces(captured_image)

    if len(faces_captured) == 0:
        return []
    
    for (x, y, w, h) in faces_captured:
        face_captured = captured_image[y:y+h, x:x+w]

        for filename in os.listdir(comparison_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')) and filename not in seen_files:
                img_path = os.path.join(comparison_dir, filename)
                comparison_image = cv2.imread(img_path)

                if comparison_image is None:
                    continue

                faces_comparison = detect_faces(comparison_image)
                if len(faces_comparison) == 0:
                    continue

                for (x_comp, y_comp, w_comp, h_comp) in faces_comparison:
                    face_comparison = comparison_image[y_comp:y_comp+h_comp, x_comp:x_comp+w_comp]
                    try:
                        result = DeepFace.verify(face_captured, face_comparison, enforce_detection=False)
                        if result['verified']:
                            ai_generated_result = detect_ai_generated(comparison_image)
                            results.append({
                                'filename': filename,
                                'compared_image_path': img_path,
                                'classification': "Real" if not ai_generated_result['is_ai_generated'] else "AI Generated"
                            })
                            seen_files.add(filename)
                    except Exception as e:
                        print(f"Error comparing with {filename}: {e}")

    return results

# --- Routes ---
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        if User.query.filter_by(username=form.username.data).first():
            return render_template('register.html', form=form, error="Username already exists.")
        if User.query.filter_by(email=form.email.data).first():
            return render_template('register.html', form=form, error="Email already in use.")
        
        new_user = User(username=form.username.data, email=form.email.data)
        new_user.set_password(form.password.data)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.check_password(form.password.data):
            session['logged_in'] = True
            # Removed storing username in session
            return redirect(url_for('index'))
        return render_template('login.html', form=form, error="Invalid credentials.")
    return render_template('login.html', form=form)

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    results = ai_results = captured_filename = None

    if request.method == 'POST':
        try:
            frame = capture_image()
            captured_filename = os.path.join(CAPTURE_DIR, 'capture.jpg')
            cv2.imwrite(captured_filename, frame)

            faces = detect_faces(frame)
            if len(faces) == 0:
                return render_template('index.html', error="No faces detected.")

            results = compare_faces(frame, COMPARISON_DIR)
            ai_results = detect_ai_generated(frame)

        except RuntimeError as e:
            return render_template('index.html', error=str(e))  # Affiche les erreurs d'accès à la caméra
        except Exception as e:
            return render_template('index.html', error=f"Error: {str(e)}")

    return render_template('index.html', 
                          results=results, 
                          ai_results=ai_results, 
                          captured_filename=captured_filename)

@app.route('/upload_image', methods=['POST'])
def upload_image():
    try:
        data = request.get_json()
        image_data = data.get('image', '').split(',')[1]
        image = base64.b64decode(image_data)
        
        nparr = np.frombuffer(image, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Invalid image data"}), 400
        
        captured_filename = os.path.join(CAPTURE_DIR, 'capture.jpg')
        cv2.imwrite(captured_filename, frame)
        
        ai_results = detect_ai_generated(frame)
        results = compare_faces(frame, COMPARISON_DIR)
        
        return jsonify({"ai_results": ai_results, "results": results}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

# --- Main ---
if __name__ == '__main__':
    db.create_all()  # Crée les tables de la base de données
    app.run(debug=True)
