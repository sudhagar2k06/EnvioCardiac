import os
import secrets
from datetime import datetime, date
from functools import wraps
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from werkzeug.security import check_password_hash, generate_password_hash
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

import pandas as pd

from submission import forecast


load_dotenv()
BASE_DIR = Path(__file__).resolve().parent


def build_database_url():
    host = os.getenv("MYSQL_HOST")
    database = os.getenv("MYSQL_DATABASE")
    user = os.getenv("MYSQL_USER")
    password = os.getenv("MYSQL_PASSWORD")
    if host and database and user:
        port = os.getenv("MYSQL_PORT", "3306")
        return f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4"
    return f"sqlite:///{BASE_DIR / 'data' / 'enviocardiac.db'}"


app = Flask(__name__, static_folder=None)
app.config["SQLALCHEMY_DATABASE_URI"] = build_database_url()
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
app.config["SECRET_KEY"] = os.getenv("APP_SECRET", "development-only-secret")
CORS(app, resources={r"/api/*": {"origins": "*"}})

db = SQLAlchemy(app)
session_serializer = URLSafeTimedSerializer(app.config["SECRET_KEY"], salt="enviocardiac-auth")


class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(255), nullable=False, default="HDHI Administrator")
    role = db.Column(db.String(50), nullable=False, default="admin")
    is_active = db.Column(db.Boolean, nullable=False, default=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)


class ForecastRun(db.Model):
    __tablename__ = "forecast_runs"
    id = db.Column(db.Integer, primary_key=True)
    cutoff_date = db.Column(db.Date, nullable=False)
    source = db.Column(db.String(50), nullable=False, default="sarimax")
    status = db.Column(db.String(20), nullable=False, default="completed")
    created_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    points = db.relationship("ForecastPoint", backref="run", cascade="all, delete-orphan")


class ForecastPoint(db.Model):
    __tablename__ = "forecast_points"
    id = db.Column(db.Integer, primary_key=True)
    run_id = db.Column(db.Integer, db.ForeignKey("forecast_runs.id"), nullable=False)
    forecast_date = db.Column(db.Date, nullable=False)
    predicted_admissions = db.Column(db.Float, nullable=False)
    lower_80 = db.Column(db.Float, nullable=False)
    upper_80 = db.Column(db.Float, nullable=False)


class Upload(db.Model):
    __tablename__ = "uploads"
    id = db.Column(db.Integer, primary_key=True)
    original_name = db.Column(db.String(255), nullable=False)
    stored_name = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(50), nullable=False)
    row_count = db.Column(db.Integer, nullable=False, default=0)
    created_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)


def auth_user():
    authorization = request.headers.get("Authorization", "")
    if not authorization.startswith("Bearer "):
        return None
    token = authorization[7:]
    try:
        payload = session_serializer.loads(token, max_age=60 * 60 * 12)
        return db.session.get(User, int(payload["user_id"]))
    except (BadSignature, SignatureExpired, KeyError, ValueError, TypeError):
        return None


def login_required(handler):
    @wraps(handler)
    def wrapped(*args, **kwargs):
        user = auth_user()
        if not user or not user.is_active:
            return jsonify({"error": "Authentication required"}), 401
        request.auth_user = user
        return handler(*args, **kwargs)
    return wrapped


def seed_admin():
    email = os.getenv("ADMIN_EMAIL", "admin@hdhi.org").strip().lower()
    password = os.getenv("ADMIN_PASSWORD", "change-me")
    user = db.session.execute(db.select(User).where(User.email == email)).scalar_one_or_none()
    if not user:
        db.session.add(User(
            email=email,
            password_hash=generate_password_hash(password),
            name="HDHI Administrator",
            role="admin",
        ))
        db.session.commit()


with app.app_context():
    (BASE_DIR / "data").mkdir(parents=True, exist_ok=True)
    db.create_all()
    seed_admin()


@app.get("/")
def dashboard():
    dashboard_path = BASE_DIR / "dashboard" / "hospital_admin_planning_view.html"
    return send_from_directory(str(dashboard_path.parent), dashboard_path.name)


@app.get("/dashboard/<path:filename>")
def dashboard_files(filename):
    return send_from_directory(str(BASE_DIR / "dashboard"), filename)


@app.get("/api/health")
def health():
    try:
        db.session.execute(text("SELECT 1"))
        database = "mysql" if app.config["SQLALCHEMY_DATABASE_URI"].startswith("mysql") else "sqlite"
        return jsonify({"status": "healthy", "database": database})
    except Exception as exc:
        return jsonify({"status": "unhealthy", "error": str(exc)}), 503


@app.post("/api/auth/login")
def login():
    body = request.get_json(silent=True) or {}
    email = str(body.get("email", "")).strip().lower()
    password = str(body.get("password", ""))
    user = db.session.execute(db.select(User).where(User.email == email)).scalar_one_or_none()
    if not user or not user.is_active or not check_password_hash(user.password_hash, password):
        return jsonify({"error": "Invalid email or password"}), 401
    token = session_serializer.dumps({"user_id": user.id})
    return jsonify({
        "token": token,
        "user": {"id": user.id, "email": user.email, "name": user.name, "role": user.role},
    })


@app.post("/api/auth/logout")
@login_required
def logout():
    return jsonify({"status": "logged_out"})


@app.get("/api/auth/me")
@login_required
def me():
    user = request.auth_user
    return jsonify({"id": user.id, "email": user.email, "name": user.name, "role": user.role})


@app.get("/api/forecast")
@login_required
def get_forecast():
    cutoff = request.args.get("cutoff_date", "2019-04-01")
    save = request.args.get("save", "false").lower() == "true"
    try:
        cutoff_dt = pd.to_datetime(cutoff)
    except Exception:
        return jsonify({"error": "cutoff_date must use YYYY-MM-DD format"}), 400
    result = forecast(cutoff_dt.strftime("%Y-%m-%d"))
    payload = result.rename(columns={
        "mean": "predicted_admissions",
        "lower_80": "lower_80",
        "upper_80": "upper_80",
    }).to_dict(orient="records")

    run_id = None
    if save:
        run = ForecastRun(cutoff_date=cutoff_dt.date(), created_by=request.auth_user.id)
        db.session.add(run)
        db.session.flush()
        for row in payload:
            db.session.add(ForecastPoint(
                run_id=run.id,
                forecast_date=date.fromisoformat(row["date"]),
                predicted_admissions=float(row["predicted_admissions"]),
                lower_80=float(row["lower_80"]),
                upper_80=float(row["upper_80"]),
            ))
        db.session.commit()
        run_id = run.id

    return jsonify({"run_id": run_id, "cutoff_date": cutoff_dt.strftime("%Y-%m-%d"), "forecast": payload})


@app.get("/api/forecast-runs")
@login_required
def forecast_runs():
    runs = ForecastRun.query.order_by(ForecastRun.created_at.desc()).limit(50).all()
    return jsonify([{
        "id": run.id,
        "cutoff_date": run.cutoff_date.isoformat(),
        "status": run.status,
        "source": run.source,
        "created_at": run.created_at.isoformat(),
        "point_count": len(run.points),
    } for run in runs])


@app.post("/api/uploads")
@login_required
def upload_csv():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "CSV file is required"}), 400
    if not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only CSV files are accepted"}), 400

    upload_dir = BASE_DIR / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    original_name = os.path.basename(file.filename)
    stored_name = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{secrets.token_hex(6)}_{original_name}"
    stored_path = upload_dir / stored_name
    file.save(stored_path)

    try:
        frame = pd.read_csv(stored_path)
        row_count = len(frame)
    except Exception:
        stored_path.unlink(missing_ok=True)
        return jsonify({"error": "The uploaded file is not valid CSV data"}), 400

    name_lower = original_name.lower()
    file_type = "admissions" if "admission" in name_lower else "weather" if "weather" in name_lower else "pollution" if "pollut" in name_lower or "airquality" in name_lower else "festival" if "festival" in name_lower else "other"
    record = Upload(
        original_name=original_name,
        stored_name=stored_name,
        file_type=file_type,
        row_count=row_count,
        created_by=request.auth_user.id,
    )
    db.session.add(record)
    db.session.commit()
    return jsonify({
        "id": record.id,
        "name": original_name,
        "type": file_type,
        "row_count": row_count,
    }), 201


if __name__ == "__main__":
    app.run(
        host=os.getenv("FLASK_HOST", "127.0.0.1"),
        port=int(os.getenv("FLASK_PORT", "5000")),
        debug=os.getenv("FLASK_ENV") == "development",
        threaded=True,
    )
