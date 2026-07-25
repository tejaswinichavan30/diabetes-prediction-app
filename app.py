import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, login_required,
    logout_user, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash

# ---------------------------------------------------------------------------
# App / DB / Login setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-me')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to use the diabetes predictor.'
login_manager.login_message_category = 'info'


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


with app.app_context():
    db.create_all()

# ---------------------------------------------------------------------------
# Load model, scaler, and the exact column order used at training time
# ---------------------------------------------------------------------------
model = pickle.load(open('diabetes_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
COLUMNS = pickle.load(open('columns.pkl', 'rb'))

# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm_password', '')

        if not username or not password:
            flash('Username and password are required.', 'danger')
        elif password != confirm:
            flash('Passwords do not match.', 'danger')
        elif len(password) < 6:
            flash('Password must be at least 6 characters.', 'danger')
        elif User.query.filter_by(username=username).first():
            flash('That username is already taken.', 'danger')
        else:
            user = User(username=username)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            flash('Account created! Please log in.', 'success')
            return redirect(url_for('login'))

    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password.', 'danger')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# ---------------------------------------------------------------------------
# App routes (protected)
# ---------------------------------------------------------------------------
@app.route('/')
@login_required
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        # Read values in the exact column order the form fields are named
        # (fixes the original bug where fields were named "1".."8" instead
        # of the real feature names).
        input_values = [float(request.form[col]) for col in COLUMNS]
        input_df = pd.DataFrame([input_values], columns=COLUMNS)

        # Scale using the SAME scaler fitted during training
        # (the original app skipped this step, which silently made every
        # prediction wrong since the model was trained on scaled data).
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)

        if prediction[0] == 1:
            result = "High Risk of Diabetes"
            risk_class = "danger"
        else:
            result = "Low Risk of Diabetes"
            risk_class = "success"

        prob = round(probability[0][1] * 100, 2)

        return render_template(
            'result.html', result=result, probability=prob, risk_class=risk_class
        )

    except (KeyError, ValueError) as e:
        flash(f'Please fill in all fields with valid numbers. ({e})', 'danger')
        return redirect(url_for('home'))


if __name__ == "__main__":
    app.run(debug=True, port=5002)
