from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from app import db

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(128))
    
    # Relationship to trials
    trials = db.relationship('Trial', backref='user', lazy='dynamic')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'

class Trial(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    upload_timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    data_fingerprint = db.Column(db.String(64), nullable=True, index=True)  # SHA256 hash of data content
    
    # Relationship to material data and fitted models
    material_data = db.relationship('MaterialData', backref='trial', lazy='dynamic', cascade='all, delete-orphan')
    fitted_models = db.relationship('FittedModel', backref='trial', lazy='dynamic', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Trial {self.filename}>'

class MaterialData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    frequency_ghz = db.Column(db.Float, nullable=False)
    dk = db.Column(db.Float, nullable=False)  # Dielectric constant
    df = db.Column(db.Float, nullable=False)  # Dissipation factor
    trial_id = db.Column(db.Integer, db.ForeignKey('trial.id'), nullable=False)
    
    def __repr__(self):
        return f'<MaterialData freq={self.frequency_ghz} dk={self.dk} df={self.df}>'

class FittedModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    trial_id = db.Column(db.Integer, db.ForeignKey('trial.id'), nullable=False)
    model_type = db.Column(db.String(50), nullable=False)  # 'debye', 'cole_cole', etc.
    model_name = db.Column(db.String(100), nullable=False)  # Human-readable name
    fit_timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    # Fit results
    fit_parameters = db.Column(db.JSON, nullable=False)  # Fitted parameter values and uncertainties
    fit_quality_metrics = db.Column(db.JSON, nullable=False)  # R-squared, chi-squared, etc.
    fitted_frequency_data = db.Column(db.JSON, nullable=False)  # Frequency points for fitted curve
    fitted_dk_data = db.Column(db.JSON, nullable=False)  # Fitted Dk values
    fitted_df_data = db.Column(db.JSON, nullable=False)  # Fitted Df values
    
    # Fit metadata
    convergence_status = db.Column(db.String(20), nullable=False)  # 'success', 'failed', 'warning'
    fit_message = db.Column(db.Text, nullable=True)  # Details about fit result
    iterations = db.Column(db.Integer, nullable=True)  # Number of iterations
    
    # Model-specific configuration
    model_config = db.Column(db.JSON, nullable=True)  # Model-specific parameters (e.g., number of poles)
    
    def __repr__(self):
        return f'<FittedModel {self.model_name} for Trial {self.trial_id}>'