from flask import render_template, redirect, url_for, flash, request
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import csv
import io
import hashlib
from app.main import bp
from app.main.forms import CSVUploadForm
from app.models import Trial, MaterialData
from app import db

@bp.route('/', methods=['GET', 'POST'])
@bp.route('/index', methods=['GET', 'POST'])
def index():
    form = CSVUploadForm()
    
    if current_user.is_authenticated and form.validate_on_submit():
        return process_csv_upload(form)
    
    # Get recent trials for logged-in users
    recent_trials = []
    if current_user.is_authenticated:
        recent_trials = current_user.trials.order_by(Trial.upload_timestamp.desc()).limit(5).all()
    
    return render_template('index.html', title='Home', form=form, recent_trials=recent_trials)

def create_data_fingerprint(material_data_records):
    """Create a SHA256 fingerprint of the material data for duplicate detection."""
    # Sort data by frequency for consistent hashing
    sorted_data = sorted(material_data_records, key=lambda x: x.frequency_ghz)
    
    # Create a string representation of the data
    data_string = ""
    for data in sorted_data:
        # Round to prevent floating point precision issues
        freq = round(data.frequency_ghz, 6)
        dk = round(data.dk, 6)
        df = round(data.df, 8)
        data_string += f"{freq},{dk},{df};"
    
    # Create SHA256 hash
    return hashlib.sha256(data_string.encode('utf-8')).hexdigest()

def check_for_duplicates(user_id, filename, data_fingerprint):
    """Check for duplicate trials by filename or data content."""
    # Check for same filename
    filename_duplicate = Trial.query.filter_by(
        user_id=user_id, 
        filename=filename
    ).first()
    
    # Check for same data content (only check trials that have fingerprints)
    content_duplicate = Trial.query.filter(
        Trial.user_id == user_id,
        Trial.data_fingerprint == data_fingerprint,
        Trial.data_fingerprint.isnot(None)
    ).first()
    
    return filename_duplicate, content_duplicate

def process_csv_upload(form):
    try:
        csv_file = form.csv_file.data
        filename = secure_filename(csv_file.filename)
        
        # Read CSV content with UTF-8 BOM handling
        raw_data = csv_file.stream.read()
        
        # Try different encodings to handle various file formats
        encoding_used = None
        try:
            # First try UTF-8 with BOM (handles files like Excel exports)
            decoded_data = raw_data.decode('utf-8-sig')
            encoding_used = 'utf-8-sig'
        except UnicodeDecodeError:
            try:
                # Fallback to UTF-8 without BOM
                decoded_data = raw_data.decode('utf-8')
                encoding_used = 'utf-8'
            except UnicodeDecodeError:
                try:
                    # Final fallback to latin-1 (covers most Windows encodings)
                    decoded_data = raw_data.decode('latin-1')
                    encoding_used = 'latin-1'
                except UnicodeDecodeError:
                    flash('Unable to decode file. Please ensure it is a valid CSV file with UTF-8 or standard encoding.', 'error')
                    return redirect(url_for('main.index'))
        
        stream = io.StringIO(decoded_data, newline=None)
        csv_reader = csv.DictReader(stream)
        
        # Validate CSV headers (strip whitespace and handle BOM issues)
        actual_headers = set(header.strip() for header in csv_reader.fieldnames if header)
        expected_headers = {'Frequency_GHz', 'Dk', 'Df'}
        
        if not expected_headers.issubset(actual_headers):
            flash(f'Invalid CSV format. Expected headers: {", ".join(expected_headers)}. Found: {", ".join(actual_headers)}', 'error')
            return redirect(url_for('main.index'))
        
        # Process CSV rows (without creating trial yet)
        material_data_records = []
        row_count = 0
        
        for row in csv_reader:
            # Skip empty rows
            if not any(row.values()) or all(not str(v).strip() for v in row.values()):
                continue
                
            try:
                frequency_ghz_str = row['Frequency_GHz'].strip()
                dk_str = row['Dk'].strip()
                df_str = row['Df'].strip()
                
                # Skip rows with empty required fields
                if not frequency_ghz_str or not dk_str or not df_str:
                    continue
                
                frequency_ghz = float(frequency_ghz_str)
                dk = float(dk_str)
                df = float(df_str)
                
                # Create temporary MaterialData object (without trial_id)
                material_data = MaterialData(
                    frequency_ghz=frequency_ghz,
                    dk=dk,
                    df=df,
                    trial_id=None  # Will be set after duplicate check
                )
                material_data_records.append(material_data)
                row_count += 1
                
            except (ValueError, KeyError) as e:
                flash(f'Error processing row {row_count + 1}: {str(e)}. Row data: {dict(row)}', 'error')
                return redirect(url_for('main.index'))
        
        if row_count == 0:
            flash('No valid data rows found in the CSV file.', 'error')
            return redirect(url_for('main.index'))
        
        # Create data fingerprint
        data_fingerprint = create_data_fingerprint(material_data_records)
        
        # Check for duplicates
        filename_duplicate, content_duplicate = check_for_duplicates(
            current_user.id, filename, data_fingerprint
        )
        
        # Handle duplicates
        if filename_duplicate:
            flash(f'A file with the name "{filename}" has already been uploaded on {filename_duplicate.upload_timestamp.strftime("%Y-%m-%d %H:%M")}. Please rename your file or use a different filename.', 'warning')
            return redirect(url_for('main.index'))
        
        if content_duplicate:
            flash(f'This data has already been uploaded as "{content_duplicate.filename}" on {content_duplicate.upload_timestamp.strftime("%Y-%m-%d %H:%M")}. The content is identical even though the filename may be different.', 'warning')
            return redirect(url_for('main.index'))
        
        # No duplicates found, create the trial
        trial = Trial(
            filename=filename, 
            user_id=current_user.id,
            data_fingerprint=data_fingerprint
        )
        db.session.add(trial)
        db.session.flush()  # Get trial ID
        
        # Set trial_id for all material data records
        for material_data in material_data_records:
            material_data.trial_id = trial.id
        
        # Bulk insert material data
        db.session.add_all(material_data_records)
        db.session.commit()
        
        flash(f'Successfully uploaded {filename} with {row_count} data points!', 'success')
        return redirect(url_for('main.index'))
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error processing file: {str(e)}', 'error')
        return redirect(url_for('main.index'))

@bp.route('/dashboard')
@login_required
def dashboard():
    # Redirect to the Dash app endpoint
    return redirect('/dashboard/')

@bp.route('/trial/<int:trial_id>')
@login_required
def view_trial(trial_id):
    trial = Trial.query.filter_by(id=trial_id, user_id=current_user.id).first_or_404()
    material_data = trial.material_data.all()
    return render_template('view_trial.html', title=f'Trial: {trial.filename}', 
                         trial=trial, material_data=material_data)