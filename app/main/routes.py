from flask import render_template, redirect, url_for, flash, request, jsonify, make_response
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import csv
import io
import hashlib
from app.main import bp
from app.main.forms import CSVUploadForm
from app.auth.models import Trial, MaterialData
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
    """Main dashboard selection page."""
    return render_template('dashboard_selection.html')

@bp.route('/dashboard/gridstack')
@login_required
def dashboard_gridstack():
    """GridStack.js dashboard proof of concept."""
    trials = Trial.query.filter_by(user_id=current_user.id).order_by(Trial.upload_timestamp.desc()).all()
    # Convert trials to JSON-serializable format
    trials_data = []
    for trial in trials:
        trials_data.append({
            'id': trial.id,
            'filename': trial.filename,
            'upload_timestamp': trial.upload_timestamp.isoformat(),
            'data_count': trial.material_data.count()
        })
    return render_template('dashboard_gridstack.html', trials=trials_data)

@bp.route('/dashboard/golden-layout')
@login_required
def dashboard_golden_layout():
    """Golden Layout dashboard proof of concept."""
    trials = Trial.query.filter_by(user_id=current_user.id).order_by(Trial.upload_timestamp.desc()).all()
    # Convert trials to JSON-serializable format
    trials_data = []
    for trial in trials:
        trials_data.append({
            'id': trial.id,
            'filename': trial.filename,
            'upload_timestamp': trial.upload_timestamp.isoformat(),
            'data_count': trial.material_data.count()
        })
    return render_template('dashboard_golden_layout.html', trials=trials_data)

@bp.route('/trial/<int:trial_id>')
@login_required
def view_trial(trial_id):
    trial = Trial.query.filter_by(id=trial_id, user_id=current_user.id).first_or_404()
    material_data = trial.material_data.all()
    return render_template('view_trial.html', title=f'Trial: {trial.filename}', 
                         trial=trial, material_data=material_data)

@bp.route('/trial/<int:trial_id>/delete', methods=['POST'])
@login_required
def delete_trial(trial_id):
    trial = Trial.query.filter_by(id=trial_id, user_id=current_user.id).first_or_404()
    filename = trial.filename
    
    try:
        db.session.delete(trial)
        db.session.commit()
        flash(f'Trial "{filename}" has been deleted successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting trial: {str(e)}', 'error')
    
    return redirect(url_for('main.index'))

@bp.route('/trial/<int:trial_id>/update', methods=['POST'])
@login_required
def update_trial(trial_id):
    trial = Trial.query.filter_by(id=trial_id, user_id=current_user.id).first_or_404()
    
    try:
        # Get the new filename from form data
        new_filename = request.form.get('filename', '').strip()
        if new_filename and new_filename != trial.filename:
            # Check for duplicate filename
            existing_trial = Trial.query.filter_by(
                user_id=current_user.id, 
                filename=new_filename
            ).first()
            
            if existing_trial:
                return jsonify({
                    'success': False, 
                    'message': f'A trial with filename "{new_filename}" already exists.'
                }), 400
            
            trial.filename = new_filename
        
        db.session.commit()
        return jsonify({
            'success': True, 
            'message': 'Trial updated successfully.',
            'trial': {
                'id': trial.id,
                'filename': trial.filename
            }
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False, 
            'message': f'Error updating trial: {str(e)}'
        }), 500

@bp.route('/api/material-data/<int:data_id>/update', methods=['POST'])
@login_required
def update_material_data(data_id):
    material_data = MaterialData.query.join(Trial).filter(
        MaterialData.id == data_id,
        Trial.user_id == current_user.id
    ).first_or_404()
    
    try:
        data = request.get_json()
        
        # Validate and update the data
        frequency_ghz = float(data.get('frequency_ghz', 0))
        dk = float(data.get('dk', 0))
        df = float(data.get('df', 0))
        
        if frequency_ghz <= 0 or dk <= 0 or df < 0:
            return jsonify({
                'success': False,
                'message': 'Invalid values. Frequency and Dk must be positive, Df must be non-negative.'
            }), 400
        
        material_data.frequency_ghz = frequency_ghz
        material_data.dk = dk
        material_data.df = df
        
        # Update the trial's data fingerprint since content changed
        trial = material_data.trial
        all_data = trial.material_data.all()
        sorted_data = sorted(all_data, key=lambda x: x.frequency_ghz)
        data_string = ""
        for data_point in sorted_data:
            freq = round(data_point.frequency_ghz, 6)
            dk_val = round(data_point.dk, 6)
            df_val = round(data_point.df, 8)
            data_string += f"{freq},{dk_val},{df_val};"
        
        trial.data_fingerprint = hashlib.sha256(data_string.encode('utf-8')).hexdigest()
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Data updated successfully.',
            'data': {
                'id': material_data.id,
                'frequency_ghz': material_data.frequency_ghz,
                'dk': material_data.dk,
                'df': material_data.df
            }
        })
        
    except ValueError:
        return jsonify({
            'success': False,
            'message': 'Invalid number format. Please enter valid numeric values.'
        }), 400
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': f'Error updating data: {str(e)}'
        }), 500

@bp.route('/api/material-data/<int:data_id>/delete', methods=['DELETE'])
@login_required
def delete_material_data(data_id):
    material_data = MaterialData.query.join(Trial).filter(
        MaterialData.id == data_id,
        Trial.user_id == current_user.id
    ).first_or_404()
    
    try:
        trial = material_data.trial
        db.session.delete(material_data)
        
        # Update the trial's data fingerprint since content changed
        remaining_data = trial.material_data.filter(MaterialData.id != data_id).all()
        if remaining_data:
            sorted_data = sorted(remaining_data, key=lambda x: x.frequency_ghz)
            data_string = ""
            for data_point in sorted_data:
                freq = round(data_point.frequency_ghz, 6)
                dk_val = round(data_point.dk, 6)
                df_val = round(data_point.df, 8)
                data_string += f"{freq},{dk_val},{df_val};"
            
            trial.data_fingerprint = hashlib.sha256(data_string.encode('utf-8')).hexdigest()
        else:
            # If no data remains, delete the trial
            db.session.delete(trial)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Data point deleted successfully.',
            'trial_deleted': len(remaining_data) == 0
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': f'Error deleting data: {str(e)}'
        }), 500

@bp.route('/api/trial/<int:trial_id>/add-data', methods=['POST'])
@login_required
def add_material_data(trial_id):
    trial = Trial.query.filter_by(id=trial_id, user_id=current_user.id).first_or_404()
    
    try:
        data = request.get_json()
        
        # Validate and create the new data
        frequency_ghz = float(data.get('frequency_ghz', 0))
        dk = float(data.get('dk', 0))
        df = float(data.get('df', 0))
        
        if frequency_ghz <= 0 or dk <= 0 or df < 0:
            return jsonify({
                'success': False,
                'message': 'Invalid values. Frequency and Dk must be positive, Df must be non-negative.'
            }), 400
        
        # Check for duplicate frequency in the same trial
        existing_data = MaterialData.query.filter_by(
            trial_id=trial_id,
            frequency_ghz=frequency_ghz
        ).first()
        
        if existing_data:
            return jsonify({
                'success': False,
                'message': f'A data point with frequency {frequency_ghz} GHz already exists in this trial.'
            }), 400
        
        # Create new material data
        new_data = MaterialData(
            frequency_ghz=frequency_ghz,
            dk=dk,
            df=df,
            trial_id=trial_id
        )
        
        db.session.add(new_data)
        db.session.flush()  # Get the new ID
        
        # Update the trial's data fingerprint since content changed
        all_data = trial.material_data.all()
        sorted_data = sorted(all_data, key=lambda x: x.frequency_ghz)
        data_string = ""
        for data_point in sorted_data:
            freq = round(data_point.frequency_ghz, 6)
            dk_val = round(data_point.dk, 6)
            df_val = round(data_point.df, 8)
            data_string += f"{freq},{dk_val},{df_val};"
        
        trial.data_fingerprint = hashlib.sha256(data_string.encode('utf-8')).hexdigest()
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'New data point added successfully.',
            'data': {
                'id': new_data.id,
                'frequency_ghz': new_data.frequency_ghz,
                'dk': new_data.dk,
                'df': new_data.df
            }
        })
        
    except ValueError:
        return jsonify({
            'success': False,
            'message': 'Invalid number format. Please enter valid numeric values.'
        }), 400
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': f'Error adding data: {str(e)}'
        }), 500

@bp.route('/trial/<int:trial_id>/download-csv')
@login_required
def download_trial_csv(trial_id):
    trial = Trial.query.filter_by(id=trial_id, user_id=current_user.id).first_or_404()
    material_data = trial.material_data.order_by(MaterialData.frequency_ghz.asc()).all()
    
    # Create CSV content
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write headers
    writer.writerow(['Frequency_GHz', 'Dk', 'Df'])
    
    # Write data rows
    for data in material_data:
        writer.writerow([
            f"{data.frequency_ghz:.6f}",
            f"{data.dk:.6f}", 
            f"{data.df:.8f}"
        ])
    
    # Create response
    csv_content = output.getvalue()
    output.close()
    
    response = make_response(csv_content)
    response.headers['Content-Type'] = 'text/csv'
    response.headers['Content-Disposition'] = f'attachment; filename="{trial.filename}"'
    
    return response