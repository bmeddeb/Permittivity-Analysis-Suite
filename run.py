from app import create_app, db
from app.models import User, Trial, MaterialData
import hashlib

app = create_app()

def migrate_existing_trials():
    """Add fingerprints to existing trials that don't have them."""
    trials_without_fingerprint = Trial.query.filter_by(data_fingerprint=None).all()
    
    for trial in trials_without_fingerprint:
        material_data = trial.material_data.all()
        if material_data:
            # Create fingerprint for existing data
            sorted_data = sorted(material_data, key=lambda x: x.frequency_ghz)
            data_string = ""
            for data in sorted_data:
                freq = round(data.frequency_ghz, 6)
                dk = round(data.dk, 6)
                df = round(data.df, 8)
                data_string += f"{freq},{dk},{df};"
            
            trial.data_fingerprint = hashlib.sha256(data_string.encode('utf-8')).hexdigest()
            print(f"Updated fingerprint for trial: {trial.filename}")
    
    db.session.commit()
    print(f"Updated {len(trials_without_fingerprint)} trials with fingerprints")

@app.shell_context_processor
def make_shell_context():
    return {'db': db, 'User': User, 'Trial': Trial, 'MaterialData': MaterialData}

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        migrate_existing_trials()  # Handle existing data
    app.run(debug=True)