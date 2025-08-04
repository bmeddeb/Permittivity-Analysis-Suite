from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
from wtforms.validators import DataRequired


class CSVUploadForm(FlaskForm):
    csv_file = FileField(
        "CSV File", validators=[FileRequired(), FileAllowed(["csv"], "CSV files only!")]
    )
    submit = SubmitField("Upload CSV")
