from flask import Flask
import dash
import dash_bootstrap_components as dbc
from layout import layout
from callbacks import register_callbacks

server = Flask(__name__)
app = dash.Dash(__name__, server=server, 
                external_stylesheets=[
                    dbc.themes.BOOTSTRAP,
                    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
                ],
                external_scripts=[
                    'https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js'
                ])

app.layout = layout
register_callbacks(app)

@server.route('/')
def index():
    return app.index()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
