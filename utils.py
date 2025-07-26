import pandas as pd
import base64, io

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        text = decoded.decode('utf-8')
    except UnicodeDecodeError:
        text = decoded.decode('latin1')

    # Try multiple delimiters
    for sep in [',', ';', '\t']:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep)
            if df.shape[1] >= 3:  # Must have at least 3 columns
                return df
        except Exception:
            continue

    raise ValueError("Unable to parse CSV – unsupported format or inconsistent columns.")


def get_numeric_data(df):
    freq_ghz = pd.to_numeric(df.iloc[:,0], errors="coerce")
    dk = pd.to_numeric(df.iloc[:,1], errors="coerce")
    df_loss = pd.to_numeric(df.iloc[:,2], errors="coerce")
    mask = ~(freq_ghz.isna() | dk.isna() | df_loss.isna())
    return freq_ghz[mask].values, dk[mask].values, df_loss[mask].values
