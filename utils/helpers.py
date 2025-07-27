import base64, io
import pandas as pd


def get_numeric_data(df):
    freq_ghz = pd.to_numeric(df.iloc[:,0], errors="coerce")
    dk = pd.to_numeric(df.iloc[:,1], errors="coerce")
    df_loss = pd.to_numeric(df.iloc[:,2], errors="coerce")
    mask = ~(freq_ghz.isna() | dk.isna() | df_loss.isna())
    return freq_ghz[mask].values, dk[mask].values, df_loss[mask].values


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        text = decoded.decode('utf-8')
    except UnicodeDecodeError:
        text = decoded.decode('latin1')

    df = None

    # Try multiple delimiters
    for sep in [',', ';', '\t']:
        try:
            tmp_df = pd.read_csv(io.StringIO(text), sep=sep)
            if tmp_df.shape[1] >= 3:
                df = tmp_df
                break
        except Exception:
            continue

    if df is None:
        raise ValueError("Unable to parse CSV – unsupported format or inconsistent columns.")

    # ✅ Clean Data
    df = df.dropna(how="all")  # Remove empty rows
    df = df.dropna()  # Remove rows with any NaN
    df = df.apply(pd.to_numeric, errors="coerce")  # Convert to numeric
    df = df.dropna()  # Drop rows with NaNs after conversion
    df = df[df.iloc[:, 0] > 0]  # Remove zero or negative frequencies

    return df

