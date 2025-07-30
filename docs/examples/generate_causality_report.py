#!/usr/bin/env python3
"""
generate_causality_report.py

Load a small Dk/Df dataset, run the Kramers–Kronig causality check,
and print a human‑readable report.
"""



import os
import sys
import io
import pandas as pd

# Add the project root to the Python path so we can find the app module
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))

from app.models import KramersKronigValidator
# ——————————————————————————————————————————————
# 1) Embed your data as a heredoc for easy copy/paste.
#    Alternatively, load from a CSV file.
# ——————————————————————————————————————————————
data = """\
Frequency_GHz	Dk	Df
1.1	3.87	0.006
10	3.84	0.007
20	3.8	0.008
26.96	3.76	0.009
29.46	3.76	0.009
31.95	3.75	0.009
34.45	3.75	0.009
36.95	3.74	0.009
39.45	3.74	0.009
41.95	3.74	0.009
44.44	3.74	0.009
46.94	3.73	0.009
49.44	3.73	0.01
51.94	3.73	0.01
54.44	3.73	0.01
56.94	3.72	0.009
59.43	3.72	0.01
61.93	3.72	0.01
64.43	3.71	0.01
66.93	3.71	0.01
69.43	3.7	    0.01
71.93	3.7	    0.01
74.42	3.7	    0.01
76.92	3.69	0.01
79.42	3.68	0.01
81.92	3.68	0.01
84.42	3.68	0.01
86.92	3.67	0.01
89.41	3.67	0.01
91.91	3.66	0.01
94.41	3.66	0.01
96.91	3.65	0.01
99.41	3.64	0.01
101.9	3.64	0.01
104.4	3.63	0.01
106.9	3.63	0.01
109.4	3.62	0.01
"""


# Read into a DataFrame
#df = pd.read_csv(io.StringIO(data), sep=r'\s+')
df = pd.read_csv(io.StringIO(data), delim_whitespace=True)
# Rename to match validator expectations
df = df.rename(columns={'Frequency_GHz': 'Frequency (GHz)'})
# ——————————————————————————————————————————————
# 2) Instantiate the validator
# ——————————————————————————————————————————————
validator = KramersKronigValidator(
    df,
    method='auto',         # automatically choose Hilbert vs trapz
    eps_inf_method='fit',  # estimate ε_inf via regression
    eps_inf=None,          # or set an explicit floating‐point override
    tail_fraction=0.1,     # fraction of high‐freq tail for ε_inf
    min_tail_points=3,     # minimum points in that tail
    window=None,           # e.g. 'hann' if you want to taper
    resample_points=None   # controls Hilbert resampling on non‐uniform grid
)

# ——————————————————————————————————————————————
# 3) Run the check and print the report
# ——————————————————————————————————————————————
results = validator.validate(causality_threshold=0.05)
print(validator.get_report())

# Optionally, access the raw numbers:
print("Mean relative error:", results['mean_relative_error'])
print("RMSE:", results['rmse'])
