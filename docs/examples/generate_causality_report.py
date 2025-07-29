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
1	3.7	0.005
2	3.69	0.0052
3	3.685	0.0055
4	3.68	0.0058
5	3.675	0.006
6	3.67	0.0063
7	3.665	0.0066
8	3.66	0.0069
9	3.655	0.0072
10	3.65	0.0075
11	3.646	0.0078
12	3.642	0.0081
13	3.638	0.0084
14	3.634	0.0087
15	3.63	0.009
16	3.626	0.0093
17	3.622	0.0096
18	3.618	0.0099
19	3.614	0.0102
20	3.61	0.0105
21	3.606	0.0108
22	3.602	0.0111
23	3.598	0.0114
24	3.594	0.0117
25	3.59	0.012
26	3.586	0.0123
27	3.582	0.0126
28	3.578	0.0129
29	3.574	0.0132
30	3.57	0.0135
31	3.566	0.0138
32	3.562	0.0141
33	3.558	0.0144
34	3.554	0.0147
35	3.55	0.015
36	3.546	0.0153
37	3.542	0.0156
38	3.538	0.0159
39	3.534	0.0162
40	3.53	0.0165
41	3.526	0.0168
42	3.522	0.0171
43	3.518	0.0174
44	3.514	0.0177
45	3.51	0.018
46	3.506	0.0183
47	3.502	0.0186
48	3.498	0.0189
49	3.494	0.0192
50	3.49	0.0195
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
