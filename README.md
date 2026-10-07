Enter the project environment:
source .venv/bin/activate

## Data cleaning

Values outside the plausible ranges below are set to NaN (the rest of the row is kept).

| Variable | Valid range | Unit | Applied in |
|---|---|---|---|
| heartrate | 20–300 | beats/min | `data_preprocessing.py` |
| sao2 | 50–100 | % | `data_preprocessing.py` |
| respiration | 1–80 | breaths/min | `data_preprocessing.py` |
| nibp_systolic | 40–300 | mmHg | `data_preprocessing.py` |
| nibp_diastolic | 10–200 | mmHg | `data_preprocessing.py` |
| nibp_mean | 20–250 | mmHg | `data_preprocessing.py` |
| temperature | 25–45 | °C (Fahrenheit converted first) | `file_read.py` |
| gcs | 3–15 | points | `file_read.py` |

Blood pressure consistency: when systolic, mean and diastolic are all present, a measurement must satisfy systolic > mean > diastolic; otherwise all three values are set to NaN.
