"""
MelanoTox-ML
Predicting Immunotherapy-Associated Toxicity in Melanoma Patients
HIPAA   : No PHI retained in any output file
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

#LOAD DATA

df = pd.read_excel("ML.xlsx")
print(f"Raw data: {df.shape}")

# REMOVE INVISIBLE CHARACTERS

df = df.map(lambda x: x.strip().replace('\xa0', '')
if isinstance(x, str) else x)

# STEP 3: FIX COLUMN NAME TYPO

df.rename(columns={"Immunotherpay Regimen": "Immunotherapy Regimen"},
          inplace=True)

# Date CHECK THEN DROP

df["Diagnosis date"] = pd.to_datetime(df["Diagnosis date"], dayfirst=True, errors="coerce")
df["Immunotherapy decision date"] = pd.to_datetime(df["Immunotherapy decision date"], dayfirst=True, errors="coerce")
df["Clinic Date"] = pd.to_datetime(df["Clinic Date"], dayfirst=True, errors="coerce")

df["days_IO_to_clinic"] = (df["Clinic Date"] - df["Immunotherapy decision date"]).dt.days
print(f"Days IO to Clinic (median): {df['days_IO_to_clinic'].median()}")

df.drop(columns=["Diagnosis date", "Immunotherapy decision date",
                 "Clinic Date", "days_IO_to_clinic"], inplace=True)

# DROP COLUMNS 

cols_to_drop = [
    # PHI
    "Patient identifier",
    # Response — too many missing, different task
    "Immunotherapy Response",
    # Toxicity severity — leakage + different task
    "Highest grade of toxiciy",
    "Highest grade of toxicity",
    "Need for hospital admission (1 Yes, 0 no)",
    "Treatment stopped after toxicity (1 Yes, 0 No)",
    "Blood and lymphatic system disorders",
    "Cardiac disorders",
    "Endocrine disorders",
    "Eye disorders",
    "Gastrointestinal disorder",
    "Hepatobiliary disorder",
    "Musculoskeletal and connective tissue disorders",
    "Nervous system disorders",
    "Renal and urinary disorders",
    "Respiratory disorder",
    "Skin and subcutaneous tissue disorders",
    "Unspecified",
    # IO combinations — 90%+ missing
    "IO only", "IO+Radio", "IO+Chemo", "IO+Surg",
    "IO+Radio+Chemo", "IO+Radio+Surg", "IO+Chemo+Surg",
    "IO+Radio+Chemo+Surg"
]
cols_to_drop = [c for c in cols_to_drop if c in df.columns]
df.drop(columns=cols_to_drop, inplace=True)
print(f"After dropping columns: {df.shape}")

# CLEAN TARGET COLUMN

df.rename(columns={"Any Toxicity associated with Immunotherapy?": "toxicity"},
          inplace=True)
df["toxicity"] = df["toxicity"].str.strip().str.upper()
df["toxicity"] = df["toxicity"].map({"Y": 1, "N": 0})

print(f"Target distribution:\n{df['toxicity'].value_counts()}")
print(f"Missing in target: {df['toxicity'].isnull().sum()}")

# RENAME COLUMNS

df.rename(columns={
    "Diagnosis Age": "age",
    "Gender": "gender",
    "Immunotherapy Regimen": "io_regimen",
    "Immunotherapy Dose": "io_dose",
    "Immunotherapy Frequency": "io_frequency",
    "Intent": "intent",
    "Line of treatment": "line_of_treatment",
    "T": "T_stage",
    "N": "N_stage",
    "M": "M_stage",
    "Melanoma stage": "melanoma_stage",
    "EAS Performance status": "ecog_ps",
    "Previous PD (1-yes, 0-no)": "prev_PD",
    "Previous SD (1-yes, 0-no)": "prev_SD",
    "Previous R (1-yes, 0-no)": "prev_R",
    "BRAF pathogenic variant Y/N": "BRAF_variant",
    "Radiotherapy": "radiotherapy",
    "Chemotherapy": "chemotherapy",
    "Surgical Resection": "surgical_resection",
}, inplace=True)

# FIX DATA TYPES

num_cols = ["age", "io_dose", "prev_PD", "prev_SD", "prev_R",
            "radiotherapy", "chemotherapy", "surgical_resection"]
for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

cat_cols = ["gender", "io_regimen", "io_frequency", "intent",
            "melanoma_stage", "T_stage", "N_stage", "M_stage",
            "BRAF_variant", "ecog_ps", "line_of_treatment"]
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.upper()

# Replace all empty strings with NaN
df.replace('', np.nan, inplace=True)

# Fix io_frequency, merge NAN and nan into proper missing
df["io_frequency"] = df["io_frequency"].replace({"NAN": np.nan, "nan": np.nan})

# Fix T_stage, merge nan into proper missing
df["T_stage"] = df["T_stage"].replace({"nan": np.nan})

# Drop completely empty columns
empty_cols = [c for c in df.columns if df[c].isnull().sum() == len(df)]
df.drop(columns=empty_cols, inplace=True)
print(f"Dropped empty columns: {empty_cols}")

# FILL MISSING VALUES

for col in df.select_dtypes(include=np.number).columns:
    if col != "toxicity":
        df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].replace("NAN", np.nan).fillna(df[col].mode()[0])

# REMOVE DUPLICATES

df.drop_duplicates(inplace=True)

#  SAVE CLEAN DATA

df.to_csv("melanoma_clean.csv", index=False)

print(f"\n--- Cleaning Summary ---")
print(f"Shape         : {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Duplicates    : {df.duplicated().sum()}")
print(f" Saved: melanoma_clean.csv")

# FEATURE ENGINEERING

# M_stage correlated 0.68 with melanoma_stage

df.drop(columns=["M_stage"], inplace=True, errors="ignore")
print(f"\n Dropped M_stage")

#  SEPARATE FEATURES AND TARGET

X = df.drop(columns=["toxicity"])
y = df["toxicity"]

print(f" Features : {X.shape}")
print(f" Target   : {y.shape}")

#  ENCODE CATEGORICAL COLUMNS

cat_cols = X.select_dtypes(include="object").columns.tolist()
print(f"\nEncoding: {cat_cols}")

le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    le_dict[col] = le
    print(f"  {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

print("Encoding complete")

# CHECK ZERO VARIANCE

zero_var = X.columns[X.var() == 0].tolist()
if zero_var:
    X.drop(columns=zero_var, inplace=True)
    print(f"Dropped zero variance columns: {zero_var}")
else:
    print("No zero variance columns")

# TRAIN / TEST SPLIT
# 80% train, 20% test, stratified

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\n✓ Train : {X_train.shape} | Toxicity rate: {y_train.mean():.2%}")
print(f"✓ Test  : {X_test.shape}  | Toxicity rate: {y_test.mean():.2%}")


# SCALE FEATURES
# Fit on train only 

scaler = StandardScaler()

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns
)

print(f" Scaling complete")

# SAVE ALL FILES

X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
X_train_scaled.to_csv("X_train_scaled.csv", index=False)
X_test_scaled.to_csv("X_test_scaled.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print(f"""
--- Final Summary ---
Total features  : {X_train.shape[1]}
Training samples: {X_train.shape[0]}
Test samples    : {X_test.shape[0]}
Train toxicity  : {y_train.mean():.2%}
Test toxicity   : {y_test.mean():.2%}

Features: {list(X_train.columns)}

Files saved:
  melanoma_clean.csv
  X_train.csv, X_test.csv           - XGBoost
  X_train_scaled.csv, X_test_scaled - Logistic Regression
  y_train.csv, y_test.csv           - both models

""")
