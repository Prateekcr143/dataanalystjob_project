# src/train_model.py
"""
Train a model to predict avg_salary. Produces models/rf_salary_model.pkl and prints evaluation metrics.

Usage:
    python src/train_model.py --input data/cleaned_jobs.csv --model_out models/rf_salary_model.pkl
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_data(path):
    df = pd.read_csv(path)
    # keep rows with avg_salary
    df = df[pd.notnull(df['avg_salary'])]
    return df

def build_pipeline(cat_cols, num_cols):
    transformers = [
        ('num', StandardScaler(), num_cols)
    ]
    
    # Only add categorical transformer if there are categorical columns
    if len(cat_cols) > 0:
        transformers.append(
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        )
    
    pre = ColumnTransformer(transformers)
    
    pipe = Pipeline([
        ('pre', pre),
        ('rf', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
    ])
    return pipe

def main(args):
    df = load_data(args.input)
    print(f"Training rows: {len(df)}")

    # Features chosen - change as needed
    cat_cols = []
    # try include job_title, state; limit cardinality to top N to avoid explosion
    if 'job_title' in df.columns:
        # reduce to top 30 titles, mark others as "Other"
        top_titles = df['job_title'].value_counts().nlargest(30).index
        df['job_title_clean'] = df['job_title'].where(df['job_title'].isin(top_titles), 'Other')
        cat_cols.append('job_title_clean')
    if 'state' in df.columns:
        df['state_clean'] = df['state'].fillna('Unknown')
        cat_cols.append('state_clean')

    # numeric columns
    num_cols = []
    for c in ['rating','tech_skill_score','jd_length','min_salary','max_salary']:
        if c in df.columns:
            num_cols.append(c)

    # Drop rows with missing numeric features for simplicity
    df_model = df.dropna(subset=['avg_salary'] + num_cols)

    X = df_model[cat_cols + num_cols]
    y = df_model['avg_salary']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    pipe = build_pipeline(cat_cols, num_cols)

    print("Fitting RandomForest...")
    pipe.fit(X_train, y_train)

    # evaluate
    y_pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    print(f"MAE: {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R2: {r2:.4f}")

    # Persist model
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump(pipe, args.model_out)
    print(f"Saved model to {args.model_out}")

    # Optional: print feature importances if pipeline
    try:
        rf = pipe.named_steps['rf']
        # get feature names
        pre = pipe.named_steps['pre']
        # numeric names
        num_names = num_cols
        # categorical names from onehot encoder
        cat_names = []
        if len(cat_cols) > 0:
            ohe = pre.named_transformers_['cat']
            if hasattr(ohe, 'get_feature_names_out'):
                cat_names = list(ohe.get_feature_names_out(cat_cols))
            else:
                # fallback
                cat_names = []
        feat_names = num_names + cat_names
        importances = rf.feature_importances_
        # show top 20
        order = np.argsort(importances)[::-1][:20]
        print("Top feature importances:")
        for i in order:
            name = feat_names[i] if i < len(feat_names) else f"f{i}"
            print(f"  {name}: {importances[i]:.4f}")
    except Exception as e:
        print("Could not compute feature importances:", e)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="cleaned_jobs.csv")
    p.add_argument("--model_out", default="models/rf_salary_model.pkl")
    args = p.parse_args()
    main(args)
