# src/data_clean.py
"""
Load raw job-listings CSV, clean it, extract salary fields and simple features,
and write cleaned CSV to data/cleaned_jobs.csv

Usage:
    python src/data_clean.py --input data/data_analyst_jobs.csv --output data/cleaned_jobs.csv
"""

import argparse
import re
import pandas as pd
import numpy as np

# ---------- Utilities ----------
def find_col(df, candidates):
    """Return first column in df that matches any candidate name (case-insensitive)."""
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    # fallback: try substring match
    for col in df.columns:
        lc = col.lower()
        for cand in candidates:
            if cand.lower() in lc:
                return col
    return None

def parse_salary(s):
    """
    Parse many typical salary strings and return (min_salary, max_salary).
    Handles:
      - "$37K-$66K", "37k - 66k", "$80,000", "$40/hr"
    Returns (None, None) if cannot parse.
    """
    if pd.isna(s):
        return (None, None)
    s = str(s).replace(',', '').strip()
    s_low = s.lower()

    # detect hourly: convert to yearly using 2080 hours
    hr = re.search(r'(\d+(?:\.\d+)?)\s*(?:/|per)\s*hr', s_low)
    if hr:
        try:
            v = float(hr.group(1)) * 2080
            return (v, v)
        except:
            pass

    # ranges with K or without
    range_k = re.search(r'(\d+(?:\.\d+)?)\s*[kK]?\s*[-–]\s*(\d+(?:\.\d+)?)\s*[kK]?', s_low)
    if range_k:
        a, b = float(range_k.group(1)), float(range_k.group(2))
        # if K present in original string, or numbers small -> treat as thousands
        if 'k' in s_low or (a < 100 and b < 100):
            return (a*1000, b*1000)
        else:
            return (a, b)

    # single value with K
    single_k = re.search(r'(\d+(?:\.\d+)?)\s*[kK]\b', s_low)
    if single_k:
        v = float(single_k.group(1))*1000
        return (v, v)

    # single dollar amount like $80000 or 80000
    single_num = re.search(r'(\d{4,7})(?!\s*/\s*yr|\s*/\s*hr)', s.replace(',', ''))
    if single_num:
        try:
            v = float(single_num.group(1))
            # heuristic: if v < 10000 it's probably missing multiplier; but we'll keep raw
            return (v, v)
        except:
            pass

    return (None, None)

# ---------- Main cleaning pipeline ----------
def clean_dataframe(df):
    # attempt to find key columns (fall back to typical names)
    salary_col = find_col(df, ['Salary Estimate', 'Salary', 'salary_estimate', 'salaryestimate'])
    jd_col = find_col(df, ['Job Description', 'Description', 'job_description', 'jd'])
    company_col = find_col(df, ['Company Name', 'Company', 'company_name'])
    location_col = find_col(df, ['Location', 'location'])
    rating_col = find_col(df, ['Rating', 'rating'])
    title_col = find_col(df, ['Job Title', 'Title', 'job_title'])

    df = df.copy()

    # Normalize company_name: remove trailing rating text like "Company\n3.8"
    if company_col:
        df['company_name'] = df[company_col].astype(str).str.split('\n').str[0].str.strip()
    else:
        df['company_name'] = np.nan

    # Location -> city,state
    if location_col:
        loc = df[location_col].astype(str).str.split(',', expand=True)
        df['city'] = loc[0].str.strip()
        if loc.shape[1] > 1:
            df['state'] = loc[1].str.strip()
        else:
            df['state'] = np.nan
    else:
        df['city'] = df['state'] = np.nan

    # Rating numeric
    if rating_col:
        df['rating'] = pd.to_numeric(df[rating_col], errors='coerce')
    else:
        df['rating'] = np.nan

    # Job title
    if title_col:
        df['job_title'] = df[title_col].astype(str).str.strip()
    else:
        df['job_title'] = np.nan

    # Job Description text
    if jd_col:
        df['job_description'] = df[jd_col].astype(str)
    else:
        df['job_description'] = ''

    # Salary parsing
    df['raw_salary'] = df[salary_col].astype(str) if salary_col else np.nan
    parsed = df['raw_salary'].apply(parse_salary)
    df[['min_salary', 'max_salary']] = pd.DataFrame(parsed.tolist(), index=df.index)
    df['avg_salary'] = df[['min_salary', 'max_salary']].mean(axis=1)

    # Flag some skills from job_description
    skills = ['python','sql','excel','tableau','power bi','r','spark','aws','hadoop','machine learning','statistics']
    for s in skills:
        col = 'skill_' + s.replace(' ', '_')
        df[col] = df['job_description'].str.contains(s, case=False, na=False).astype(int)

    # tech skill score
    skill_cols = [c for c in df.columns if c.startswith('skill_')]
    df['tech_skill_score'] = df[skill_cols].sum(axis=1)

    # job description length
    df['jd_length'] = df['job_description'].str.split().str.len()

    # drop columns that are duplicates or huge raw text if you want smaller file (keep raw_salary & jd)
    # Keep original columns too for traceability
    return df

def main(args):
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")
    df_clean = clean_dataframe(df)
    df_clean.to_csv(args.output, index=False)
    print(f"Wrote cleaned data to {args.output}")
    print("Columns included (sample):", df_clean.columns.tolist()[:20])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data_analyst_jobs.csv")
    p.add_argument("--output", default="cleaned_jobs.csv")
    args = p.parse_args()
    main(args)
