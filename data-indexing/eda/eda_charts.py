#!/usr/bin/env python3
"""Torque Diagnostics: two targeted EDA charts.

Chart Q1: distribution of *retrieved pages* for questions without a drive.
          Requires a real retrieval-log CSV; never substitutes corpus counts.
Chart Q2: procedure-vs-specification indicators in data sheets and manuals.
          Uses reproducible keyword heuristics (not verified semantic labels).

Run from the repository root:
  uv add boto3 duckdb pandas matplotlib
  uv run python eda_two_charts.py

Optional retrieval logs:
  EDA_RETRIEVAL_LOG=retrieval_log.csv uv run python eda_two_charts.py

Retrieval-log CSV required columns:
  document_id, page_number, no_drive
where no_drive is true/false or 1/0, and each row represents one retrieved page.
Optional query_id is used to deduplicate repeated pages within a query.

Output: eda-output/q1_retrieved_pages_by_kind.png (only with log),
        eda-output/q2_content_by_document_type.png, plus corresponding CSVs.
"""
from __future__ import annotations
import os
from pathlib import Path
import re
import boto3
import duckdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

BUCKET = os.getenv('EDA_BUCKET', 'g34305547-dats6450-s2026')
PREFIX = os.getenv('EDA_PREFIX', 'torque-diagnostics/chunks').strip('/')
REGION = os.getenv('AWS_REGION', 'us-east-1')
CSV = Path(os.getenv('EDA_CSV', 'ABB_100GB_Download_List.csv'))
LOG = Path(os.getenv('EDA_RETRIEVAL_LOG', 'retrieval_log.csv'))
OUT = Path(os.getenv('EDA_OUTPUT', 'eda-output'))
BATCH_SIZE = 100

# Indicators of procedure steps vs specification / rating / ordering facts.
# These are simple text heuristics; they may overlap and require manual validation.
PROCEDURE_RE = re.compile(r'\b(?:install|installation|mount|mounting|disconnect|connect|tighten|loosen|remove|replace|configure|reset|commission|step\s+\d+|procedure|before\s+starting)\b', re.I)
SPEC_RE = re.compile(r'\b(?:rated|rating|voltage|current|power|frequency|dimensions?|weight|ordering\s+code|order\s+code|product\s+code|specifications?|technical\s+data)\b', re.I)


def save_chart(fig, filename):
    fig.tight_layout()
    dest = OUT / filename
    fig.savefig(dest, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {dest}')


def catalog():
    if not CSV.is_file():
        raise FileNotFoundError(f'Missing {CSV}. Run from the repo root or set EDA_CSV.')
    df = pd.read_csv(CSV, dtype={'document_id': str}, low_memory=False)
    needed = {'document_id', 'document_kind'}
    if not needed.issubset(df.columns):
        raise ValueError(f'CSV needs {sorted(needed)}; found {list(df.columns)}')
    df = df[['document_id', 'document_kind']].dropna(subset=['document_id']).copy()
    df['document_id'] = df.document_id.astype(str).str.strip()
    df['document_kind'] = df.document_kind.fillna('Unknown').astype(str).str.strip()
    # Original download list has unique document IDs; guard against accidental duplicates.
    return df.drop_duplicates('document_id')


def q1_retrieval_chart(cat):
    if not LOG.exists():
        print(f'Q1 SKIPPED: {LOG} not found. A retrieval log is needed to measure retrieved-page shares.')
        return
    logs = pd.read_csv(LOG, dtype={'document_id': str}, low_memory=False)
    required = {'document_id', 'page_number', 'no_drive'}
    if not required.issubset(logs.columns):
        raise ValueError(f'Retrieval log needs columns {sorted(required)}; found {list(logs.columns)}')
    flags = logs.no_drive.astype(str).str.strip().str.lower().isin(['true', '1', 'yes', 'y'])
    logs = logs.loc[flags].copy()
    if logs.empty:
        print('Q1 SKIPPED: no no-drive retrieval results in log.')
        return
    keys = ['document_id', 'page_number']
    if 'query_id' in logs.columns:
        keys.insert(0, 'query_id')
    logs = logs.drop_duplicates(keys)
    joined = logs.merge(cat, on='document_id', how='left')
    joined['document_kind'] = joined.document_kind.fillna('Unknown')
    # This chart describes document kinds among retrieved pages. It does not infer
    # library membership, which needs an explicit S3 family or source-folder field.
    counts = joined.document_kind.value_counts().rename_axis('document_kind').reset_index(name='retrieved_pages')
    counts['percentage'] = 100 * counts.retrieved_pages / counts.retrieved_pages.sum()
    counts.to_csv(OUT / 'q1_retrieved_pages_by_kind.csv', index=False)
    top = counts.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top.document_kind, top.percentage)
    ax.set(xlabel='Share of retrieved pages (%)', title='Q1: Document types retrieved for questions without a drive')
    for y, value in enumerate(top.percentage):
        ax.text(value + .2, y, f'{value:.1f}%', va='center', fontsize=9)
    ax.set_xlim(0, max(top.percentage) * 1.18 if len(top) else 100)
    save_chart(fig, 'q1_retrieved_pages_by_kind.png')
    selected = counts[counts.document_kind.str.lower().isin(['catalogue', 'article'])]
    print(f'Q1 catalogue + article share: {selected.percentage.sum():.2f}% of logged retrieved pages')
    if 'asset_family' not in logs.columns:
        print('Q1 note: to establish that these pages came from *library*, add asset_family to retrieval logs.')
    else:
        lib = logs.merge(cat, on='document_id', how='left')
        lib.document_kind = lib.document_kind.fillna('Unknown')
        mask = lib.asset_family.astype(str).str.lower().eq('library') & lib.document_kind.str.lower().isin(['catalogue', 'article'])
        print(f'Q1 library catalogue/article share: {100*mask.mean():.2f}%')


def s3_files():
    client = boto3.client('s3', region_name=REGION)
    files = []
    for page in client.get_paginator('list_objects_v2').paginate(Bucket=BUCKET, Prefix=PREFIX + '/'):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if '/asset_family=' in key and key.endswith('.parquet'):
                files.append(f's3://{BUCKET}/{key}')
    return files


def q2_content_chart(cat):
    # Scan only files whose document IDs appear in the data sheet/manual catalog.
    target = cat[cat.document_kind.str.lower().isin(['data sheet', 'manual'])].copy()
    if target.empty:
        raise ValueError('No data sheet or manual rows in catalog.')
    ids = set(target.document_id)
    paths = [uri for uri in s3_files() if Path(uri).stem in ids]
    if not paths:
        raise RuntimeError('No matching Parquet files found in S3; check bucket/prefix and CSV IDs.')
    print(f'Q2: scanning {len(paths):,} matching document shards', flush=True)
    con = duckdb.connect()
    con.execute('INSTALL httpfs; LOAD httpfs;')
    con.execute(f"CREATE OR REPLACE SECRET eda_s3 (TYPE s3, PROVIDER credential_chain, REGION '{REGION}')")
    con.create_function('is_procedure', lambda s: bool(PROCEDURE_RE.search(s or '')), ['VARCHAR'], 'BOOLEAN', null_handling='special')
    con.create_function('is_spec', lambda s: bool(SPEC_RE.search(s or '')), ['VARCHAR'], 'BOOLEAN', null_handling='special')
    con.register('catalog', target)
    frames = []
    for start in range(0, len(paths), BATCH_SIZE):
        con.read_parquet(paths[start:start+BATCH_SIZE], union_by_name=True, hive_partitioning=False).create_view('passages', replace=True)
        frames.append(con.execute('''
            SELECT lower(c.document_kind) AS document_kind,
                   count(*) AS passages,
                   sum(CASE WHEN is_procedure(p.text) THEN 1 ELSE 0 END) AS procedure_mentions,
                   sum(CASE WHEN is_spec(p.text) THEN 1 ELSE 0 END) AS specification_mentions
            FROM passages p JOIN catalog c ON cast(p.document_id AS VARCHAR)=c.document_id
            WHERE p.text IS NOT NULL AND trim(p.text) <> ''
            GROUP BY 1
        ''').df())
        print(f'  processed {min(start+BATCH_SIZE,len(paths)):,}/{len(paths):,}', flush=True)
    summary = pd.concat(frames).groupby('document_kind', as_index=False).sum(numeric_only=True)
    if summary.empty:
        raise RuntimeError('No matching text passages found.')
    for col in ('procedure_mentions', 'specification_mentions'):
        summary[col + '_pct'] = 100 * summary[col] / summary.passages
    summary.to_csv(OUT / 'q2_content_by_document_type.csv', index=False)
    positions = range(len(summary))