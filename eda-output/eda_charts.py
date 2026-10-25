#!/usr/bin/env python3
"""Generate six initial ABB EDA charts, without reading Parquet contents.

From repo root:
    uv add pandas matplotlib boto3
    uv run python data-indexing/eda/eda_five_charts.py

Uses ABB_100GB_Download_List.csv for figures 2-4, and S3 chunk-shard
filenames for figures 1 and 5. A shard is one successfully chunked PDF;
shard counts may differ slightly from source-PDF counts. No S3 writes.
"""
from __future__ import annotations
import os
from collections import Counter
from pathlib import Path
import boto3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

BUCKET = os.getenv('EDA_BUCKET', 'g34305547-dats6450-s2026')
PREFIX = os.getenv('EDA_PREFIX', 'torque-diagnostics/chunks').strip('/')
REGION = os.getenv('AWS_REGION', 'us-east-1')
CSV = Path(os.getenv('EDA_CSV', 'ABB_100GB_Download_List.csv'))
OUT = Path(os.getenv('EDA_OUTPUT', 'eda-output'))
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {OUT / name}', flush=True)


def horizontal(labels, values, title, xlabel, filename):
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.42 * len(labels) + 1.2)))
    ax.barh(labels[::-1], values[::-1])
    ax.set(title=title, xlabel=xlabel)
    ax.grid(axis='x', alpha=0.18)
    ax.set_axisbelow(True)
    save(fig, filename)


def source_csv_charts():
    if not CSV.is_file():
        raise FileNotFoundError(f'Missing {CSV}. Run from repository root or set EDA_CSV.')
    df = pd.read_csv(CSV, dtype={'document_id': str}, low_memory=False)
    required = {'document_id', 'document_kind', 'file_size_mb'}
    if not required.issubset(df.columns):
        raise ValueError(f'CSV must have {sorted(required)}; found {list(df.columns)}')
    df['document_kind'] = df['document_kind'].fillna('Unknown').astype(str).str.strip().replace('', 'Unknown')
    df['file_size_mb'] = pd.to_numeric(df['file_size_mb'], errors='coerce')
    print(f'Source CSV: {len(df):,} rows, {df.document_id.nunique():,} unique document IDs')

    kinds = df.groupby('document_kind').agg(pdf_count=('document_id', 'size'), total_mb=('file_size_mb', 'sum')).reset_index()
    kinds['total_gb'] = kinds.total_mb / 1024
    kinds.to_csv(OUT / 'document_kind_summary.csv', index=False)
    counts = kinds.sort_values('pdf_count', ascending=False).head(12)
    horizontal(counts.document_kind.tolist(), counts.pdf_count.tolist(),
               'Figure 2. PDF count by document type', 'Number of source PDFs', '02_document_types.png')

    storage = kinds.sort_values('total_gb', ascending=False).head(12)
    horizontal(storage.document_kind.tolist(), storage.total_gb.tolist(),
               'Figure 3. PDF storage by document type', 'Total source PDF size (GiB)', '03_document_types_by_size.png')

    sizes = df.file_size_mb.dropna()
    names = ['Under 1 MB', '1 to <10 MB', '10 to <50 MB', '50 MB or more']
    vals = [int((sizes < 1).sum()), int(((sizes >= 1) & (sizes < 10)).sum()),
            int(((sizes >= 10) & (sizes < 50)).sum()), int((sizes >= 50).sum())]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(names, vals)
    ax.set(title='Figure 4. Distribution of source PDF sizes', ylabel='Number of PDFs')
    for i, value in enumerate(vals):
        ax.annotate(f'{value:,}', (i, value), xytext=(0, 4), textcoords='offset points', ha='center')
    save(fig, '04_pdf_size_distribution.png')
    pd.DataFrame({'size_group': names, 'pdf_count': vals}).to_csv(OUT / 'pdf_size_groups.csv', index=False)

    # FIGURE 6: DOCUMENT TYPES AMONG THE 20 LARGEST SOURCE PDFS
    largest = df.dropna(subset=['file_size_mb']).nlargest(20, 'file_size_mb')
    top20_counts = largest['document_kind'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top20_counts.index[::-1], top20_counts.values[::-1], color='steelblue')
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.08, bar.get_y() + bar.get_height() / 2,
                f'{int(width)}', va='center')
    ax.set(title='Figure 6. Document types among the 20 largest ABB PDFs',
           xlabel='Number of PDFs', ylabel='Document type')
    ax.set_xlim(0, top20_counts.max() + 1)
    ax.grid(axis='x', alpha=0.2)
    ax.set_axisbelow(True)
    save(fig, '06_largest_pdf_document_types.png')
    top20_counts.rename_axis('document_kind').reset_index(name='pdf_count').to_csv(
        OUT / 'largest_20_document_kind_counts.csv', index=False)


def family_shard_charts():
    s3 = boto3.client('s3', region_name=REGION)
    counts = Counter()
    total = 0
    print(f'Listing S3 shard names in s3://{BUCKET}/{PREFIX}/ (no Parquet reads)...', flush=True)
    for page in s3.get_paginator('list_objects_v2').paginate(Bucket=BUCKET, Prefix=PREFIX + '/'):
        for obj in page.get('Contents', []):
            key = obj['Key']
            marker = '/asset_family='
            if not key.endswith('.parquet') or marker not in key:
                continue
            family = key.split(marker, 1)[1].split('/', 1)[0]
            if family:
                counts[family] += 1
                total += 1
    if not counts:
        raise RuntimeError(f'No chunk shards found in s3://{BUCKET}/{PREFIX}/')
    print(f'Found {total:,} Parquet shards across {len(counts)} families')
    summary = pd.DataFrame(sorted(counts.items()), columns=['asset_family', 'chunked_pdf_count'])
    summary.to_csv(OUT / 'equipment_family_shard_counts.csv', index=False)

    # Figure 1 uses the source-PDF summary from the initial EDA report,
    # not Parquet shard counts (one source PDF did not yield a shard).
    # File-size values are source PDF totals, NOT Parquet storage size.
    source_counts = [25873, 2793]
    source_gb = [78.67, 23.17]
    categories = ['library', 'Named equipment']
    colors = ['#8998a8', '#24547f']
    fig, (ax_files, ax_size) = plt.subplots(1, 2, figsize=(12.8, 6.2))
    fig.suptitle('Library compared with the named equipment folders', fontsize=20, y=0.97)
    for ax, values, heading, ylabel, fmt in [
        (ax_files, source_counts, 'Files', 'PDFs', lambda x: f'{x:,}'),
        (ax_size, source_gb, 'File size', 'Gigabytes', lambda x: f'{x:.2f}'),
    ]:
        bars = ax.bar(categories, values, color=colors, width=0.62)
        ax.set_title(heading, fontsize=17, pad=12)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_ylim(0, max(values) * 1.18)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='x', labelsize=12)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, value + max(values)*0.025,
                    fmt(value), ha='center', va='bottom', fontsize=12)
    fig.subplots_adjust(top=0.79, bottom=0.13, wspace=0.28)
    fig.savefig(OUT / '01_library_vs_equipment.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {OUT / "01_library_vs_equipment.png"}', flush=True)

    top = [(family, n) for family, n in counts.items() if family != 'library']
    top.sort(key=lambda pair: (-pair[1], pair[0]))
    top = top[:10]
    horizontal([x[0] for x in top], [x[1] for x in top],
               'Figure 5. Top named equipment folders by chunked PDFs',
               'Number of Parquet shards (one per chunked PDF)', '05_top_equipment_folders.png')


def main():
    source_csv_charts()
    family_shard_charts()
    print('\nDone: six PNG charts and four CSV summaries in', OUT)
    print('Note: Figure 1 uses reported source-PDF totals; Figure 5 counts chunked PDF shards.')


if __name__ == '__main__':
    main()
