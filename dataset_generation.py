import os
import re
import pickle
import json
import time

import pandas as pd
import numpy as np
from tqdm import tqdm

import download_pubmed_data as dpd

ROOT = "dset_gen"


def make_manual_dataset(dataset_name, use_fulltext=False):
    with open(f"{ROOT}/{dataset_name}-setup.json", 'r') as fh:
        setup_data = json.load(fh)
    rct_text_data = dpd.get_rct_pmid_data()
    fulltext_used = 0
    content_type = 'fulltext' if use_fulltext else 'abstract'
    if use_fulltext:
        rct_fulltext_data = dpd.get_rct_fulltext()
    review_to_rct_pmids = dpd.get_review_to_references_map(id_type="pmid")
    qa_dataset = []
    for review_pmid, review_qs in tqdm(setup_data.items()):
        if len(review_qs) < 1:
            continue
        all_review_rcts = review_to_rct_pmids[review_pmid]
        for q in review_qs:
            assert all([pmid in all_review_rcts for pmid in q['relevant_sources']]),\
                f"relevant={q['relevant_sources']}, all={all_review_rcts}"
        article_data = []
        for rct_pmid in all_review_rcts:
            rct_data = {'article_id': rct_pmid, **rct_text_data[rct_pmid]}
            if use_fulltext and (rct_pmid in rct_fulltext_data):
                rct_data['content'] = rct_fulltext_data[rct_pmid]
                fulltext_used += 1
            article_data.append(rct_data)
        qa_dataset.append({
            "original_review": review_pmid,
            "question_data": review_qs,
            "sources": article_data
        })
    if use_fulltext:
        print("fulltexts used:", fulltext_used)
    dset_path = f'run-med-evidence/datasets/{dataset_name}-{content_type}.jsonl'
    with open(dset_path, 'w') as fh:
        for review_data in qa_dataset:
            json.dump(review_data, fh)
            fh.write('\n')
    print("dataset made and written to", dset_path)

def convert_manual_spreadsheet(dataset_name):
    csv_path = f'{ROOT}/{dataset_name}-data_spreadsheet.csv'
    df = pd.read_csv(csv_path).fillna("")
    df['relevant_sources'] = [
        [elem.strip() for elem in text_list.split(',')]
        for text_list in df["Relevant Cited PubMed IDs"].values
    ]
    print(df.head())
    rows = df.drop(columns=['Timestamp', 'Email Address', 'Relevant Cited PubMed IDs']).rename(columns={
        "Cochrane PubMed ID": "original_review",
        "Question": "question",
        "Answer": "answer",
        "Evidence Quality": "evidence_quality",
        "Open-Access Full-text Needed": "fulltext_required",
        "Comment": "comment"
    }).reset_index(names="question_id").to_dict(orient='records')
    proto_dataset = {}
    for row in rows:
        review_pmid = row.pop('original_review')
        proto_dataset.setdefault(review_pmid, []).append(row)
    out_path = f"{ROOT}/{dataset_name}-setup.json"
    with open(out_path, 'w') as fh:
        json.dump(proto_dataset, fh)
    print('dataset setup written to', out_path)

if __name__ == '__main__':
    # parse_llm_response_file()
    # abstract_data = get_samples()
    # out_name = get_experiment_output_name()
    
    # abstract_data = get_all_nonempty_pmid_reviews()
    # dset_name = "all_nonempty_pmids"
    # response_name = f"responses-{dset_name}".replace('.', '_')
    # response_out_name = get_experiment_output_name(response_name)
    # parse_review_results(abstract_data, response_out_name)
    # make_dataset(dataset_name=dset_name, use_fulltext=False)
    
    name = 'manual_250'
    convert_manual_spreadsheet(name)
    make_manual_dataset(name, use_fulltext=False)
    make_manual_dataset(name, use_fulltext=True)