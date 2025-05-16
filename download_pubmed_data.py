import os
import sys
import re
import json
import pickle
import time
import urllib.parse
import requests
import datetime
from collections import defaultdict
import urllib
import ast

import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from Bio import Entrez
import pymed

from xmlparser import XMLArticle, parse_xml_batch

SUBSETS = ['commercial', 'noncommercial', 'other']
DATA_ROOT = 'cochrane_data'
BATCH_SIZE = 610
BIOMEDICA_INDEX_PATH = "you'll have to download the biomedica index yourself" # https://github.com/biomedica-marvl/biomedica-index
EMAIL = "PLACEHOLDER@university.edu"

Entrez.email = EMAIL
# Entrez.sleep_between_tries = 3

def load_if_present(filepath):
    read_mode = 'rb' if filepath.endswith('.pkl') else 'r'
    if not os.path.exists(filepath):
        return None
    with open(filepath, read_mode) as fh:
        if filepath.endswith('.pkl'):
            out = pickle.load(fh)
        elif filepath.endswith('.json'):
            out = json.load(fh)
        elif filepath.endswith('.jsonl'):
            out = [json.loads(line) for line in fh]
        else:
            out = fh.read()
    return out


def get_biomedica_pmcids():
    with open(f'{BIOMEDICA_INDEX_PATH}/full_text-kw/local_pmcid_map.json', 'r') as fh:
        article_maps = json.load(fh)

    flat_map = {}
    for subset in SUBSETS:
        for pmcid, batch_file in tqdm(article_maps[subset].items()):
            flat_map[pmcid] = (batch_file, subset)
    return flat_map


def get_2014_to_2024_cochrane_reviews(review_id_path=DATA_ROOT+'/review_ids.pkl'):
    if os.path.exists(review_id_path):
        with open(review_id_path, 'rb') as fh:
            cochrane_review_ids = pickle.load(fh)
    else:
        query = '("The Cochrane database of systematic reviews"[Journal]) AND (("2014/01/01"[Date - Publication] : "2024/04/04"[Date - Publication])) AND systematic [sb]'
        with Entrez.esearch(db="pubmed", term=query, retmax=10_000) as handle:
            search_record = Entrez.read(handle)

        cochrane_review_ids = search_record['IdList']
        print("TOTAL REVIEWS TAGGED:", len(cochrane_review_ids))
        with open(review_id_path, 'wb') as fh:
            pickle.dump(cochrane_review_ids, fh)
    return cochrane_review_ids


def _save_records():
    cochrane_review_ids = get_2014_to_2024_cochrane_reviews()
    included_records = []
    offset = 0
    batch_iter = tqdm(range(BATCH_SIZE*offset, len(cochrane_review_ids), BATCH_SIZE))
    for batch_id, ix in enumerate(batch_iter):
        pmid_batch = cochrane_review_ids[ix:ix+BATCH_SIZE]
        batch_ids = ','.join(pmid_batch)
        batch_iter.set_description('fetching')
        with Entrez.efetch(db="pubmed", id=batch_ids, retmode="xml") as handle:
            batch_iter.set_description('reading')
            res = handle.read()
        batch_iter.set_description('writing')
        with open(f"{DATA_ROOT}/batch{batch_id+offset}.xml", 'wb') as fh:
            fh.write(res)


def _cleanup_id(id_str):
    clean = id_str.replace(',','.') # specific known edge case/typo
    clean = re.sub(r'[^a-zA-Z0-9.\-_;/()[\]:+]', '', clean) # remove things invalid to a DOI
    return clean


def _get_ids_for_study(ref_study, id_types=['doi', 'pmid', 'pmcid']):
    # ids_list: list of (id_type, id_value) tuples
    xml_type_to_regular_type = {
        'pubmed': 'pmid',
        'pmc': 'pmcid'
    }
    ids_list = []
    for citation in ref_study['Reference']:
        if 'ArticleIdList' not in citation:
            continue
        for cite_id in citation['ArticleIdList']:
            xml_type = cite_id.attributes['IdType']
            regular_type = xml_type_to_regular_type.get(xml_type, xml_type)
            if regular_type in id_types:
                out_id = _cleanup_id(str(cite_id))
                ids_list.append((regular_type, out_id))
    return ids_list


def __convert_ids(id_list, from_type='doi', to_type='pmid'):
    batch_size = 100
    service_root = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
    parameters = {
        'tool': 'idconverter',
        'email': EMAIL
    }
    records = []
    for k in tqdm(range(0, len(id_list), batch_size), desc=f"{from_type}->{to_type}"):
        id_batch = id_list[k:k+batch_size]

        query_url = f"{service_root}?ids={','.join(id_batch)}&format=json&idtype={from_type}"
        output = requests.get(query_url, params=parameters).json()
        if output['status'] == 'ok':
            records += output['records']
        else:
            parsed_msg = urllib.parse.unquote(output['message'])
            open_quote = parsed_msg.find("'")
            close_quote = parsed_msg.rfind("'")
            invalid_id = parsed_msg[open_quote+1:close_quote]
            assert invalid_id in id_batch
            id_batch.remove(invalid_id)
            query_url = f"{service_root}?ids={','.join(id_batch)}&format=json&idtype={from_type}"
            output = requests.get(query_url, params=parameters).json()
            assert output['status'] == 'ok'
            records += output['records']
        time.sleep(0.5)
    # convert pmcid -> fulltext mapping to pmid -> fulltext mapping
    id_mapping = {}
    for r in records:
        if r.get(to_type):
            id_mapping[r[from_type]] = r[to_type]
    return id_mapping


def _make_basic_review_ref_map(id_types=['doi', 'pmid', 'pmcid']):
    generic_map_file = DATA_ROOT+'/review_to_generic_reference_map.json'
    cochrane_review_ids = get_2014_to_2024_cochrane_reviews()
    print(len(cochrane_review_ids))
    ### get raw reference lists
    included_refs = {}
    for i,k in enumerate(tqdm(range(0, len(cochrane_review_ids), BATCH_SIZE))):
        pmid_batch = cochrane_review_ids[k:k+BATCH_SIZE]
        records = Entrez.read(f"{DATA_ROOT}/batch{i}.xml")
        for j, review_pmid in enumerate(pmid_batch):
            try:
                root_title = records['PubmedArticle'][j]['PubmedData']['ReferenceList'][0].get('Title')
                if 'included' in root_title.lower():
                    ref_list = records['PubmedArticle'][j]['PubmedData']['ReferenceList'][0]['ReferenceList']
                    if len(ref_list) > 0:
                        included_refs[review_pmid] = ref_list
            except:
                pass
    print(f"Total included papers: {len(included_refs)} ({len(included_refs)/len(cochrane_review_ids)})")
    ref_counts = {k:len(v) for k,v in included_refs.items()}
    ref_ct_arr = np.array(list(ref_counts.values()))
    avg_n_refs = ref_ct_arr.mean()
    print("Average # ref lists:", avg_n_refs)
    ### get the actual citation ids (doi, pmid, pmcid)
    ref_ids_per_review = {}
    for review_pmid, ref_list in included_refs.items():
        ref_ids = {id_type: [] for id_type in id_types}
        for ref_study in ref_list:
            ids_list = _get_ids_for_study(ref_study, id_types=id_types)
            for id_type, id_value in ids_list:
                ref_ids[id_type].append(id_value)
        ref_ids_per_review[review_pmid] = ref_ids
    with open(generic_map_file, 'w') as fh:
        json.dump(ref_ids_per_review, fh)


def get_total_included_references_per_review():
    ref_count_file = DATA_ROOT+f'/review_reference_counts.json'
    if (out := load_if_present(ref_count_file)):
        return out
    cochrane_review_ids = get_2014_to_2024_cochrane_reviews()

    ### count references
    def count_total_refs(review_ref_list):
        total_count = 0
        for ref_study in review_ref_list:
            total_count += len(ref_study.get("Reference",[]))
        return total_count
    ### get raw reference lists
    ref_counts = defaultdict(int) # start counts at 0
    for i,k in enumerate(tqdm(range(0, len(cochrane_review_ids), BATCH_SIZE))):
        pmid_batch = cochrane_review_ids[k:k+BATCH_SIZE]
        records = Entrez.read(f"{DATA_ROOT}/batch{i}.xml")
        for j, review_pmid in enumerate(pmid_batch):
            try:
                root_title = records['PubmedArticle'][j]['PubmedData']['ReferenceList'][0].get('Title')
                if 'included' in root_title.lower():
                    ref_list = records['PubmedArticle'][j]['PubmedData']['ReferenceList'][0]['ReferenceList']
                    ref_counts[review_pmid] = count_total_refs(ref_list)
            except:
                pass
    print("Total overall reference citations:", sum(ref_counts.values()))
    print("Average references/review:", sum(ref_counts.values())/len(ref_counts))
    with open(ref_count_file, 'w') as fh:
        json.dump(ref_counts, fh)
    return ref_counts


def get_review_to_study_map(id_types=['doi', 'pmid', 'pmcid']):
    pass



def get_review_to_references_map(id_type=None, all_id_types=['doi', 'pmid', 'pmcid']):
    specific_map_file = DATA_ROOT+f'/review_to_{id_type}_reference_map.json'
    if (out := load_if_present(specific_map_file)):
        return out

    generic_map_file = DATA_ROOT+'/review_to_generic_reference_map.json'
    generic_map = load_if_present(generic_map_file)
    if not generic_map:
        _make_basic_review_ref_map(id_types=all_id_types)
        generic_map = load_if_present(generic_map_file)
    
    if not id_type:
        return generic_map

    ### convert ALL known ids to the desired id_type where possible
    specific_map = defaultdict(set)
    for from_type in all_id_types:
        if from_type == id_type:
            for review_pmid, ref_ids_by_type in generic_map.items():
                specific_map[review_pmid].update(ref_ids_by_type[from_type])
        else: # from_type is not id_type -> need to convert ids where possible
            from_ids = set()
            for review_pmid, ref_ids_by_type in generic_map.items():
                from_ids.update(ref_ids_by_type[from_type])
            from_ids = list(from_ids)
            id_conversion_map = __convert_ids(from_ids, from_type=from_type, to_type=id_type)
            for review_pmid, ref_ids_by_type in generic_map.items():
                for from_id in ref_ids_by_type[from_type]:
                    if from_id in id_conversion_map:
                        to_id = id_conversion_map[from_id]
                        specific_map[review_pmid].add(to_id)
    # convert the sets to lists
    specific_map = {k: list(v) for k,v in specific_map.items()}
    with open(specific_map_file, 'w') as fh:
        json.dump(specific_map, fh)
    return specific_map


def get_review_to_pmcoa_references_map():
    map_file = DATA_ROOT+f'/review_to_pmcoa_reference_map.json'
    if (out := load_if_present(map_file)):
        return out

    pmcid_refs = get_review_to_references_map(id_type='pmcid')
    flat_map = get_biomedica_pmcids()
    pmcoa_refs = {}
    for review_pmid, pmcid_list in pmcid_refs.items():
        pmcoa_refs[review_pmid] = [pmcid for pmcid in pmcid_list if (pmcid in flat_map)]
    with open(map_file, 'w') as fh:
        json.dump(pmcoa_refs, fh)
    return pmcoa_refs


def _get_pmid_data(pmids, bsz=100):
    print("num pmids:", len(pmids))
    pmid_data = {}
    failed = 0
    pubmed = pymed.PubMed(tool="PmidTextDataDownloader", email=EMAIL)
    for k in tqdm(range(0, len(pmids), bsz)):
        pmid_batch = pmids[k:k+bsz]
        articles = pubmed._getArticles(article_ids=pmid_batch)
        for pmid, article in zip(pmid_batch, articles):
            abstr = getattr(article, "abstract", None)
            title = getattr(article, "title", None)
            date = getattr(article, "publication_date", None)
            date = date.strftime("%Y-%m-%d") if isinstance(date, datetime.date) else None
            failed += int(abstr is None)
            pmid_data[pmid] = {"content": abstr, "title": title, "date": date}
    print(f"N FAILED: {failed}")
    return pmid_data


def get_rct_pmid_data():
    data_path = DATA_ROOT+"/rct_known_pmid_data.pkl"
    if (out := load_if_present(data_path)):
        return out

    pmid_refs = get_review_to_references_map(id_type="pmid")
    unique_pmids = list(set([_id for pmid_list in pmid_refs.values() for _id in pmid_list]))
    rct_pmid_data = _get_pmid_data(unique_pmids)
    with open(data_path, 'wb') as fh:
        pickle.dump(rct_pmid_data, fh)
    print("RCT PMID DATA SAVED")
    return rct_pmid_data


def get_review_pmid_data():
    data_path = DATA_ROOT+"/review_known_pmid_data.pkl"
    if (out := load_if_present(data_path)):
        return out

    pmid_refs = get_review_to_references_map()
    unique_pmids = list(pmid_refs.keys())
    review_pmid_data = _get_pmid_data(unique_pmids)
    with open(data_path, 'wb') as fh:
        pickle.dump(review_pmid_data, fh)
    print("REVIEW PMID DATA SAVED")
    return review_pmid_data


class SimpleArticleLoader:
    def __init__(self, key='abstract'):
        with open(f'{BIOMEDICA_INDEX_PATH}/full_text-kw/local_pmcid_map.json', 'r') as fh:
            self.article_maps = json.load(fh)
        self.key = key
    
    def get_content(self, batch_file, pmcid):
        with open(batch_file, 'r') as fh:
            article_batch = json.load(fh)
        for article in article_batch:
            if article["accession_id"] == pmcid:
                return article if (not self.key) else article.get(self.key)
        return None

    def __getitem__(self, pmcid):
        for subset in SUBSETS:
            if (batch_file := self.article_maps[subset].get(pmcid)) is not None:
                return self.get_content(batch_file, pmcid)
        return None


def _get_pmcoa_fulltext(pmcid_list):
    fulltext_loader = SimpleArticleLoader(key='nxml')
    pmcid_fulltext = {}
    failed = 0
    for pmcid in tqdm(pmcid_list):
        text = fulltext_loader[pmcid]
        if text:
            pmcid_fulltext[pmcid] = text
        else:
            failed += 1
    print(f"INVALID PMCIDS: {failed}")
    return pmcid_fulltext


def get_rct_fulltext():
    data_path = DATA_ROOT+"/rct_pmcid_fulltext-mapped_by_pmid.pkl"
    if (out := load_if_present(data_path)):
        return out
    pmcoa_map = get_review_to_pmcoa_references_map()
    unique_pmcids = list(set([_id for pmcid_list in pmcoa_map.values() for _id in pmcid_list]))
    pmcid2pmid = __convert_ids(unique_pmcids, from_type="pmcid", to_type="pmid")
    fulltext_map = _get_pmcoa_fulltext(unique_pmcids)
    pmid_fulltext_map = {pmcid2pmid[pmcid]:fulltext 
        for pmcid,fulltext in fulltext_map.items() if (pmcid in pmcid2pmid)}
    with open(data_path, 'wb') as fh:
        pickle.dump(pmid_fulltext_map, fh)
    print("RCT FULLTEXT DATA SAVED")
    return pmid_fulltext_map


def get_pmcoa_cochrane_reviews():
    data_path = DATA_ROOT+"/pmcoa_cochrane_biomedica_data.pkl"
    if (out := load_if_present(data_path)):
        return out

    cochrane_review_ids = get_2014_to_2024_cochrane_reviews()
    flat_map = get_biomedica_pmcids()
    all_data_loader = SimpleArticleLoader(key=None)
    cochrane_pmcoa = []
    for i,k in enumerate(tqdm(range(0, len(cochrane_review_ids), BATCH_SIZE))):
        pmid_batch = cochrane_review_ids[k:k+BATCH_SIZE]
        records = Entrez.read(f"{DATA_ROOT}/batch{i}.xml")
        for i, pmid in enumerate(pmid_batch):
            article_ids = records['PubmedArticle'][i]['PubmedData']['ArticleIdList']
            pmcid = None
            for cite_id in article_ids:
                if cite_id.attributes['IdType'] == 'pmc':
                    pmcid = str(cite_id)
                    break
            if flat_map.get(pmcid) is not None:
                batch_file, _ = flat_map[pmcid]
                fulltext_data = all_data_loader.get_content(batch_file, pmcid)
                if fulltext_data is not None:
                    cochrane_pmcoa.append(fulltext_data)
    print("Total Biomedica PMC-OA cochrane reviews:", len(cochrane_pmcoa))
    with open(data_path, 'wb') as fh:
        pickle.dump(cochrane_pmcoa, fh)
    print("FULL-TEXT DATA SAVED")
    return cochrane_pmcoa

def get_pmcoa_xml_data(batch_size=50):
    data_path = DATA_ROOT+"/pmcoa_cochrane_rawXML.pkl"
    if (out := load_if_present(data_path)):
        return out
    
    reviews = get_pmcoa_cochrane_reviews()
    pmcids = [rev['accession_id'] for rev in reviews]
    pubmed = pymed.PubMed(tool='load_pmc_oa', email='clcp@stanford.edu')
    pubmed.parameters['db'] = 'pmc'

    batches = []
    for k in trange(0, len(pmcids), batch_size):
        pmcid_batch = pmcids[k:k+batch_size]
        params = pubmed.parameters.copy()
        params['id'] = pmcid_batch
        response = pubmed._get(
            url="/entrez/eutils/efetch.fcgi", parameters=params, output="xml"
        )
        assert isinstance(response, str)
        batches.append((pmcid_batch, response))
    with open(data_path, 'wb') as fh:
        pickle.dump(batches, fh)
    return batches


def get_pmcoa_study_characteristics(ref_type='pmid', batch_size=50):
    """
    THE CHALLENGE HERE IS IN THE PARSE_INCLUDED_STUDIES() FUNCTION
    TL;DR MOST XML SOURCES DON'T ACTUALLY PROVIDE ANY!! WAY OF MAPPING THEIR CITATIONS BACK TO ACTUAL PAPERS
    """
    data_path = DATA_ROOT+f"/all_pmcoa_cochrane_studyChars-{ref_type}_only.pkl"
    # if (out := load_if_present(data_path)):
    #     return out

    xml_batches = get_pmcoa_xml_data()
    pmcid_included_studies = {}
    omitted = []
    for (pmcid_batch, xml_string) in tqdm(xml_batches):
        batch_root = parse_xml_batch(xml_string)
        for pmcid, article_root in zip(pmcid_batch, batch_root):
            print(f"=== NEW ARTICLE: {pmcid} ===")
            try:
                article = XMLArticle(article_root, keep_fallbacks=False, ref_type=ref_type)
                study_data = article.parse_included_studies()
                if len(study_data) > 0:
                    pmcid_included_studies[pmcid] = study_data
            except:
                print("PARSING FAILED -- excluding study")
                omitted.append(pmcid)
    print("omitted studies:", omitted)
    with open(data_path, 'wb') as fh:
        pickle.dump(pmcid_included_studies, fh)
    print(f"STUDY CHARACTERISTICS FOR {len(pmcid_included_studies)} REVIEWS SAVED")
    return pmcid_included_studies


def convert_pmcoa_studyChars_dois_to_pmids():
    data_path = DATA_ROOT+"/all_pmcoa_cochrane_studyChars-doi2pmid.pkl"
    if (out := load_if_present(data_path)):
        return out

    doi_studychar = get_pmcoa_study_characteristics()
    dois = set()
    for meta_review, studies_data in doi_studychar.items():
        dois.update((doi for doi in studies_data))
    dois = list(dois)
    doi_pmid_map = __convert_ids(dois, from_type='doi', to_type='pmid')

    pmid_studychar = {}
    total_studies = 0
    for meta_review, studies_data in doi_studychar.items():
        filtered_studies = {}
        for doi, study in studies_data.items():
            if (pmid := doi_pmid_map.get(doi)):
                filtered_studies[pmid] = study
                total_studies += 1
        if len(filtered_studies) > 0:
            pmid_studychar[meta_review] = filtered_studies
    with open(data_path, 'wb') as fh:
        pickle.dump(pmid_studychar, fh)
    print(f"STUDY CHARACTERISTICS FOR {len(pmid_studychar)} REVIEWS SAVED ({total_studies} total studies)")
    return pmid_studychar


def get_rct_abstract_studychar_pairs(
        data_path=DATA_ROOT+"/pmcoa_cochrane_reviews-rctAbstract_studychar_pairs.pkl"):
    if os.path.exists(data_path):
        with open(data_path, 'rb') as fh:
            data = pickle.load(fh)
        return data

    cochrane_studies_data = get_pmcoa_study_characteristics(
        data_path=DATA_ROOT+"/all_pmcoa_cochrane_studyChars-doi2pmid.pkl")
    rcts = []
    known_pmid_abstracts = get_review_pmid_abstracts()
    missing_pmids = []
    for included_studies in cochrane_studies_data.values():
        for pmid, parsed_table_sections in included_studies.items():
            study_chars = parsed_table_sections.get(
                'Study characteristics',
                parsed_table_sections.get('unknown')) # default
            if (study_chars is not None) and ('Methods' in study_chars):
                rcts.append((pmid, study_chars))
                if pmid not in known_pmid_abstracts:
                    missing_pmids.append(pmid)
    print("N missing pmids:", len(missing_pmids))
    print("known abstracts before:", len(known_pmid_abstracts))
    known_pmid_abstracts.update(get_pmid_abstracts(missing_pmids, data_path=""))
    print("known abstracts after:", len(known_pmid_abstracts))
    abstract_studychar_pairs = [(known_pmid_abstracts.get(pmid,"NO ABSTRACT PROVIDED"), studychar)
                                for (pmid, studychar) in rcts]
    with open(data_path, 'wb') as fh:
        pickle.dump(abstract_studychar_pairs, fh)
    return abstract_studychar_pairs

if __name__ == '__main__':
    # cochrane_review_ids = get_2014_to_2024_cochrane_reviews()
    # _save_records(cochrane_review_ids)
    # _make_basic_review_ref_map()
    # get_total_included_references_per_review()
    # get_review_to_references_map(id_type='pmid')
    # get_review_to_references_map(id_type='pmcid')
    # get_review_to_pmcoa_references_map()
    # get_rct_pmid_data()
    # get_review_pmid_data()
    # get_rct_fulltext()
    # get_pmcoa_cochrane_reviews()
    # get_pmcoa_xml_data()
    # get_pmcoa_study_characteristics()


    # get_pmcoa_study_characteristics()
    # convert_pmcoa_studyChars_dois_to_pmids()
    # abstract_characteristics = get_rct_abstract_studychar_pairs()
    # get_cochrane_abstracts()
    # get_rct_pmid_data()
    # get_pmcoa_rct_fulltext()
    pass