from typing import List, Dict

import numpy as np
import json
import torch
from tqdm import tqdm
from rank_bm25 import *
import argparse
import os
import pickle
import glob
from utils import bm25_tokenizer, calculate_f2

from sentence_transformers import SentenceTransformer, util


def load_model(model_dir: str, model_path: str) -> SentenceTransformer:
    model_path = os.path.join(model_dir, model_path)
    model = SentenceTransformer(model_name_or_path=model_path)

    return model


def encode_legal_data(legal_dict_json, model: SentenceTransformer) -> np.ndarray:
    with open(legal_dict_json, "r") as f:
        doc_data = json.load(f)
        
    emb_list = []
    
    for k, doc in tqdm(doc_data.items()):
        emb = model.encode(doc_data[k]["title"] + " " + doc_data[k]["text"])
        emb_list.append(emb)
        
    emb_array = np.array(emb_list)
    
    return emb_array


def encode_question(questions: List[str], model: SentenceTransformer) -> Dict[str, np.ndarray]:
    emb_dict = {}
    for _, item in tqdm(enumerate(questions)):
        question_id = item["question_id"]
        question = item["question"]
        emb_dict[question_id] = model.encode(question)
    
    return emb_dict


def load_bm25(bm25_path) -> BM25Plus:
    with open(bm25_path, "rb") as bm_file:
        bm25 = pickle.load(bm_file)
    return bm25
        

def load_encoded_legal_data(legal_data_path):
    print("Start loading embedding of legal data")
    with open(legal_data_path, "rb") as f:
        emb_legal_data = pickle.load(f)
    return emb_legal_data    

    
def get_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", type=str, default="zac2021_ltr_data")
    parser.add_argument("--test_json_path", type=str, default="for_test_question_answer.json")
    parser.add_argument("--legal_dict_json", default="generated_data/legal_dict.json", type=str)
    parser.add_argument("--model_dir", default="saved_model", type=str)
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--bm25_path", default="saved_model/bm25_Plus_04_06_model_full_manual_stopword", type=str)
    parser.add_argument("--encoded_legal_data_path", default="encoded_data/encoded_legal_data.pkl", type=str)
    parser.add_argument("--doc_refers_path", default="saved_model/doc_refers_saved", type=str)
    parser.add_argument("--range_score", default=2.6, type=float)
    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    args = get_args()
    
    data_dir = args.data_dir
    test_json_path = args.test_json_path
    legal_dict_json = args.legal_dict_json
    model_dir = args.model_dir
    model_path = args.model_path
    bm25_path = args.bm25_path
    encoded_legal_data_path = args.encoded_legal_data_path
    doc_refers_path = args.doc_refers_path
    range_score = args.range_score
    
    print("Load model {}".format(model_path))
    model = load_model(model_dir=model_dir, model_path=model_path)
    
    if encoded_legal_data_path:
        print("Load embedding of legal data")
        emb_legal_data = load_encoded_legal_data(legal_data_path=encoded_legal_data_path)[model_path]
    else:
        print("Compute embedding of legal data against models")
        print("Encode legal data using {}".format(model_path))
        emb_legal_data = encode_legal_data(legal_dict_json, model=model)
    
    test_path = os.path.join(data_dir, test_json_path)
    data = json.load(open(test_path))
    items = data["items"]
    print("Num test question: {}".format(len(items)))
    
    print("Load BM25 model")
    bm25 = load_bm25(bm25_path)
    
    
    print("Load doc refer")
    with open(doc_refers_path, "rb") as doc_refer_file:
        doc_refers = pickle.load(doc_refer_file)
    
    print("Compute questions embedding against models")
    question_emb_dict = encode_question(questions=items, model=model)
    
    top_n = 61425
    
    total_f2 = 0
    total_precision = 0
    total_recall = 0
    
    for idx, item in tqdm(enumerate(items)):
        question_id = item["question_id"]
        question = item["question"]
        relevant_articles = item["relevant_articles"]
        actual_positive = len(relevant_articles)
        
        tokenized_query = bm25_tokenizer(question)
        doc_scores = bm25.get_scores(tokenized_query)
        
        quest_emb = question_emb_dict[question_id]
        scores = util.cos_sim(quest_emb, emb_legal_data)
        scores = scores.squeeze(0).numpy()
        new_scores = doc_scores * scores
        max_score = np.max(new_scores)
        
        predictions = np.argpartition(new_scores, len(new_scores) - top_n)[-top_n:]
        new_scores = new_scores[predictions]
        
        new_predictions = np.where(new_scores >= (max_score - range_score))[0]
        
        map_ids = predictions[new_predictions]
        
        new_scores = new_scores[new_scores >= (max_score - range_score)]
        
        if new_scores.shape[0] > 5:
            predictions_2 = np.argpartition(new_scores, len(new_scores) - 5)[-5:]
            map_ids = map_ids[predictions_2]
            
        true_positive = 0
        false_positive = 0
        for idx_3, idx_pred in enumerate(map_ids):
            pred = doc_refers[idx_pred]
            
            for article in relevant_articles:
                if pred[0] == article["law_id"] and pred[1] == article["article_id"]:
                    true_positive += 1
                else:
                    false_positive += 1
            
        precision = true_positive / (true_positive + false_positive + 1e-20)
        recall = true_positive / actual_positive
        
        f2 = calculate_f2(precision=precision, recall=recall)
        total_precision += precision
        total_recall += recall
        total_f2 += f2
        
    print(f"Average F2: \t\t\t\t{total_f2/len(items)}")
    print(f"Average Precision: {total_precision/len(items)}")
    print(f"Average Recall: {total_recall/len(items)}\n")