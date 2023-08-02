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
from utils import bm25_tokenizer, calculate_f2, load_model, load_cross_encoder, encode_legal_data, encode_question, load_bm25, load_encoded_legal_data

from sentence_transformers import SentenceTransformer, util

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", type=str, default="zac2021_ltr_data")
    parser.add_argument("--test_json_path", type=str, default="for_test_question_answer.json")
    parser.add_argument("--legal_dict_json", default="generated_data/legal_dict.json", type=str)
    parser.add_argument("--model_dir", default="saved_model", type=str)
    parser.add_argument("--model_paths", default=None, type=str)
    parser.add_argument("--bm25_path", default="saved_model/bm25_Plus_04_06_model_full_manual_stopword", type=str)
    parser.add_argument("--cross_encoder_name", type=str, default=None)
    parser.add_argument("--primary_top_n", type=int, default=25)
    parser.add_argument("--encoded_legal_data_path", default="encoded_data/encoded_legal_data.pkl", type=str)
    parser.add_argument("--doc_refers_path", default="saved_model/doc_refers_saved", type=str)
    parser.add_argument("--range_score", default=2.6, type=float)
    
    args = parser.parse_args()
    
    
if __name__ == "__main__":
    args = get_args()
    
    data_dir = args.data_dir
    test_json_path = args.test_json_path
    legal_dict_json = args.legal_dict_json
    model_dir = args.model_dir
    model_paths = args.model_paths.split(",")
    bm25_path = args.bm25_path
    cross_encoder_name = args.cross_encoder_name
    primary_top_n = args.primary_top_n
    encoded_legal_data_path = args.encoded_legal_data_path
    doc_refers_path = args.doc_refers_path
    range_score = args.range_score
    
    if encoded_legal_data_path:
        print("Load embedding of legal data")
        emb_legal_data = load_encoded_legal_data(legal_data_path=encoded_legal_data_path)
    else:
        print("Compute embedding of legal data against models")
        emb_legal_data: Dict[str, np.ndarray] = {}
        for model_path in tqdm(model_paths):
            print("Load model {}".format(model_path))
            model = load_model(model_dir=args.model_dir, model_path=model_path)
            print("Encode legal data using {}".format(model_path))
            emb_legal_data[model_path] = encode_legal_data(args.legal_dict_json, model=model)
    
    test_path = os.path.join(data_dir, test_json_path)
    data = json.load(open(test_path))
    items = data["items"]
    print("Num test question: {}".format(len(items)))
    
    print("Load BM25 model")
    bm25 = load_bm25(bm25_path=bm25_path)
    
    print("Load doc refer")
    with open(doc_refers_path, "rb") as doc_refer_file:
        doc_refers = pickle.load(doc_refer_file)
        
    print("Compute questions embedding against models")
    question_emb_dict = {}
    for model_path in tqdm(model_paths):
        model = load_model(model_dir=model_dir, model_path=model_path)
        question_emb_dict[model_path] = encode_question(questions=items, model=model)
        
    cross_encoder = load_cross_encoder(model_path=cross_encoder_name)
    
    weights = [0.25, 0.25, 0.125, 0.25, 0.125]
    
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
        
        cos_sim = []
        
        for idx_2, model_path in enumerate(model_paths):
            emb1 = question_emb_dict[model_path][question_id]
            emb2 = emb_legal_data[model_path]
            
            scores = util.cos_sim(emb1, emb2)
            
            cos_sim.append(weights[idx_2] * scores)
            
        cos_sim = torch.cat(cos_sim, dim=0)
        
        cos_sim = torch.sum(cos_sim, dim=0).squeeze(0).numpy()
        
        predictions = np.argsort(cos_sim)[::-1][:primary_top_n]
        scores = cos_sim[predictions]
        
        couples = []
        
        for idx_2, doc_idx in enumerate(predictions):
            couples.append([question, doc_refers[doc_idx][2]])
            
        pair_scores = cross_encoder.predict(couples)
        
        new_scores = 0.3 * pair_scores + 0.7 * scores
        new_scores = doc_scores[predictions] * new_scores
        max_score = np.max(new_scores)
        
        new_predictions = np.where(new_scores >= (max_score - range_score))[0]
        
        map_ids = predictions[new_predictions]
        
        new_scores = new_scores[new_scores >= (max_score - range_score)]
        
        if new_scores.shape[0] > 5:
            predictions_2 = np.argsort(new_scores)[::-1][:5]
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