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
from utils import load_model, encode_legal_data, encode_question, load_bm25, load_encoded_legal_data


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", type=str, default="zac2021_ltr_data")
    parser.add_argument("--train_json_path", type=str, default="for_train_question_answer.json")
    parser.add_argument("--legal_dict_json", default="generated_data/legal_dict.json", type=str)
    parser.add_argument("--model_dir", default="saved_model", type=str)
    parser.add_argument("--model_paths", default=None, type=str)
    parser.add_argument("--top_k", default=25, type=int, help="top k hard negative mining")
    parser.add_argument("--bm25_path", default="saved_model/bm25_Plus_04_06_model_full_manual_stopword", type=str)
    parser.add_argument("--encoded_legal_data_path", default="encoded_data/encoded_legal_data.pkl", type=str)
    parser.add_argument("--doc_refers_path", default="saved_model/doc_refers_saved", type=str)
    parser.add_argument("--save_dir", default="pair_data", type=str)
    
    args = parser.parse_args()
    
    return args

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
from utils import load_model, encode_legal_data, encode_question, load_bm25, load_encoded_legal_data


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", type=str, default="zac2021_ltr_data")
    parser.add_argument("--train_json_path", type=str, default="for_train_question_answer.json")
    parser.add_argument("--legal_dict_json", default="generated_data/legal_dict.json", type=str)
    parser.add_argument("--model_dir", default="saved_model", type=str)
    parser.add_argument("--model_paths", default=None, type=str)
    parser.add_argument("--top_k", default=25, type=int, help="top k hard negative mining")
    parser.add_argument("--bm25_path", default="saved_model/bm25_Plus_04_06_model_full_manual_stopword", type=str)
    parser.add_argument("--encoded_legal_data_path", default="encoded_data/encoded_legal_data.pkl", type=str)
    parser.add_argument("--doc_refers_path", default="saved_model/doc_refers_saved", type=str)
    parser.add_argument("--save_dir", default="pair_data", type=str)
    
    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    
    args = get_args()
    
    data_dir = args.data_dir
    train_json_path = args.train_json_path
    legal_dict_json = args.legal_dict_json
    model_dir = args.model_dir
    model_paths = args.model_paths.split(",")
    top_k = args.top_k
    bm25_path = args.bm25_path
    encoded_legal_data_path = args.encoded_legal_data_path
    doc_refers_path = args.doc_refers_path
    save_dir = args.save_dir
    
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
            
    train_path = os.path.join(data_dir, train_json_path)
    data = json.load(open(train_path))
    items = data["items"]
    print("Num train question: {}".format(len(items)))
    
    print("Load BM25 model")
    bm25 = load_bm25(bm25_path=bm25_path)
    
    print("Load doc refer")
    with open(doc_refers_path, "rb") as doc_refer_file:
        doc_refers = pickle.load(doc_refer_file)
    
    doc_path = os.path.join(legal_dict_json)
    df = open(doc_path)
    doc_data = json.load(df)
    
    print("Compute questions embedding against models")
    question_emb_dict = {}
    for model_path in tqdm(model_paths):
        model = load_model(model_dir=model_dir, model_path=model_path)
        question_emb_dict[model_path] = encode_question(questions=items, model=model)
        
    weights = [0.25, 0.25, 0.125, 0.25, 0.125]
    
    save_pairs = []
    
    for idx, item in tqdm(enumerate(items)):
        question_id = item["question_id"]
        question = item["question"]
        relevant_articles = item["relevant_articles"]
        actual_positive = len(relevant_articles)
        
        for article in relevant_articles:
            save_dict = {}
            save_dict["question"] = question
            concat_id = article["law_id"] + "_" + article["article_id"]
            save_dict["document"] = doc_data[concat_id]["title"] + " " + doc_data[concat_id]["text"]
            save_dict["relevant"] = 1
            save_pairs.append(save_dict)
            
        cos_sim = []
        
        for idx_2, model_path in enumerate(model_paths):
            emb1 = question_emb_dict[model_path][question_id]
            emb2 = emb_legal_data[model_path]
            
            scores = util.cos_sim(emb1, emb2)
            
            cos_sim.append(weights[idx_2] * scores)
            
        cos_sim = torch.cat(cos_sim, dim=0)
        
        cos_sim = torch.sum(cos_sim, dim=0).squeeze(0).numpy()
        new_scores = cos_sim
        predictions = np.argpartition(new_scores, len(new_scores) - top_k)[-top_k:]
        
        
        for idx_2, idx_pred in enumerate(predictions):
            pred = doc_refers[idx_pred]
            
            check = 0
            for article in relevant_articles:
                if pred[0] == article["law_id"] and pred[1] == article["article_id"]:
                    check += 1
                    break
                
                
            if check == 0:
                save_dict = {}
                save_dict["question"] = question
                concat_id = pred[0] + "_" + pred[1]
                save_dict["document"] = doc_data[concat_id]["title"] + " " + doc_data[concat_id]["text"]
                save_dict["relevant"] = 0
                save_pairs.append(save_dict)
                
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"save_combined_pairs_top{top_k}.pkl"), "wb") as pair_file:
        pickle.dump(save_pairs, pair_file)