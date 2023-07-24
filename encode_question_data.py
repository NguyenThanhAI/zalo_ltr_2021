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


def encode_question(questions: List[str], model: SentenceTransformer) -> Dict[str, np.ndarray]:
    emb_dict = {}
    for _, item in tqdm(enumerate(questions)):
        question_id = item["question_id"]
        question = item["question"]
        emb_dict[question_id] = model.encode(question)
    
    return emb_dict


def get_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", type=str, default="zac2021_ltr_data")
    parser.add_argument("--test_json_path", type=str, default="for_test_question_answering.json")
    parser.add_argument("--model_dir", default="saved_model", type=str)
    parser.add_argument("--model_paths", default=None, type=str)
    parser.add_argument("--save_dir", default="encoded_data", type=str)
    
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = get_args()
    
    data_dir = args.data_dir
    test_json_path = args.test_json_path
    model_dir = args.model_dir
    model_paths = args.model_paths.split(",")
    save_dir = args.save_dir
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    test_path = os.path.join(data_dir, test_json_path)
    data = json.load(open(test_path))
    items = data["items"]
    print("Num test question: {}".format(len(items)))
    
    print("Compute questions embedding against models")
    question_emb_dict = {}
    for model_path in tqdm(model_paths):
        model = load_model(model_dir=model_dir, model_path=model_path)
        question_emb_dict[model_path] = encode_question(questions=items, model=model)

    with open(os.path.join(save_dir, "encode_{}.pkl".format(os.path.splitext(os.path.basename(test_json_path))[0])), "wb") as f:
        pickle.dump(question_emb_dict, f)
        