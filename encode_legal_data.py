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
from utils import bm25_tokenizer

from sentence_transformers import SentenceTransformer, util


def load_model(model_dir: str, model_path: str) -> SentenceTransformer:
    model_path = os.path.join(model_dir, model_path)
    model = SentenceTransformer(model_name_or_path=model_path)

    return model


def encode_legal_data(legal_dict_json, model: SentenceTransformer) -> np.ndarray:
    with open(legal_dict_json, "r") as f:
        doc_data = json.load(f)
        
    emb_list = []
    print("Num articles: {}".format(len(doc_data.items())))
    for k, doc in tqdm(doc_data.items()):
        emb = model.encode(doc_data[k]["title"] + " " + doc_data[k]["text"])
        emb_list.append(emb)
        
    emb_array = np.array(emb_list)
    
    return emb_array
        
    
    
    
def get_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--legal_dict_json", default="generated_data/legal_dict.json", type=str)
    parser.add_argument("--model_dir", default="saved_model", type=str)
    parser.add_argument("--save_dir", default="encoded_data", type=str)
    
    
    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    args = get_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    model_paths = []
    
    model_to_embed: Dict[str, np.ndarray] = {}
    
    for model_path in tqdm(model_paths):
        print("Load model {}".format(model_path))
        model = load_model(model_dir=args.model_dir, model_path=model_path)
        print("Encode legal data using {}".format(model_path))
        model_to_embed[model_path] = encode_legal_data(args.legal_dict_json, model=model)
        
        print("Shape of embedding: {}".format(model_to_embed[model_path].shape))
    
    with open(os.path.join(args.save_dir, "encoded_legal_data.pkl"), "wb") as f:
        pickle.dump(model_to_embed, f)
        
        