import string
from underthesea import word_tokenize
import os
import json

from typing import List, Dict
import pickle
from sentence_transformers import SentenceTransformer
from rank_bm25 import *
from tqdm import tqdm

number = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
chars = ["a", "b", "c", "d", "đ", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o"]
stop_word = number + chars + ["của", "và", "các", "có", "được", "theo", "tại", "trong", "về", 
            "hoặc", "người",  "này", "khoản", "cho", "không", "từ", "phải", 
            "ngày", "việc", "sau",  "để",  "đến", "bộ",  "với", "là", "năm", 
            "khi", "số", "trên", "khác", "đã", "thì", "thuộc", "điểm", "đồng",
            "do", "một", "bị", "vào", "lại", "ở", "nếu", "làm", "đây", 
            "như", "đó", "mà", "nơi", "”", "“"]

def remove_stopword(w):
    return w not in stop_word
def remove_punctuation(w):
    return w not in string.punctuation
def lower_case(w):
    return w.lower()

def bm25_tokenizer(text):
    tokens = word_tokenize(text)
    tokens = list(map(lower_case, tokens))
    tokens = list(filter(remove_punctuation, tokens))
    tokens = list(filter(remove_stopword, tokens))
    return tokens

def calculate_f2(precision, recall):        
    return (5 * precision * recall) / (4 * precision + recall + 1e-20)

def load_json(path):
    return json.load(open(path))


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