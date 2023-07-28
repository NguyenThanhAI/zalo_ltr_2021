import pickle
import os
import numpy as np
import json
import torch
from tqdm import tqdm
from rank_bm25 import *
import argparse
import warnings 
from sentence_transformers import SentenceTransformer, util
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="phobert_base")
    parser.add_argument("--model_path", default="saved_model/bm25_Plus_04_06_model_full_manual_stopword", type=str)
    parser.add_argument("--sentence_bert_path", default="", type=str, help="path to round 1 sentence bert model")
    parser.add_argument("--data_path", default="zac2021-ltr-data", type=str, help="path to input data")
    parser.add_argument("--save_path", default="pair_data", type=str)
    parser.add_argument("--top_k", default=20, type=int, help="top k hard negative mining")
    parser.add_argument("--path_doc_refer", default="generated_data/doc_refers_saved.pkl", type=str, help="path to doc refers")
    parser.add_argument("--path_legal", default="generated_data/legal_dict.json", type=str, help="path to legal dict")
    args = parser.parse_args()
    
    
    # load training data from json
    data = json.load(open(os.path.join(args.data_path, "for_train_question_answer.json")))

    training_data = data["items"]
    print(len(training_data))

    # load bm25 model
    with open(args.model_path, "rb") as bm_file:
        bm25 = pickle.load(bm_file)

    with open(args.path_doc_refer, "rb") as doc_refer_file:
        doc_refers = pickle.load(doc_refer_file)

    doc_path = os.path.join(args.path_legal)
    df = open(doc_path)
    doc_data = json.load(df)
    
    # load hard negative model
    model = SentenceTransformer(args.sentence_bert_path)

    # add embedding for data
    # if you already have data with encoded sentence uncoment line 47 - 54
    import pickle
    embed_list = []
    for k, v in tqdm(doc_data.items()):
        embed = model.encode(v['title'] + ' ' + v['text'])
        doc_data[k]['embedding'] = embed

    with open(os.path.join("generated_data", f'legal_corpus_{args.model_name}_embedding.pkl'), 'wb') as pkl:
        pickle.dump(doc_data, pkl)

    with open(os.path.join("generated_data", f'legal_corpus_{args.model_name}_embedding.pkl'), 'rb') as pkl:
        data = pickle.load(pkl)
        
    top_k = args.top_k
    save_triplets = []
    
    for idx, item in tqdm(enumerate(training_data)):
        question_id = item["question_id"]
        question = item["question"]
        relevant_articles = item["relevant_articles"]
        actual_positive = len(relevant_articles)
        
        pos_passage = []
        
        for article in relevant_articles:
            concat_id = article["law_id"] + "_" + article["article_id"]
            pos_passage.append({"pos_id": concat_id, "passage": doc_data[concat_id]["title"] + " " + doc_data[concat_id]["text"]})
            break
            
        encoded_question  = model.encode(question)
        list_embs = []
        
        
        for k, v in data.items():
            emb_2 = torch.tensor(v['embedding']).unsqueeze(0)
            list_embs.append(emb_2)
            
        matrix_emb = torch.cat(list_embs, dim=0)
        all_cosine = util.cos_sim(encoded_question, matrix_emb).numpy().squeeze(0)
        predictions = np.argpartition(all_cosine, len(all_cosine) - top_k)[-top_k:]
        
        neg_passage = []
        
        for idx_2, idx_pred in enumerate(predictions):
            pred = doc_refers[idx_pred]
            
            check = 0
            for article in relevant_articles:
                if pred[0] == article["law_id"] and pred[1] == article["article_id"]:
                    check += 1
                    break
            
            if check == 0:
                concat_id = pred[0] + "_" + pred[1]
                neg_passage.append({"neg_id": concat_id, "passage": doc_data[concat_id]["title"] + " " + doc_data[concat_id]["text"]})
                
        for pos in pos_passage:
            for neg in neg_passage:
                save_triplets.append({"question_id": question_id,
                                      "pos_id": pos["pos_id"],
                                      "neg_id": neg["neg_id"],
                                      "triplets": (question, pos["passage"], neg["passage"])})
                
                
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, f"save_mnrl_triplets_{args.model_name}_top{top_k}.pkl"), "wb") as triplet_file:
        pickle.dump(save_triplets, triplet_file)