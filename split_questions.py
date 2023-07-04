import os
import argparse
import json
import random

from tqdm import tqdm
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", type=str, default="zac2021-ltr-data")
    
    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    
    np.random.seed(42)
    args = get_args()
    
    
    with open(os.path.join(args.data_dir, "train_question_answer.json")) as f:
        items = json.load(f)["items"]
    
    group_laws = {}
    
    for item in tqdm(items):
        law_id_set = []
        rev_articles = item["relevant_articles"]
        for arc in rev_articles:
            law_id = arc["law_id"]
            law_id = "-".join(law_id.split("/")[-1].split("-")[:2])
            law_id_set.append(law_id)
        law_id_set = list(set(law_id_set))
        for law_id in law_id_set:
            if law_id not in group_laws:
                group_laws[law_id] = []
            group_laws[law_id].append(item)
    
    train_list = []
    test_list = []
        
    # for item in tqdm(items):
    #     if np.random.rand() > 0.7:
    #         test_list.append(item)
    #     else:
    #         train_list.append(item)
    
    for law_id in group_laws:
        laws = group_laws[law_id]
        random.shuffle(laws)
        print(law_id, len(laws))
        train_list.extend(laws[:int(0.8 * len(laws))])
        test_list.extend(laws[int(0.8 * len(laws)):])    
    train_dict = {}
    train_dict["items"] = train_list
    
    test_dict = {}
    test_dict["items"] = test_list
    
    with open(os.path.join(args.data_dir, "for_train_question_answer.json"), "w") as f:
        json.dump(train_dict, f, indent=3)
        
    with open(os.path.join(args.data_dir, "for_test_question_answer.json"), "w") as f:
        json.dump(test_dict, f, indent=3)