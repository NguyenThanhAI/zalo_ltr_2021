import os
import argparse
import json
from collections import Counter

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
    
    
    with open(os.path.join(args.data_dir, "legal_corpus.json")) as f:
        items = json.load(f)
    
    group_laws = {}
    
    for item in tqdm(items):
        law_id = item["law_id"]
        law_id = "-".join(law_id.split("/")[-1].split("-")[:2])
        if law_id not in group_laws:
            group_laws[law_id] = []
        group_laws[law_id].append(item)
        
    for law_id in group_laws:
        print(law_id, len(group_laws[law_id]))