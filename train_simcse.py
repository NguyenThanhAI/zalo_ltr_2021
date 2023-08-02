from sentence_transformers import SentenceTransformer, models, InputExample, losses, LoggingHandler
from torch.utils.data import DataLoader
import math
import pickle
import json
from sentence_transformers import evaluation
import logging
import argparse
import os
#### Just some code to print debug information to stdout

def load_legal(legal_dict_json):
    with open(legal_dict_json, "r") as f:
        doc_data = json.load(f)
    
    train_data = []
    
    for k, doc in doc_data.items():
        sent = doc_data[k]["title"] + " " + doc_data[k]["text"]
        train_data.append(InputExample(texts=[sent, sent]))
        
    return train_data
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", default="", type=str, help="path to your language model")
    parser.add_argument("--max_seq_length", default=256, type=int, help="maximum sequence length")
    parser.add_argument("--legal_dict_json", type=str, default="", help="path to saved pair data")
    parser.add_argument("--num_val", default=2500, type=int, help="number of eval data")
    parser.add_argument("--epochs", default=5, type=int, help="Number of training epochs")
    parser.add_argument("--saved_model", default="", type=str, help="path to savd model directory.")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate for training")
    parser.add_argument("--margin", type=float, default=0.5)
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    
    
    word_embedding_model = models.Transformer(args.pretrained_model, max_seq_length=args.max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    print(model)
    
    
    train_examples = load_legal(args.legal_dict_json)
    train_dataloader = DataLoader(train_examples, batch_size=args.batch_size, shuffle=True)
    
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    output_path = args.saved_model
    os.makedirs(output_path, exist_ok=True)
    
    warmup_steps = math.ceil(len(train_dataloader) * args.epochs * 0.1)

    model.fit(train_objectives=[(train_dataloader, train_loss)], 
              epochs=args.epochs, 
              warmup_steps=warmup_steps,
              optimizer_params={'lr': args.lr},
              save_best_model=True,
              output_path=output_path,
              use_amp=True,
              show_progress_bar=True)
    