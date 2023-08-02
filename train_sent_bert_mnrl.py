from sentence_transformers import SentenceTransformer, models, InputExample, losses, LoggingHandler
from torch.utils.data import DataLoader
import pickle
from sentence_transformers import evaluation
import logging
import argparse
import os
#### Just some code to print debug information to stdout


def load_triplets_data(triplet_data_path):
    with open(triplet_data_path, "rb") as f:
        triplets = pickle.load(f)
        
    return triplets


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", default="", type=str, help="path to your language model")
    parser.add_argument("--max_seq_length", default=256, type=int, help="maximum sequence length")
    parser.add_argument("--triplet_data_path", type=str, default="", help="path to saved pair data")
    parser.add_argument("--round", default=1, type=int, help="training round ")
    parser.add_argument("--num_val", default=200, type=int, help="number of eval data")
    parser.add_argument("--epochs", default=5, type=int, help="Number of training epochs")
    parser.add_argument("--saved_model", default="", type=str, help="path to savd model directory.")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate for training")
    parser.add_argument("--scale", type=float, default=0.5)
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    
    if args.round == 1:
        print(f"Training round 1")
        word_embedding_model = models.Transformer(args.pretrained_model, max_seq_length=args.max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        print(model)
    else:
        print("Training round 2")
        model = SentenceTransformer(args.pretrained_model)
        print(model)
        
    save_triplets = load_triplets_data(args.triplet_data_path)
    
    print(f"There are {len(save_triplets)} triplets sentences.")
    
    train_examples = []
    sent1 = []
    sent2 = []
    scores = []
    num_train = len(save_triplets) - args.num_val
    
    for idx, triplet in enumerate(save_triplets):
        query = triplet["passage"]
        pos_sent = triplet["positive"]["passage"]
        if idx < num_train:

            for neg in triplet["negative"]:
                neg_sent = neg["passage"]
                example = InputExample(texts=[query, pos_sent, neg_sent])
                train_examples.append(example)
        else:
            sent1.append(query)
            sent2.append(pos_sent)
            scores.append(float(1))
            
            for neg in triplet["negative"]:
                sent1.append(query)
                neg_sent = neg["passage"]
                sent2.append(neg_sent)
                scores.append(float(0))
        
            
    print("Number of sample for training: ", len(train_examples))
    
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size, collate_fn=model.smart_batching_collate)
    train_loss = losses.MultipleNegativesRankingLoss(model=model, scale=args.scale)
    
    output_path = args.saved_model
    os.makedirs(output_path, exist_ok=True)
    
    #evaluator = evaluation.EmbeddingSimilarityEvaluator(sent1, sent2, scores)
    evaluator = evaluation.BinaryClassificationEvaluator(sent1, sent2, scores)
    
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=args.epochs,
              warmup_steps=1000,
              optimizer_params={'lr': args.lr},
              save_best_model=True,
              evaluator=evaluator,
              evaluation_steps=1000,
              output_path=output_path,
              use_amp=True,
              show_progress_bar=True)