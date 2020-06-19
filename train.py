import json
import jsonlines
import random
import pandas as pd
import numpy as np
import torch
import time
import os
from models import tokenize_and_cut, tokenizer, bertModel, optimizer
from torchtext import data
from utils import train, evaluate, epoch_time
from torch.nn import BCEWithLogitsLoss

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128 # BERT imposed

def data_prep(file):
    # Shuffles and splits 70-15-15

    df = pd.DataFrame(json.load(open(file,'r')))
    dicts = df.to_dict(orient='records')
    dictsProcessed = []
    for d in dicts:
        if d['quote'] != []:
            dictsProcessed.append({'quote':d['quote'],'label':d['label']})

    split1 = round(len(dictsProcessed) * 0.7)
    split2 = round(len(dictsProcessed) * 0.15)

    random.shuffle(dictsProcessed)

    with jsonlines.open('train_data.json', mode='w') as writer:
        writer.write_all(dictsProcessed[:split1])
    with jsonlines.open('validation_data.json', mode='w') as writer:
        writer.write_all(dictsProcessed[split1 + 1: split1 + 1 + split2])
    with jsonlines.open('test_data.json', mode='w') as writer:
        writer.write_all(dictsProcessed[-split2:])


def create_batches(file, batch_size):
    data_prep(file)
    init_token_idx = tokenizer.cls_token_id
    eos_token_idx = tokenizer.sep_token_id
    pad_token_idx = tokenizer.pad_token_id
    unk_token_idx = tokenizer.unk_token_id
    max_input_length = tokenizer.max_model_input_sizes['bert-base-cased']
    TEXT = data.Field(batch_first = True,
                    use_vocab = False,
                    tokenize = tokenize_and_cut,
                    preprocessing = tokenizer.convert_tokens_to_ids,
                    init_token = init_token_idx,
                    eos_token = eos_token_idx,
                    pad_token = pad_token_idx,
                    unk_token = unk_token_idx)

    LABEL = data.LabelField(dtype = torch.float)
    fields = {'quote': ('text', TEXT), 'label': ('label', LABEL)}
    train_data, validation_data, test_data = data.TabularDataset.splits(
                                path = './',
                                train = 'train_data.json',
                                validation= 'validation_data.json',
                                test = 'test_data.json',
                                format = 'json',
                                fields = fields
    )
    LABEL.build_vocab(train_data)
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, validation_data, test_data), 
    batch_size = batch_size, 
    sort_key = lambda x: x.text,
    sort_within_batch= False,
    device = device)
    return train_iterator, valid_iterator, test_iterator 

def handler(file, weights_file='model-weights.pt', N_EPOCHS=50):
    train_iterator, valid_iterator, test_iterator  = create_batches(file, BATCH_SIZE)
    # freezing bert params 
    model = bertModel.to(device)
    criterion = BCEWithLogitsLoss()
    criterion = criterion.to(device)

    for name, param in bertModel.named_parameters():                
        if name.startswith('bert'):
            param.requires_grad = False

    best_valid_loss = float('inf')

    model_path = weights_file

    if os.path.isfile(model_path):
            model.load_state_dict(torch.load(model_path))

    i = 0
    total_epoch_time = 0
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        total_epoch_time += epoch_mins * 60 + epoch_secs
        if valid_loss < best_valid_loss:
            i = 0
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_path)
        else:
            i +=1
            
        if i == 10:
            print("10 epochs without valid loss improvement. Stopping training.")
            break

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Inference on trained model")
    parser.add_argument('-f', '--file', help='Training file - Must be json array', required=True)
    parser.add_argument('-w','--weights_file', help='Model .pt file', default='model-weights.pt', type=str)
    parser.add_argument('-n','--epochs', default=50, type=int,help='Number of EPOCHS')
    args = parser.parse_args()
    handler(args.file, args.weights_file, args.epochs)
