import os
import time
import boto3
import torch
import traceback
from models import tokenizer, bertModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict_sentiment(sentence):
    init_token_idx = tokenizer.cls_token_id
    eos_token_idx = tokenizer.sep_token_id
    max_input_length = tokenizer.max_model_input_sizes['bert-base-cased']
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length-2]
    indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(bertModel(tensor))
    return prediction.item()


def handler(sentence, model_path ='tut7-model.pt'):
    t = time.time()
    try:                       
        bertModel.load_state_dict(torch.load(model_path,map_location='cpu'))
        bertModel.eval()
        pred = predict_sentiment(sentence)
        answer = {
                'class': 'content' if pred >=0.5 else 'sponsor',
                'probability': pred if pred >=0.5 else 1-pred
            }
        print(answer)
        print(f"Inference took {(time.time() - t):.3f}s")
        return answer
    except Exception as e:
        traceback.print_exc()
        return {
            'error': str(e),
        }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Inference on trained model")
    parser.add_argument('-s', '--sentence', help='Sentence to predict', required=True)
    parser.add_argument('-w','--weights', help='model .pt file', default='tut7-model.pt', type=str)
    args = parser.parse_args()
    handler(args.sentence, args.weights)
