import os
import time
import boto3
import torch
import traceback
import subprocess
import pandas as pd
import math
import webvtt
from models import tokenizer, bertModel
from datetime import datetime, timedelta

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


def strToTimedelta(s):
    t = datetime.strptime(s,"%H:%M:%S.%f")
    return timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)

def processCaptions(captions):
    noreturn = [c.replace('\n',",") for c in captions]
    allsentences = []
    for index, c in enumerate(noreturn):
        for sp in c.split(','):
            allsentences.append(sp)
    seen = set()
    seen_add = seen.add
    return " ".join([x for x in allsentences if not (x in seen or seen_add(x)) if x != ' '])

def dlYt(videoURL, file_name="subtitles"):
    videoID = videoURL.split("v=")[-1]
    subprocess.call(
            [
                "youtube-dl", "--write-auto-sub", "--sub-lang", 
                "en", "--skip-download","--ignore-errors", "-o",f"{file_name}.%(ext)s",
                "--cookies", "cookies.txt",
                f"https://www.youtube.com/watch?v={videoID}",
            ])
    b = os.path.isfile(file_name + '.en.vtt')
    if b:
        return file_name + '.en.vtt'
    else: 
        None

def captionsToDf(file_path, chunk_seconds = 10):
        vtt = webvtt.read(file_path)
        os.remove(file_path)
        df2 = pd.DataFrame([{'startTime':strToTimedelta(caption.start).total_seconds(),"endTime":strToTimedelta(caption.end).total_seconds(),"text":caption.text} for caption in vtt])
        lastTimestamp = max(df2.endTime.tolist())
        res = []
        for i in range(math.ceil(lastTimestamp//10) + 1):
            dfFilter = df2[((i*chunk_seconds < df2.endTime) & (df2.endTime < (i+1)*chunk_seconds))]
            res.append({"startTime":min(dfFilter.startTime.tolist()),
                        "endTime":max(dfFilter.endTime.tolist()),
                       "quote": processCaptions(dfFilter.text.tolist())})
        df2 = pd.DataFrame(res)
        df2["pred"] = df2.quote.map(lambda x: predict_sentiment(x))
        df2['label'] = df2.pred.map(lambda x: "sponsor" if x <= 0.5 else "content")
        df2['probability'] = df2.pred.map(lambda x: 1-x if x <= 0.5 else x )
        print(df2)
        df2.to_json('labeled_results.json')
        return df2.to_dict(orient='records')

def handler(sentence, url, model_path ='full-model.pt'):
    t = time.time()
    try:                       
        bertModel.load_state_dict(torch.load(model_path,map_location='cpu'))
        bertModel.eval()
        if sentence:
            pred = predict_sentiment(sentence)
            answer = {
                    'class': 'content' if pred >=0.5 else 'sponsor',
                    'probability': pred if pred >=0.5 else 1-pred
                }
            print(answer)
            print(f"Inference took {(time.time() - t):.3f}s")
            return answer
        if url:
            fp = dlYt(url)
            if fp:
                print(f"Video downloading and labeling took {(time.time() - t):.3f}s")
                return captionsToDf(fp)
            else:
                return {"error":"no captions found"}
    except Exception as e:
        traceback.print_exc()
        return {
            'error': str(e),
        }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Inference on trained model")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-s', '--sentence', help='Sentence to predict')
    group.add_argument('-u', '--url', help='Youtube URL to label')
    parser.add_argument('-w','--weights', help='model .pt file', default='full-model.pt', type=str)
    args = parser.parse_args()
    handler(args.sentence, args.url,args.weights)
