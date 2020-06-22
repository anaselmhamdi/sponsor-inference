# Labeling Sponsored Segments in a Youtube Video

## Install !

`git clone https://github.com/anaselmhamdi/sponsor-inference.git`  
`cd sponsor-inference`  
`pip3 install -r requirements.txt`  
`wget https://anas-models.s3.amazonaws.com/tut7-model.pt --directory-prefix="./app"`  

*You'll need around 2.5 Gb of free space and about 2 Gb of RAM to run optimally*

## Train freezing BERT parameters

Using a 70-15-15 training-validation-test split.  
The script will download and cache the BERT cased modes (11M+ parameters)

`python3 app/train.py -f training_file.json`

The training file should contain a list of objects such as:  

`[{"This video is sponsored by Squarespace","label":"sponsor"}, {"Welcome to this NLP tutorial","label":"content"}]`  

I trained this model with a dataset I built based off of [SponsorBlock's](https://github.com/ajayyy/SponsorBlock#sponsorblock) labels. 

I used youtube-dl to get the english auto captions when they were available on the diffrent videos.  

The dataset is publicly [available here on Kaggle.](https://www.kaggle.com/anaselmhamdi/sponsor-block-subtitles-80k?) 

It took 4 hours 48 minutes to train on a 16 Gb GPU on a [Kaggle kernel available here](https://www.kaggle.com/anaselmhamdi/transformers-sponsorblock)

It yielded a test accuracy of 93.85%.  

## Inference on a sentence

`python3 app/inference.py -s "This video is sponsored by Squarespace"`  

Should print:  

`{
    "class": "sponsor",
    "probability": 0.9990846614236943
}`

## Inference on a Youtube video

`python3 app/inference.py -u "https://www.youtube.com/watch?v=MlOPPuNv4E"`  

Will download the video captions, and label it by 10s chunks.  
The results will be written in the `labeled_results.json` file.  
