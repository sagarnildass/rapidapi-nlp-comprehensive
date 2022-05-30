import uvicorn
from fastapi import FastAPI
from fastapi.logger import logger
from fastapi import APIRouter, Depends, HTTPException
from ktrain.text.zsl import ZeroShotClassifier
from ktrain.text import shallownlp as snlp
from ktrain.text.summarization import TransformerSummarizer
from ktrain.text.translation import EnglishTranslator, Translator
from ktrain.text.kw import KeywordExtractor
from ktrain.text.textextractor import TextExtractor
import tika
from typing import List
import pandas as pd
import os
import requests
import codecs
import lxml.html.clean
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

zsl = ZeroShotClassifier()
ts = TransformerSummarizer()
#chinese_translator = EnglishTranslator(src_lang='zh')
#arabic_translator = EnglishTranslator(src_lang='ar')
#russian_translator = EnglishTranslator(src_lang='ru')
#german_translator = EnglishTranslator(src_lang='de')
#french_translator = EnglishTranslator(src_lang='fr')
#spanish_translator = EnglishTranslator(src_lang='es')

#ner = snlp.NER('en') #en, zh, ru

# Instantiate the app
app = FastAPI()

def _removeNonAscii(s):
    return "".join(i for i in s if  ord(i)<128)

def make_lower_case(text):
    return text.lower()

def remove_stop_words(text):
    text = text.split()
    stopword = stopwords.words('english')
    newStopWords = ['color', 'rgba', 'html', 'hover']
    stopword.extend(newStopWords)
    stops = set(stopword)
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text

def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

def remove_punctuation(text):
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text)
    text = " ".join(text)
    return text

def remove_single_character(text):
    return ' '.join( [w for w in text.split() if len(w)>1] )


# Predict Topic
@app.post("/zeroshotclassifier/")
def predict_topic(sentence: List[str], labels: List[str]):
    #logger.info(labels)
    pred = zsl.predict(sentence, labels=labels, include_labels=True)
    if pred is None:
        raise HTTPException(status_code=404, detail="Could not make prediction")

    return pd.DataFrame(pred).to_dict(orient="records")


@app.post("/sentimentclassifier/")
def predict_sentiment(sentence: List[str]):
    #logger.info(labels)
    pred = zsl.predict(sentence, labels=["positive", "negative"], include_labels=True)
    if pred is None:
        raise HTTPException(status_code=404, detail="Could not make prediction")

    return pd.DataFrame(pred).to_dict(orient="records")


@app.post("/emotionclassifier/")
def predict_emotion(sentence: List[str]):
    #logger.info(labels)
    pred = zsl.predict(sentence, labels=["happy", "sad", "angry", "hateful", "neutral", "inquisitive", "informative"], include_labels=True)
    if pred is None:
        raise HTTPException(status_code=404, detail="Could not make prediction")

    return pd.DataFrame(pred).to_dict(orient="records")


#NER
@app.post("/document_search/")
def search_document(list_of_docs: List[str], search_strings: List[str]):
    pred = snlp.search(search_strings, list_of_docs, keys=['doc'+str(i+1) for i in range(len(list_of_docs))])
    if pred is None:
        raise HTTPException(status_code=404, detail="Could not make prediction")
    return pd.DataFrame(pred, columns=['Document', 'Search_String', 'Occurences']).to_dict(orient="records")

#summarizer
@app.post("/text_summarize/")
def summarize(text: List[str]):
    df = pd.DataFrame(text, columns=['text'])
    df['summary'] = df['text'].map(ts.summarize)
    
    return dict(df['summary'])

#translator
#@app.post("/translator/")
#def translate_to_english(text: str, language: str):
#    if language=='zh':
#        pred = chinese_translator.translate(text)
#    elif language=='ar':
#        pred = arabic_translator.translate(text)
#    elif language=='ru':
#        pred = russian_translator.translate(text)
#    elif language=='de':
#        pred = german_translator.translate(text)
#    elif language=='fr':
#        pred = french_translator.translate(text)
#    elif language=='es':
#        pred = spanish_translator.translate(text)
#    return pred


#research paper keyword extractor
@app.post("/research_paper_keyword_extractor/")
def research_paper_keyword_extract(url: List[str]):


    url = url[0]

    try:
        os.system('wget --user-agent="Mozilla" {} -O /tmp/{} -q'.format(url, url.split('/')[-1]))
    except:
        raise HTTPException(status_code=404, detail="URL is not valid")

    text = TextExtractor().extract('/tmp/{}'.format(url.split('/')[-1]))
    

    kwe = KeywordExtractor()

    keywords = kwe.extract_keywords(text, candidate_generator='ngrams')
    
    os.system('rm /tmp/{}'.format(url.split('/')[-1]))

    return {'keywords':keywords}


#website keyword extractor
@app.post("/website_keyword_extractor/")
def website_keyword_extract(url: List[str]):

    
    url = url[0]

    try:
        req = requests.get(url, 'html.parser')
        if url.split('/')[-1] == '':
            html_output_name = url.split('/')[-2]
        else:
            html_output_name = url.split('/')[-1]
        with open(html_output_name, 'w') as f:
            f.write(req.text)
            f.close()

        f=codecs.open(html_output_name, 'r')
        text = f.read()
        text = lxml.html.clean.clean_html(text)
        text = remove_html(text)
        text = _removeNonAscii(text)
        text = make_lower_case(text)
        text = remove_stop_words(text)
        text = remove_punctuation(text)
        text = remove_single_character(text)

        kwe = KeywordExtractor()

        keywords = kwe.extract_keywords(text, candidate_generator='noun_phrases')

        os.system('rm {}'.format(html_output_name))
        return {url:keywords}
    except:
        raise HTTPException(status_code=404, detail="URL is not valid")



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

