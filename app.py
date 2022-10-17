from flask import Flask, render_template, redirect, request
from bs4 import BeautifulSoup as bs
import pandas as pd
from splinter import Browser
import requests

#------------------------------
# ML IMPORTS
#-----------------------------
import pickle
from sklearn.pipeline import Pipeline

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

port_stem = PorterStemmer()
nltk.download('stopwords')
def stemming(content):
        stemmed_content = re.sub('[^a-zA-Z]',' ',content)
        stemmed_content = stemmed_content.lower()
        stemmed_content = stemmed_content.split()
        stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
        stemmed_content = ' '.join(stemmed_content)
        return stemmed_content


app = Flask(__name__)

#--------------------------------------
# Flask Routes (website URL's)
#--------------------------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

#--------------------------------------
# Flask API CALLS (javascript calling python code)
#--------------------------------------

# test
@app.route("/api/exampleScrape")
def exampleScrape(testParameter):
    return testParameter

# web scrapper -  ARTICLES
# This only returns the scrapped article and headline 
@app.route("/api/ArticleScrape/<path:input>")
def article_reader(input):

    #variables
    url = input
    response = requests.get(url)
    soup = bs(response.text, 'html.parser')

    title = soup.title.text.strip()

    articles = soup.body.find_all('p')
    article_text = ""
    for article in articles:
        article_text += article.text

    scraped_data = {
        'title' : title,
        'text' : article_text,
    }

    return scraped_data


    # web scrapper -  general article scraper and returns result
@app.route("/api/ArticleAnalysis/<path:input>")
def ArticleAnalysis(input):

    #variables
    url = input
    response = requests.get(url)
    soup = bs(response.text, 'html.parser')

    title = soup.title.text.strip()
    title = title.split("|")[0]

    articles = soup.body.find_all('p')
    article_text = ""
    for article in articles:
        article_text += article.text

    # CNN author
    # authour = soup.find('span', class_='byline__name')
    # authour = authour.text.strip()

    # ML MODEL
    with open('models/ben_test.pickle', 'rb') as picklefile:
        saved_pipe = pickle.load(picklefile)

    # ML MODEL DATA PROCESSING
    data = {'title': [title],
    'text':[article_text],}
    input_data = pd.DataFrame.from_dict(data)
    content = input_data['title']+''+input_data['text']
    content=content.str.lower()
    content = content.apply(stemming)
    X = content.values
    result = saved_pipe.predict(X)
    result=str(result[0])
    
    return result


    # web scrapper -  CNN article only
@app.route("/api/ArticleAnalysis2/<path:input>")
def ArticleAnalysis2(input):

    #variables
    url = input
    response = requests.get(url)
    soup = bs(response.text, 'html.parser')

    title = soup.title.text.strip()
    title = title.split("|")[0]

    articles = soup.body.find_all('p')
    article_text = ""
    for article in articles:
        article_text += article.text

    #  author
    # if soup.find('span', class_='byline__name') != '':
    #     author = soup.find('span', class_='byline__name')
    #     author = author.text.strip()
    # else:
    #     author = ''

    # ML MODEL
    #with open('models/naive_bayes_ben__headline_body.pickle', 'rb') as picklefile:
    with open('models/regression_pipeline.pickle', 'rb') as picklefile:
        saved_pipe = pickle.load(picklefile)

    # ML MODEL DATA PROCESSING
    data = {'title': [title],
    'text':[article_text],}
    input_data = pd.DataFrame.from_dict(data)
    content = input_data['title']+''+input_data['text']
    X = content.values
    result = saved_pipe.predict(X)
    result=str(result[0])

    return result

if __name__ == '__main__':
    app.run(debug=True)

