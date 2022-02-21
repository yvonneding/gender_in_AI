import csv
import urllib, urllib.request
import json
import feedparser
from tqdm import tqdm
import sys
import time
import requests

csv.field_size_limit(sys.maxsize)

def cosine_match(str1, str2):
    from nltk.tokenize import RegexpTokenizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import CountVectorizer
    corpus = [str1, str2]
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    count_vectorizer = CountVectorizer(ngram_range=(1,1),tokenizer=token.tokenize, stop_words='english')
    # if corpus:
    #     return False
    vec = count_vectorizer.fit_transform(corpus)
    score = cosine_similarity(vec, vec)[0][1]
    # print(score)
    if score > 0.90:
        return True
    return False

with open("ai_paper_features_100k_updated.csv") as f, open("new.csv", 'a') as w:
    csv_reader = csv.reader(f)
    fieldnames = ["paper_title", "gs_id", "s2_id", "authors_name", "arxiv_id", "pdf_link", "source_link"]
    csv_writer = csv.DictWriter(w, fieldnames=fieldnames)
    first_row = next(csv_reader)

    
    t = 1
    for row in tqdm(csv_reader):
        if(t % 1000 == 0):
            time.sleep(900)
        t += 1
        gs_id = row[4]
        s2_id = row[3]
        title = row[5]
        source = "https://paperswithcode.com/api/v1/search/?q={}"
        query = urllib.parse.quote(title)
        url = source.format(query)
        headers = {"accept": "application/json", "X-CSRFToken": "pdUStGX9Mv9H5iUbyhASDIprrd1V65kw5B6TBvKc3sn5nzvFn4CbZ9IuQvHrpEas"}
        data = requests.get(url, headers=headers)
        result = data.json()
        authors_name = []
        arxiv_id = ''
        pdf_link = ''
        source_link = ''

        current_row = {"paper_title": title, "gs_id": gs_id, "s2_id": s2_id}

        if(result["count"] != 0):
            searched_title = result["results"][0]["paper"]["title"]
            if (cosine_match(title, searched_title)): 
                for each in result["results"][0]["paper"]["authors"]:
                    authors_name.append(each)
                arxiv_id = result["results"][0]["paper"]["arxiv_id"]
                pdf_link = result["results"][0]["paper"]["url_pdf"]
                source_link = url

        else:
            source = 'http://export.arxiv.org/api/query?search_query=all:{}&start=0&max_results=3'
            url = source.format(query)
            data = urllib.request.urlopen(url)
            feed = feedparser.parse(data)
            try:
                for i in range(3):
                    searched_title = feed["entries"][i].title
                    if (cosine_match(title, searched_title)):
                        author_names = feed["entries"][i].authors
                        for each in author_names:
                            authors_name.append(each['name'])
                        arxiv_id = feed["entries"][i].id
                        pdf_link = feed["entries"][i].link
                        source_link = url
            except:
                pdf_link = ''

        current_row["authors_name"] = authors_name
        current_row["arxiv_id"] = arxiv_id
        current_row["pdf_link"] = pdf_link
        current_row["source_link"] = source_link

        csv_writer.writerow(current_row)
        
        
# import pandas as pd

# df = pd.read_csv("new.csv")
# df.drop_duplicates(subset ="Unnamed: 0", keep = False, inplace = True)
# df.to_csv("ai_100k_updated_authors.csv")




# title = "The Higgs legacy of the LHC Run I"
# query = urllib.parse.quote(title)
# source = "https://paperswithcode.com/api/v1/search/?q={}"
# url = source.format(query)
# headers = {"accept": "application/json", "X-CSRFToken": "pdUStGX9Mv9H5iUbyhASDIprrd1V65kw5B6TBvKc3sn5nzvFn4CbZ9IuQvHrpEas"}

# data = requests.get(url, headers=headers)
# result = data.json()
# print(result)
# if(result["count"] != 0):
#     searched_title = result["results"][0]["paper"]["title"]
#     if (cosine_match(title, searched_title)): 
#         print(result["results"][0]["paper"])