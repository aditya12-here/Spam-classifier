import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

import tarfile

import urllib

def fetch_spam_data(ham_file_name,spam_file_name,ham_dir_name,spam_dir_name):
  spam_root="http://spamassassin.apache.org/old/publiccorpus/"
  ham_url=spam_root+ham_file_name
  spam_url=spam_root+spam_file_name
  spam_path=Path()/'datasets'/'spams'
  spam_path.mkdir(parents=True,exist_ok=True)
  for dir_name,tar_name,url in ((ham_dir_name,'ham',ham_url),(spam_dir_name,'spam',spam_url)):
    if not (spam_path/dir_name).is_dir():
      path=(spam_path/tar_name).with_suffix('.tar.bz2')
      print('Downloading',path)
      urllib.request.urlretrieve(url,path)
      tar_bz2_file=tarfile.open(path)
      tar_bz2_file.extractall(path=spam_path,filter='data')
      tar_bz2_file.close()
  return [spam_path/dir_name for dir_name in(ham_dir_name,spam_dir_name)]

import email
import email.policy

def load_email(filepath):
  with open(filepath,'rb') as f:
    return email.parser.BytesParser(policy=email.policy.default).parse(f)

def get_email_structure(email):
  if isinstance(email,str):
    return email
  payload=email.get_payload()
  if isinstance(payload,list):
    multipart=','.join([get_email_structure(parts) for parts in payload])
    return f"multipart({multipart})"
  return email.get_content_type()

from collections import Counter

def structure_counter(emails):
  structures={}
  for i,email in enumerate(emails):
    structure=get_email_structure(email)
    structures.setdefault(structure,[]).append(i)
  return structures

import re
from html import unescape

def html_to_plain(html):
  text=re.sub(r'<head.*?>.*?</head>','',html,flags=re.S|re.I)
  text=re.sub(r'<a\s.*?>','HYPERLINK',text,flags=re.S|re.I)
  text=re.sub(r'<.*?>','',text,flags=re.S)
  text=re.sub(r'(\s*\n)+','\n',text,flags=re.S)
  return unescape(text)

from email.message import EmailMessage

def email_to_text(email):
  html_parts=[]
  for part in email.walk():
    ctype=part.get_content_type()
    if not ctype in ('text/plain','text/html'):
      continue
    try:
      content=ctype.get_content()
    except:
      content=str(part.get_payload())
    if ctype=='text/plain':
      return content
    else:
      html_parts.append(content)
  if html_parts:
    return "\n".join(html_to_plain(html) for html in html_parts)

import nltk
stemmer=nltk.PorterStemmer()

import os
os.system("python -m pip install urlextract")

import urlextract
url_extractor=urlextract.URLExtract()

from sklearn.base import BaseEstimator, TransformerMixin

class EmailToWordCounterTransformer(BaseEstimator,TransformerMixin):
  def __init__(self,strip_headers=True,lower_case=True,remove_punctuation=True,
               replace_urls=True,replace_numbers=True,stemming=True):
    self.strip_headers=strip_headers
    self.lower_case=lower_case
    self.remove_punctuation=remove_punctuation
    self.replace_urls=replace_urls
    self.replace_numbers=replace_numbers
    self.stemming=stemming
  def fit(self,X,y=None):
    return self
  def transform(self,X,y=None):
    X_transformed=[]
    for email in X:
      text=email_to_text(email) or ""
      if not self.strip_headers:
        headers='\n'.join(f"{key}:{value}" for key, value in email.items())
        text=headers + "\n" + text
      if self.lower_case:
        text=text.lower()
      if self.replace_urls and url_extractor is not None:
        urls=list(set(url_extractor.find_urls(text)))
        urls.sort(key=lambda url:len(url),reverse=True)
        for url in urls:
          text=text.replace(url,'URL')
      if self.replace_numbers:
        text=re.sub(r'\d+(?:\.\d*)?(?:[eE][+-]?\d+)?','NUMBER',text)
      if self.remove_punctuation:
        text=re.sub(r'\W+',' ', text)
      word_counters=Counter(text.split())
      if self.stemming and stemmer is not None:
        stemmed_word_counts=Counter()
        for word, count in word_counters.items():
          stemmed_word=stemmer.stem(word)
          stemmed_word_counts[stemmed_word]+=count
        word_counters=stemmed_word_counts

      X_transformed.append(word_counters)
    return np.array(X_transformed)

from scipy.sparse import csr_matrix

class WordCounterToVectorTransformer(BaseEstimator,TransformerMixin):
  def __init__(self,vocabulary_size=1000):
    self.vocabulary_size=vocabulary_size
  def fit(self,X,y=None):
    total_word=Counter()
    for word_count in X:
      for word, count in word_count.items():
        total_word[word]+=min(count,10)
    vocabulary=total_word.most_common()[:self.vocabulary_size]
    self.count_of_words=vocabulary
    self.vocabulary_={word:i+1 for i,(word,_) in enumerate(vocabulary)}
    return self
  def transform(self,X,y=None):
    rows=[]
    cols=[]
    data=[]
    for row,word_count in enumerate(X):
      for word,count in word_count.items():
        rows.append(row)
        cols.append(self.vocabulary_.get(word,0))
        data.append(count)
    return csr_matrix((data,(rows,cols)),shape=(len(X),self.vocabulary_size+1))