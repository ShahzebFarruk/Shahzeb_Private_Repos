# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 11:44:34 2021

@author: shahz
"""

import random
import pickle
import spacy
from spacy.lang.en import English
import nltk
from nltk.corpus import wordnet as wn
import pandas as pd

from gensim import corpora
from nltk.stem.wordnet import WordNetLemmatizer

from nltk import corpus


import gensim
class most_searched_topic_logic():
    def most_searched_topic_logic_function(self,records):
        content_list1=records
        spacy.load('en_core_web_sm')
        parser = English()
        def tokenize(text):
            lda_tokens = []
            tokens = parser(text)
            for token in tokens:
                if token.orth_.isspace():
                    continue
                elif token.like_url:
                    lda_tokens.append('URL')
                elif token.orth_.startswith('@'):
                    lda_tokens.append('SCREEN_NAME')
                else:
                    lda_tokens.append(token.lower_)
            return lda_tokens


        def get_lemma(word): #for checking if a synonym is available or not
            lemma = wn.morphy(word)
            if lemma is None:
                return word
            else:
                return lemma
        

        def get_lemma2(word):       #lemmatize based upon if the system has found any synonyms or else lemmatize
            return WordNetLemmatizer().lemmatize(word) 

        en_stop = set(nltk.corpus.stopwords.words('english'))
        text='at the american school of dubai students cultivate a lifetime appreciation, enjoyment and love of the arts through creating, performing and experiencing any one of the disciplines. through the wide range of arts offerings at asd, both during and after school, students are exposed to and develop an understanding of a variety of the visual and performing arts. by working creatively and gaining competence in various artistic genres and media, our students develop an aesthetic understanding of the arts that will continue throughout their lives.\n\na wide range of visual and performing arts opportunities are available to elementary, middle and high school students. students may learn dance, music, visual arts, theater, acting, choir and band. annual productions include full-length musicals in high school and middle school, plays, an elementary musical, variety shows, art shows and more.'


        def prepare_text_for_lda(text):
            tokens = tokenize(text)
            tokens = [token for token in tokens if len(token) > 4]
            tokens = [token for token in tokens if token not in en_stop]
            tokens = [get_lemma(token) for token in tokens]
            tokens = [get_lemma2(token) for token in tokens]
            return tokens
        #text='at the american school of dubai students cultivate a lifetime appreciation, enjoyment and love of the arts through creating, performing and experiencing any one of the disciplines. through the wide range of arts offerings at asd, both during and after school, students are exposed to and develop an understanding of a variety of the visual and performing arts. by working creatively and gaining competence in various artistic genres and media, our students develop an aesthetic understanding of the arts that will continue throughout their lives.\n\na wide range of visual and performing arts opportunities are available to elementary, middle and high school students. students may learn dance, music, visual arts, theater, acting, choir and band. annual productions include full-length musicals in high school and middle school, plays, an elementary musical, variety shows, art shows and more.'

        text_data = []
        for line in content_list1:
            tokens=prepare_text_for_lda(line)
            text_data.append(tokens)
            

        print(text_data)
        tokens = prepare_text_for_lda(text)
        text_data.append(tokens)
        #print(tokens)
        #print(text_data)



        dictionary = corpora.Dictionary(text_data)


        corpus = [dictionary.doc2bow(text) for text in text_data]


        pickle.dump(corpus, open('corpus.pkl', 'wb'))
        dictionary.save('dictionary.gensim')


        NUM_TOPICS = 10
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
        ldamodel.save('data/intermediate/lda_model_all')

        topics = ldamodel.print_topics(num_words=5)
        for topic in topics:
            print(topic)
spacy.load('en_core_web_sm')
essay_dataset=pd.read_csv('data/intermediate/essays_and_scores.csv', encoding = "ISO-8859-1")
essays=essay_dataset['essay']
meth=most_searched_topic_logic()
q=meth.most_searched_topic_logic_function(essays)