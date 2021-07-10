# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 19:06:18 2020

@author: shahzeb

IDE: Spyder(Python3.8)
"""
"""  
Introduction: From the problem statement the required is as follows: F1-Score and Accuracy score
for the given sentiment_analysis.txt. The file contains column0 which has corpus sentences and 
column1 which has the emotion for 0:Negative and 1:Positive.

The given dataset is in the form of Labeled data, hence supervised learning is obvious for data. And
pre-processing the data implementing Classification Algorithms is a good way to train
and test models for this paticular dataset.

Steps followed in this code assignment:
   
    A) Pre-Processing:
        
        1) Tokenizing: The dataset is tokenized with 3 different tokenizers. Namely, Word_tokenizer,
            Treebank_tokenizer and Regexp_tokenizer. And after that pre-processing step-2,3,4 are applied to
            all the 3 tokenizers. 
        2) Removing the Stop words.
        3) POS Tagging: It's resonable to do POS tagging after removing stop words as it'll remove
            unimportant words, thus we have to process only important words: saving time and computation power.
        4) After that lemmatization(WordNetLemmatizer is used in this assignment) is chosen over 
            stemmitization as it'll give the root word for any given word in the'
        
        5)Data Cleaning: Numbers,Hyphens,quotation marks,new line characters,apostrophes,etc 
            are removed from the txt.
    
    Note: I wanted to experiment with different tokenizers, lemmatizer,etc. The 3 different tokenizers 
    were considered, but only 1 lemmatizer was used throughout this project as the result from the
    tokenizers were considerably similar, varying by 4-5% , So given the time constraint I 
    only tried to do implement the Naive Bayers Classifier in CBOW for 3 tokenizers. And for 
    remining only Treebank Tokenizer was used along with WordNetlemmatizer.
    
    
    B) Splitting Dataset/Corpus:
        The dataset/corpus is split into 66.6% of Training data and 33.3% of test data. 
        Moreover, the data is shuffledeverytime the code is re-run such that different instances 
        are shuffled between test and train datasets.
    C) Feature Engineering: In order to run machine learning algorithms we need to convert 
    the text files into numerical feature vectors. Bag of Words (CountVectorizer()) is used
    in 1st for Naive Bayers and Random Forest Algorithms. A term-Document Matrix was constructed 
    from the CountVectorizer() containing words or terms along columns and sentences along rows.
    The bag of words only considers the count of words which is not a good practice for doing sentiment analysis. 
    Because some common words appear in many sentences which contain less importantnce. 
    Therefore,TF-IDF is used later in this assignment for Naive Bayers and Random Forest Algorithms
    which takes into account the word based upon its uniqueness.
    
    D) Model Construction: MultiNomial Naive Bayers was constructed from sklearn.model_selection
    library and Random Forest Classifier was constructed from sklearn.ensemble.
    E) Evaluation: The accuracy_score, F1-scores are printed into the console.As well as confusion matrix is also calculated.

"""
#import nltk
#nltk.download()

# importing necessary libraries 
import re
import nltk
import warnings 
from nltk.tokenize import RegexpTokenizer,word_tokenize,TreebankWordTokenizer

warnings.filterwarnings(action = 'ignore') 
import pandas as pd
X = pd.read_csv('sentiment_analysis.txt', sep="\t", header=None)
target=X[1]
#taking the output into target variable
#To print the corpus from the dataset
#corpus=X[0]
#for line in corpus:
#    print(line)

#importing stopwords library to later remove the stop words from our corpus
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from nltk.stem import WordNetLemmatizer    #WordLemmatizer is used to cut down the word back to it's root word
from nltk.corpus import wordnet            #wordnet is a lexical database for the English language, we'll use it to find POS tags for words
lemmatizer = WordNetLemmatizer()          # creating instance of class WordNetLemmatizer

print("The corpus dataset, Column 0: is for Sentence Corpus and  Column 1: is for emotions. For 0 value in Column 1 it's negative sentiment for the sentence and for value 1 its positive sentiment")
X
                            # (A) PRE-PROCESSING
#The function below converts nltk tag to wordnet tags for POS tagging.
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

# The function below is used for tokenization followed by removing stop words, POS Tagging words, Lemmatization. This function is for Wordtokenizer only.
def Wordtoken_lemmatize_sentence(sentence):
   
    tokenizer_word= word_tokenize(str(sentence))    #Tokenize the sentence 
    filtered_sentence = ' '.join([w for w in tokenizer_word if not w in stop_words]) #After tokenizing find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(filtered_sentence))  #nltk_tagged contains the POS tag that will be used by lemmatizer to effectively find root words
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #If no available POS tag then append the token as it is.
            lemmatized_sentence.append(word)
        else:        
            #Else use the POS tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag).lower()) #appending the words to sentances
    return " ".join(lemmatized_sentence)  #returning the lematized sentance

X['Wordtoken_cleaned_txt']=X[0].apply(Wordtoken_lemmatize_sentence)  #calling the above function(Wordtoken_lemmatize_sentence) and stores the values to X dataset in a new column named as Wordtoken_cleaned_txt.

X
# The function below is used for tokenization followed by removing stop words, POS Tagging words, Lemmatization. This function is for TreeBank Tokenizer only.
def Treetoken_lemmatize_sentence(sentence):
    treebank_tokenizer= TreebankWordTokenizer().tokenize(str(sentence))   #Tokenize the sentence 
    filtered_sentence = ' '.join([w for w in treebank_tokenizer if not w in stop_words]) #After tokenizing find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(filtered_sentence))    #nltk_tagged contains the POS tag that will be used by lemmatizer to effectively find root words
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
             #If no available POS tag then append the token as it is.
            lemmatized_sentence.append(word)
        else:        
            #Else use the POS tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag).lower())  #appending the words to sentances
    return " ".join(lemmatized_sentence)                    #returning the lematized sentance

X['TreeToken_cleaned_txt']=X[0].apply(Treetoken_lemmatize_sentence)  #calling the above function(Treetoken_lemmatize_sentence) and stores the values to X dataset in a new column named as TreeToken_cleaned_txt.

Regtokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')

# The function below is used for tokenization followed by removing stop words, POS Tagging words, Lemmatization. This function is for RegExp Tokenizer only.
def Regextoken_lemmatize_sentence(sentence):
    #tokenize the sentence and find the POS tag for each token
    regex_token=Regtokenizer.tokenize(str(sentence.lower()))    #Tokenize the sentence 
    filtered_sentence = ' '.join([w for w in regex_token if not w in stop_words])  #After tokenizing find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(filtered_sentence))     #nltk_tagged contains the POS tag that will be used by lemmatizer to effectively find root words
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
             #If no available POS tag then append the token as it is.
            lemmatized_sentence.append(word)
        else:        
             #Else use the POS tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag).lower())
    return " ".join(lemmatized_sentence)          #returning the lematized sentance


X['Regtoken_cleaned_txt']=X[0].apply(Regextoken_lemmatize_sentence) #calling the above function(Regextoken_lemmatize_sentence) and stores the values to X dataset in a new column named as Regtoken_cleaned_txt.

X
#print("The Dataset after applying the tokenizers, POS tagging, removing stop words and Lematizing", X)
print("------------------"*10)

'''
#The function cleantxt() as represented below makes sure only character from a-z 
#and A-Z are present and remaining ones are removed from the pre-processed dataset. 
#We apply this to all the 3 cleaned columns in the dataset X for each of the tokenizer used.'''

def cleantext(retext):
    return re.sub('[^a-zA-Z]',' ',str(retext))#.lower()
X['TreeToken_cleaned_txt']=X['TreeToken_cleaned_txt'].apply(cleantext)
X['Regtoken_cleaned_txt']=X['Regtoken_cleaned_txt'].apply(cleantext)
X['Wordtoken_cleaned_txt']=X['Wordtoken_cleaned_txt'].apply(cleantext)
#X_test['cleanedtxt']=X_test['Regtokenizedtxt'].apply(cleantext)
#The function removechar() as represented below removes the new line character if any present,
# if any apostrophes,hyphens,quotation marks,etc.
def removechar(text):
    text = re.sub('[0-9]+.\t','',str(text))
    # removing new line characters
    text = re.sub('\n ','',str(text))
    text = re.sub('\n',' ',str(text))
    # removing apostrophes
    text = re.sub("'s",'',str(text))
    # removing hyphens
    text = re.sub("-",' ',str(text))
    text = re.sub("— ",'',str(text))
    # removing quotation marks
    text = re.sub('\"','',str(text))
    # removing any reference to outside text
    text = re.sub("[\(\[].*?[\)\]]", "", str(text))
    return text
                 
#The below removechar is called for all the 3 columns in X dataset.
X['TreeToken_cleaned_txt']=X['TreeToken_cleaned_txt'].apply(removechar)
X['Regtoken_cleaned_txt']=X['Regtoken_cleaned_txt'].apply(removechar)
X['Wordtoken_cleaned_txt']=X['Wordtoken_cleaned_txt'].apply(removechar)

'''
             (B) Splitting Dataset/Corpus:
    #The dataset is split into 33.3% for test and 66.6%. This is taken randomly 
    any other split for test and train data is possible.
'''

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X['TreeToken_cleaned_txt'], target, test_size=0.33)   #test and train data split for the data to which treebank token is applied
X_train2, X_test2, y_train2, y_test2 = train_test_split(X['Regtoken_cleaned_txt'], target, test_size=0.33)      #test and train data split for the data to which Regextoken is applied
X_train3, X_test3, y_train3, y_test3 = train_test_split(X['Wordtoken_cleaned_txt'], target, test_size=0.33)     #test and train data split for the data to which wordtoken is applied
print(" The dataset is split as: For Training 66.6% and for Testing as 33.3%. \nThe sizes of: X_Train={0}\nX_Test={1}\ny_train={2}\ny_test={3}".format(X_train.size,X_test.size,y_train.size,y_test.size))

# As the X_train,y_train,X_train2,...,etc are of type pandas.series.series, but 
#for passing these values to CountVectorizer it needs to be of type pandas.Dataframe. 
#Hence, converting the type.
X_train=X_train.to_frame()
y_train=y_train.to_frame()
X_test=X_test.to_frame()
y_test=y_test.to_frame()

X_train2=X_train2.to_frame()
y_train2=y_train2.to_frame()
X_test2=X_test2.to_frame()
y_test2=y_test2.to_frame()

X_train3=X_train3.to_frame()
y_train3=y_train3.to_frame()
X_test3=X_test3.to_frame()
y_test3=y_test3.to_frame()

'''                (C) Feature Engineering
        Bag of Words:The important part is to find the features from the data to make
        machine learning algorithms works. In this case, we have text. We need to convert this 
        text into numbers that we can do calculations on. We use word frequencies. That is treating
        every document as a set of the words it contains. Our features will be the counts of each of 
        these words.
            The term-document matrix is constructed in the block below for each of the 3 tokenizers.
'''
from sklearn.feature_extraction.text import CountVectorizer
count_vect=CountVectorizer()
count_vect2=CountVectorizer()
count_vect3=CountVectorizer()

counts=count_vect.fit_transform(X_train['TreeToken_cleaned_txt'])   #fitting the data
counts2=count_vect2.fit_transform(X_train2['Regtoken_cleaned_txt'])
counts3=count_vect3.fit_transform(X_train3['Wordtoken_cleaned_txt'])

counts_test=count_vect.transform(X_test["TreeToken_cleaned_txt"])   #transform the data
counts_test2=count_vect2.transform(X_test2["Regtoken_cleaned_txt"])
counts_test3=count_vect3.transform(X_test3["Wordtoken_cleaned_txt"])


'''               (D) and (E)  Classifiers construction and evaluation :Different estimators are better 
            suited for different types of data and different problems. The Naive Byers is chosen for this
            dataset as it relies on a very simple representation of the document (called the bag of words representation). 
            Also, it recommended on sklearn cheatsheet that if the dataset is <100k and it's text then Naive 
            Bayers is a good option. However other algorithms maybe applied as well.
'''
from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()

nb.fit(counts, y_train)         #fitting the model for Treebank tokenizer model

nb2=MultinomialNB()
nb2.fit(counts2, y_train2)          #fitting the model for Regex tokenizer model

nb3=MultinomialNB()
nb3.fit(counts3, y_train3)      #fitting the model for word tokenizer model
print("-----------"*10)
print("For Multinomial Naive Bayers Classifier for Continuous Bag of Words (CBOW)")
print("Accuracy for NB using TreeBankTokenizer :", nb.score(counts_test, y_test))
print("Accuracy for NB using RegExp Tokenizer:", nb2.score(counts_test2, y_test2))
print("Accuracy for NB using Word Tokenizer:", nb3.score(counts_test3, y_test3))
y_pred=nb.predict(counts_test)      #evaluating the test set
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,classification_report  #importing all the metrics
confusion_matrix(y_test, y_pred)            # confusion matrix also known as error matrix, allows visualization of the performance of an algorithm

y_pred2=nb2.predict(counts_test2)           #evaluating the test set for Regex tokenizer model
y_pred3=nb3.predict(counts_test3)           #evaluating the test set for word tokenizer model
f1_score(y_test,y_pred, average="macro" )
print("--------"*10)
print("Classification report for Naive Bayers Classifier using WordTokenizer\n",classification_report(y_test,y_pred))
print("--------"*10)
print("Classification report for Naive Bayers Classifier using RegexTokenizer\n",classification_report(y_test2,y_pred2))
print("--------"*10)
print("Classification report for Naive Bayers Classifier using Treetokenizer\n",classification_report(y_test3,y_pred3))
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=1000, bootstrap=True, max_features='sqrt')
clf.fit(counts, y_train)                #fitting the Random Forest Model
y_pred=clf.predict(counts_test)         #evaluating the test set for random forest. Note: After here on only TreeBank tokenizer model
print("-----------"*10)
print("For Random Forest Classifier for Continuous Bag of Words (CBOW) considering TreeBankTokenizer")
print("The accuracy for RandomForest Classifier is :", accuracy_score(y_test, y_pred))
print("The F1-Score for RandomForest Classifier is :", f1_score(y_test,y_pred, average="macro" ))
'''
                    Feature Engineering
            TF-IDF: A term-Document Matrix was constructed from the CountVectorizer() containing words or terms along columns and sentences along rows.
    The bag of words only considers the count of words which is not a good practice for doing sentiment analysis. 
    Because some common words appear in many sentences which contain less importantnce. 
    Therefore,TF-IDF is used in this assignment for Naive Bayers and Random Forest Algorithms
    which takes into account the word based upon its uniqueness.

'''
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(counts)     #counts is the count of words from Bag of words. To which we fit and transform.
X_train_tfidf.shape                                     #Displayes the dimensions of the variable
from sklearn.naive_bayes import MultinomialNB
clf1 = MultinomialNB().fit(X_train_tfidf, y_train)
y_pred1 = clf1.predict(counts_test)

accuracy_score(y_test, y_pred1)                 #evaluating the test set
print("-----------"*10)
print("For Multinomial Naive Bayers Classifier for TF-iDF, considering TreeBankTokenizer")
print("The accuracy for Multinomial Naive Bayers Classifier is :", accuracy_score(y_test, y_pred1))
print("The F1-Score for Multinomial Naive Bayers Classifier is :", f1_score(y_test,y_pred1, average="macro" ))
clf_1_randomforest= RandomForestClassifier(n_estimators=1000, bootstrap=True, max_features='sqrt')
clf_1_randomforest.fit(X_train_tfidf, y_train)   #fitting the Random Forest Model
y_pred1=clf.predict(counts_test)                    #evaluating the test set for random forest. 
accuracy_score(y_test, y_pred1)                     #accuracy sore for the model 
print("-----------"*10)
print("For Random Forest Classifier for TF-iDF, considering TreeBankTokenizer")
print("The accuracy for RandomForest Classifier is :", accuracy_score(y_test, y_pred1))
print("The F1-Score for RandomForest Classifier is :", f1_score(y_test,y_pred1, average="macro" ))