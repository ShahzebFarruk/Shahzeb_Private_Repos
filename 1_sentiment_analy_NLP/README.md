<p align="center">
  <img src="https://user-images.githubusercontent.com/61950234/115967868-060e0880-a503-11eb-828e-4057f8bc2986.png?raw=true" alt="Sentiment Analysis Outputs"/>
</p>
 

The repo contains two code files containing same code, one is named as [sentiment_analysis_solution_jupyternotebook.ipynb](https://github.com/ShahzebFarruk/Sentiment_Analysis/blob/master/sentiment_analysis_solution_jupyternotebook.ipynb) if you want to run on jupyter Notebook please use this. Or else there's a python file named as [sentiment_analysis_solution_code.py](https://github.com/ShahzebFarruk/Sentiment_Analysis/blob/master/sentiment_analysis_solution_code.py) please run it using any IDE which has all the libraries installed in it. I used Anaconda with Spyder.

### Libraries Needed:
   * nltk
   * pandas
   * sklearn
   * numpy


### Note:
  The code was run and executed in both Jupyter notebook as well as Spyder IDEs. It is assumed that the user has installed anaconda from https://www.anaconda.com/products/individual. Open the anaconda IDE, then click on install for jupyternotebook or spyder by clicking at the install button if not installed. If already installed click 'launch'.

If anaconda is not installed follow the below Environment setting steps:

### Envirnoment Setting Up steps:
   1)Install anaconda IDE if not already installed. Go to website https://www.anaconda.com/products/individual and install the anaconda IDE.
   
   2)install python latest version
   
   3)open anaconda shell and type 
conda install -c anaconda nltk
type 'y' and press ENTER.

### For running code on Spyder

   4) Open Spyder IDE 
   
   5)In console type: 
import nltk
nltk.download()

then NLTK downloader pop-up will show on screen select all the install all nltk libraries, if already installed nltk library please do not do this step.

   6) Load the code sentiment_analysis_solution_code.py file. And click on run button. Results will be displayed in Console.

### For running code on Jupyter Notebook:

   4) Open Jupyter notebook from anaconda. 
   
   5)If using nltk for the 1st time type and run:
import nltk
nltk.download()
	If already installed nltk library please ignore this step.
    6) Load the file named as sentiment_analysis_solution_jupyternotebook.ipynb and click on 'restart the kernel, and re-run the whole code'. Results will be displayed in Console.

### What is this repository for? ###

* It contains the sentiment analysis solution code. In this code assignment various concepts of tokenizatation, lemmatization, POS tagging, stop words removal, noise removal, 2 different Classifiers as well as Bag of Words and TF-IDF methods are used. In order to preprocess then test and train the data corpus on Random Forest, Naive Bayers Algorithms. Ultimately, the overall accuracy and F1-scores ranges between 75-85%, as we are using train_test_split method from sklearn.model_selection which will split the dataset X into 33.3% Test and 66.6% train set randomly each time its re-run. 
## Introduction: ##
From the problem statement the required is as follows: F1-Score and Accuracy score for the given sentiment_analysis.txt. The file contains column0 which has corpus sentences and 
column1 which has the emotion for 0:Negative and 1:Positive.

The given dataset is in the form of Labeled data, hence supervised learning is obvious for data. And pre-processing the data implementing Classification Algorithms is a good way to train and test models for this paticular dataset.

### Steps followed in this code assignment: 

###  A) Pre-Processing:
 * 1) Tokenizing: The dataset is tokenized with 3 different tokenizers. Namely, Word_tokenizer, Treebank_tokenizer and Regexp_tokenizer. And after that pre-processing step-2,3,4 are applied to
        all the 3 tokenizers. 
 * 2) Removing the Stop words.
 * 3) POS Tagging: It's resonable to do POS tagging after removing stop words as it'll remove unimportant words, thus we have to process only important words: saving time and computation power.
*  4) After that lemmatization(WordNetLemmatizer is used in this assignment) is chosen over stemmitization as it'll give the root word for any given word in the'     
*  5)Data Cleaning: Numbers,Hyphens,quotation marks,new line characters,apostrophes,etc 
        are removed from the txt.
    
   Note: I wanted to experiment with different tokenizers, lemmatizer,etc. The 3 different tokenizers 
   were considered, but only 1 lemmatizer was used throughout this project as the result from the
   tokenizers were considerably similar, varying by 4-5% , So given the time constraint I 
   only tried to do implement the Naive Bayers Classifier in CBOW for 3 tokenizers. And for 
   remining only Treebank Tokenizer was used along with WordNetlemmatizer.

### B) Splitting Dataset/Corpus: ###
* The dataset/corpus is split into 66.6% of Training data and 33.3% of test data. Moreover, the data is shuffled everytime the code is re-run such that different instances 
       are shuffled between test and train datasets.

### C) Feature Engineering: ###

* In order to run machine learning algorithms we need to convert the text files into numerical feature vectors. Bag of Words (CountVectorizer()) is used
    in 1st for Naive Bayers and Random Forest Algorithms. A term-Document Matrix was constructed from the CountVectorizer() containing words or terms along columns and sentences along rows.
    The bag of words only considers the count of words which is not a good practice for doing sentiment analysis. Because some common words appear in many sentences which contain less importantnce. 
    Therefore,TF-IDF is used later in this assignment for Naive Bayers and Random Forest Algorithms which takes into account the word based upon its uniqueness.

### D) Model Construction: ###

* MultiNomial Naive Bayers was constructed from sklearn.model_selection library and Random Forest Classifier was constructed from sklearn.ensemble.

### E) Evaluation: ###

* The accuracy_score, F1-scores are printed into the console. As well as confusion matrix is also calculated.

### Who do I talk to? ###
### Details ###
* Created on Wed Dec 28 19:06:18 2020
* @author: shahzeb
* IDE: Spyder(Python3.8)
* sshah184@uottawa.ca


### Last but not the least if you liked my work looking forward to contribute to NLP you can on [Knowledge Graph Project](https://github.com/ShahzebFarruk/Knowledge-graph). Thank you!
