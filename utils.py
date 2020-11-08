import pandas as pd
import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
#NLTK Libraries
import nltk
nltk.download('punkt',quiet=True)
nltk.download('stopwords',quiet=True)
nltk.download('wordnet',quiet=True)
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import SnowballStemmer

class Utils:

    def load_from_csv(self, path):
        return pd.read_csv(path)

    def features_target(self, dataset, drop_cols, y):
        X = dataset.drop(drop_cols, axis=1)
        y = dataset[y]
        return X, y

    def model_export(self, clf, score):
        print('name: ' + str(clf))
        score = np.abs(score)
        joblib.dump(clf, './models/' + str(score))

    def vectorized_fiting(self,path,wordToVectorized):
        dataFrame1 = self.load_from_csv('./in/ropa.csv')
        alldata = pd.concat([dataFrame1])
        alldata_data = alldata.values[:, 0]
        vectorizer = TfidfVectorizer()
        vectorizer.fit_transform(alldata_data)
        x_test = vectorizer.transform([wordToVectorized])
        return x_test

    def textData_cleaning(self,text):
        porter = PorterStemmer()
        lancaster=LancasterStemmer()
        wordnet = WordNetLemmatizer()
        spanish_stemmer = SnowballStemmer('spanish')

        text=TweetTokenizer().tokenize(text)
        text=self.stopwords_cleaner(text)
        #print(text)
        thematik=''
        for w in text:
            thematik=thematik+' '+w
        #print(thematik)
        return thematik
        
    
    def stopwords_cleaner(self,text):
        stoped = stopwords.words('spanish')
        content = [w for w in text if w.lower() not in stoped]
        return content