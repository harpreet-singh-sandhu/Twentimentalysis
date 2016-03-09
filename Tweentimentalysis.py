
# coding: utf-8

# # <u> Sentiment Analysis On Twitter Posts <u>

# In[1]:

import pandas as pd


# In[2]:

data_df = pd.read_csv("sentiment.csv", quotechar='"', encoding= "ISO-8859-1")
data_df.shape


# In[3]:

data_df=data_df.drop('Topic',1)
data_df=data_df.drop('TweetId',1)
data_df=data_df.drop('TweetDate',1)
data_df.shape


# In[4]:

data_df.columns = ["Sentiment","TweetText"]
positive_sentiment = data_df[data_df['Sentiment']=='positive']
negative_sentiment = data_df[data_df['Sentiment']=='negative']
print(positive_sentiment.shape)
print(negative_sentiment.shape)


# In[5]:

val = 1000
frames = [positive_sentiment[:val], negative_sentiment[:val]]
data_df = pd.concat(frames)
data_df.shape


# In[6]:

data_df.head()


# In[7]:

data_df.Sentiment.value_counts()


# In[8]:

import numpy as np
print("Average # words per post: ",np.mean([len(s.split(" ")) for s in data_df.TweetText]))


# In[9]:

test_set_length = int(0.3*(len(data_df)))
training_set_length = int((len(data_df)) - test_set_length)
print(test_set_length)
print(training_set_length)
print(training_set_length +test_set_length)


# In[10]:

training_set = data_df[0:training_set_length]
test_set = data_df[training_set_length:]
print("training_set shape: ",training_set.shape)
print("test_set shape: ",test_set.shape)


# In[11]:

del data_df
#import gc
#gc.collect()


# In[12]:

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")


# In[13]:

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


# In[14]:

import re as regex , nltk
def tokenize(text):
    # remove non letters
    text = regex.sub("[^a-zA-Z]", " ", text)
    # tokenize
    tokens = nltk.word_tokenize(text)
    # stem
    stems = stem_tokens(tokens, stemmer)
    return stems


# In[15]:

from sklearn.feature_extraction.text import CountVectorizer 
vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = tokenize, # tokenize is a user defined function declared above
    lowercase = True,
    stop_words = 'english',
    #max_features = 200
)


# In[16]:

features_matrix = vectorizer.fit_transform(training_set.TweetText.tolist() + test_set.TweetText.tolist())
#features_matrix = vectorizer.fit_transform(train_data_df.Text.tolist() + test_data_df.Text.tolist())


# In[17]:

features_matrix = features_matrix.toarray()
#features.shape
#del features_matrix


# In[18]:

vocab = vectorizer.get_feature_names()
#print(vocab)
#del vocab


# In[19]:

#dist = np.sum(features, axis=0)
#for tag, count in zip(vocab, dist):
#    print count, tag


# In[20]:

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test  = train_test_split(
        features_matrix, 
        training_set.Sentiment.tolist() + test_set.Sentiment.tolist(),#data_df.Sentiment
        test_size=0.30, 
        random_state=3)


# In[21]:

del training_set
del test_set


# In[22]:

from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression()
log_model = log_model.fit(X=X_train, y=y_train)
y_pred = log_model.predict(X_test)


# In[23]:

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[24]:

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
NB_model = gnb.fit(X=X_train, y=y_train)
NB_predictions = NB_model.predict(X_test)


# In[25]:

print(classification_report(y_test, NB_predictions))


# In[26]:

from sklearn.svm import SVC
svm_clf = SVC()
svm_model = svm_clf.fit(X=X_train, y=y_train) 
svm_predictions = svm_model.predict(X_test)


# In[27]:

print(classification_report(y_test, svm_predictions))


# In[28]:

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
clf_predictions = clf.predict(X_test)


# In[29]:

print(classification_report(y_test, clf_predictions))


# In[30]:

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=40)
clf = clf.fit(X_train, y_train)
clf_predictions = clf.predict(X_test)


# In[31]:

print(classification_report(y_test, clf_predictions))


# In[32]:

from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()
clf.fit(X_train, y_train)
clf_predictions = clf.predict(X_test)


# In[33]:

print(classification_report(y_test, clf_predictions))


# In[34]:

from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier()
clf.fit(X_train, y_train)
clf_predictions = clf.predict(X_test)


# In[35]:

print(classification_report(y_test, clf_predictions))

