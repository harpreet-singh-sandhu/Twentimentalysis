
# coding: utf-8

# # <u> Sentiment Analysis On Twitter Posts <u>

# In[42]:

import pandas as pd


# In[43]:

data_df = pd.read_csv("sentiment_short.csv", quotechar='"', encoding= "ISO-8859-1")
data_df.shape


# In[44]:

data_df.head()


# In[45]:

data_df.Sentiment.value_counts()


# In[46]:

import numpy as np
print("Average # words per post: ",np.mean([len(s.split(" ")) for s in data_df.TweetText]))


# In[47]:

test_set_length = int(0.3*(len(data_df)))
training_set_length = int((len(data_df)) - test_set_length)
print(test_set_length)
print(training_set_length)
print(training_set_length +test_set_length)


# In[48]:

training_set = data_df[0:training_set_length]
test_set = data_df[training_set_length:]
print("training_set shape: ",training_set.shape)
print("test_set shape: ",test_set.shape)


# In[49]:

del data_df
import gc
gc.collect()


# In[50]:

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")


# In[51]:

#def stem_tokens(tokens, stemmer):
#    stemmed = []
#    for item in tokens:
#        stemmed.append(stemmer.stem(item))
#    return stemmed


# In[52]:

import re as regex , nltk
def tokenize(text):
    # remove non letters
    text = regex.sub("[^a-zA-Z]", " ", text)
    # tokenize
    tokens = nltk.word_tokenize(text)
    # stem
#    stems = stem_tokens(tokens, stemmer)
    return tokens


# In[53]:

from sklearn.feature_extraction.text import CountVectorizer 
vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = tokenize, # tokenize is a user defined function declared above
    lowercase = True,
    stop_words = 'english',
   # max_features = 85
)


# In[54]:

features_matrix = vectorizer.fit_transform(training_set.TweetText.tolist() + test_set.TweetText.tolist())
#features_matrix = vectorizer.fit_transform(train_data_df.Text.tolist() + test_data_df.Text.tolist())


# In[55]:

features = features_matrix.toarray()
features.shape
del features_matrix


# In[56]:

vocab = vectorizer.get_feature_names()
print(vocab)
del vocab


# In[ ]:

dist = np.sum(features, axis=0)
for tag, count in zip(vocab, dist):
    print count, tag


# In[57]:

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test  = train_test_split(
        features, 
        training_set.Sentiment.tolist() + test_set.Sentiment.tolist(),#data_df.Sentiment
        test_size=0.30, 
        random_state=3)


# In[58]:

del training_set
del test_set


# In[59]:

from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression()
log_model = log_model.fit(X=X_train, y=y_train)


# In[60]:

y_pred = log_model.predict(X_test)


# In[61]:

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[62]:

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
NB_model = gnb.fit(X=X_train, y=y_train)
Nb_predictions = NB_model.predict(X_test)


# In[63]:

from sklearn.metrics import classification_report
print(classification_report(y_test, Nb_predictions))


# In[ ]:

from sklearn.svm import SVC
svm_clf = SVC()
svm_model = svm_clf.fit(X=X_train, y=y_train) 
svm_predictions = svm_model.predict(X_test)


# In[ ]:

from sklearn.metrics import classification_report
print(classification_report(y_test, svm_predictions))


# In[ ]:



