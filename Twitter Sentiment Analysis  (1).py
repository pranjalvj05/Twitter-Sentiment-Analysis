
# coding: utf-8

# ## Twitter Sentimental Analysis using NLP

# ### Implementation
# 

# In[45]:


import pandas as pd
import numpy as np
import pickle
import os
import nltk


# ### Authorization of an application to access Twitter account data

# In[46]:


consumer_key   =    'xxxxxxxxxxxxxxxxxxxxxxxxx'
consumer_secret =   'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
    
access_token  =     'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
   
access_secret =     'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'   


# In[47]:


get_ipython().system('pip install credentials')
from credentials import *   


# In[48]:


def twitter_setup():
  
   authorizin = tweepy.OAuthHandler(consumer_key, consumer_secret)
   authorizin.set_access_token(access_token, access_secret)

   Api = tweepy.API(authorizin)
   return Api


# ### Getting the twitter data
# 
# 
#  

# In[49]:


get_ipython().system('pip install tweepy')


# In[50]:


import tweepy

extractor = twitter_setup()

no_of_tweets = extractor.user_timeline(screen_name="MSDhoni", count=300)
print("Number of tweets extracted: {}.\n".format(len(no_of_tweets)))

print("10  tweets recently:\n")
for tweet in no_of_tweets[:5]:
    print(tweet.text)
    print()

data = pd.DataFrame(data=[tweet.text for tweet in no_of_tweets], columns=['Number_of_Tweets'])


display(data.head(10))


# ### Natural Language Processing 

# In[51]:


data.Number_of_Tweets.value_counts()


# In[52]:


#Finding average number of tweets
np.mean([len(s.split(" ")) for s in data.Number_of_Tweets])


# In[53]:


#Natural Language processing for feature extraction
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer        
from nltk.stem.porter import PorterStemmer


# In[54]:


stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


# In[60]:


def tokenize(Number_of_Tweets):
   
    text = re.sub("[^a-zA-Z]", " ", Number_of_Tweets)
 
    tokens = nltk.word_tokenize(Number_of_Tweets)
  
    stems = stem_tokens(tokens, stemmer)
    return stems


# In[61]:


vectorization = CountVectorizer(
    tokenizer = tokenize,
    analyzer = 'word',
    stop_words = 'english',
    max_features = 68,
    lowercase = True
    
)


# In[62]:


corpus_features_tweets = vectorization.fit_transform(
    data.Number_of_Tweets.tolist() + data.Number_of_Tweets.tolist())


# In[63]:


corpus_features_tweets_nd = corpus_features_tweets.toarray()
corpus_features_tweets_nd.shape


# In[ ]:


vocabolary_tweets = vectorization.get_feature_names()
print (vocabolary_tweets)


# Sum up the counts of each vocabulary word
#  For each, print the vocabulary word and the number of times it 
#  appears in the data set

# In[ ]:


sumup = np.sum(corpus_features_tweets_nd, axis=0)
    
for tag, count in zip(vocabolary_tweets, sumup):
    print (count, tag)
    


# ### Starting of bag of words
# 
# Now we partition our data into train and test to apply the Logistic Regression model. 

# In[ ]:


from sklearn.cross_validation import train_test_split

X_Train, X_Test, y_Train, y_Test  = train_test_split(
        corpus_features_tweets_nd[0:len(data)], 
        data.Number_of_Tweets,
        train_size=0.75, 
        random_state=1435)


# In[ ]:


from sklearn.linear_model import LogisticRegression
    
Classification_model = LogisticRegression()
Classification_model = Classification_model.fit(X=X_Train, y=y_Train)


# ### Prediction on test

# In[ ]:


y_prediction = Classification_model.predict(X_Test)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_Test, y_prediction))

