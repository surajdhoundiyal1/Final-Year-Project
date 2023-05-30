#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


fake = pd.read_csv('train.csv')


# In[3]:


fake.head()


# Checking Null Values and filling

# In[4]:


fake.isnull().sum()


# In[5]:


fake = fake.fillna(' ')


# In[6]:


fake.isnull().sum()


# Checking data duplicacy

# In[7]:


fake.duplicated()
#Here no duplicacy of data found.


# Checking data balancing

# In[8]:


import seaborn as sns
target_col= ["label"]
xx = fake[target_col[0]].value_counts().reset_index()
sns.barplot(x = "index", y = "label", data=xx, palette = "cividis");
#Data is balanced


# In[9]:


## Get the Independent Features

X=fake.drop('label',axis=1)


# In[10]:


X.head()


# In[11]:


## Get the Dependent features
y=fake['label']


# In[12]:


y.head()


# In[13]:


fake.shape


# In[14]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer


# In[15]:


fake.head(20800)


# In[16]:


messages=fake.copy()


# In[17]:


messages.reset_index(inplace=True)


# In[18]:


messages.head(10)


# In[19]:


messages['text'][6]


# In[20]:


import nltk
nltk.download('stopwords')


# In[21]:


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re


# In[22]:


ps = PorterStemmer()
corpus = []

    


# In[23]:


for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['text'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


# In[24]:


corpus[3]


# # Tfidf Vectorizer

# In[25]:


## TFidf Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorize=TfidfVectorizer(max_features=5000,ngram_range=(1,3))
X=vectorize.fit_transform(corpus).toarray()


# In[26]:


X.shape


# In[27]:


y=messages['label']


# # Splitting dataset into testing and training

# In[28]:


## Divide the dataset into Train and Test
from sklearn.model_selection import train_test_split


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# In[35]:


vectorize.get_feature_names_out()[:20]


# In[36]:


vectorize.get_params()


# In[37]:


count_df = pd.DataFrame(X_train, columns=vectorize.get_feature_names_out())


# In[38]:


count_df.head()


# # To Create Matrix

# In[49]:


import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[53]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# In[59]:


from sklearn import metrics
import numpy as np
import itertools


# # Logistic Regression

# In[60]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train , y_train)


# In[61]:


y_predict = lr.predict(X_test)


# In[62]:


accuracy_score(y_test , y_predict)


# In[63]:


print(classification_report(y_test, y_predict))


# In[64]:


cm = metrics.confusion_matrix(y_test, y_predict)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])


# # Naive Bayes

# In[65]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train , y_train)


# In[66]:


y_predict = nb.predict(X_test)
accuracy_score(y_test , y_predict)


# In[67]:


print(classification_report(y_test, y_predict))


# In[68]:


cm = metrics.confusion_matrix(y_test, y_predict)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])


# # Decision Tree

# In[69]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train , y_train)


# In[70]:


y_predict = dt.predict(X_test)
accuracy_score(y_test , y_predict)


# In[71]:


print(classification_report(y_test, y_predict))


# In[72]:


cm = metrics.confusion_matrix(y_test, y_predict)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])


# # Random Forest

# In[74]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train , y_train)


# In[76]:


y_predict = rf.predict(X_test)
accuracy_score(y_test , y_predict)


# In[77]:


print(classification_report(y_test, y_predict))


# In[78]:


cm = metrics.confusion_matrix(y_test, y_predict)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])


# In[ ]:




