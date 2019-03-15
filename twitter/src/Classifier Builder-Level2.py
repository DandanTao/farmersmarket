
# coding: utf-8

# <h3> Importing dataframe for 2nd level classification i.e. <br><br> For Different Labels </h3>

# In[20]:


import pandas as pd
df = pd.read_csv('../csv_data/new_tweet_classifier2.csv')
df.head()


# <h3> Mapping word labels to integers as "category_id". </h3>

# In[21]:


from io import StringIO
col = ['Label', 'Tweet']
df = df[col]
df = df[pd.notnull(df['Tweet'])]
df.columns = ['Label', 'Tweet']
df['category_id'] = df['Label'].factorize()[0]
category_id_df = df[['Label', 'category_id']].drop_duplicates().sort_values('category_id')
print(category_id_df)
#category_id_df = category_id_df.astype(str)
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Label']].values)
df.head()


# <h3> Looking at Data distribution </h3>

# In[22]:


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,6))
df.groupby('Label').Tweet.count().plot.bar(ylim=0)
plt.show()


# <h3> Extracting features from each tweet. </h3> Features will be used to differentiate 1 tweet from another for different classifiers.

# In[23]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=2, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.Tweet).toarray()
labels = df.category_id
features.shape


# <h3> Looking at data distribution for correlated words. </h3>

# In[24]:


from sklearn.feature_selection import chi2
import numpy as np
N = 2
#print(category_to_id.items())
for Label, category_id in category_to_id.items():
    features_chi2 = chi2(features, labels == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("# '{}':".format(Label))
    print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
    print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))


# Note: If multiple data distributions have similar correlated words, it means that we have too much data for 1 word or vice versa.

# <h4> Rough Work. Can Ignore </h4>

# In[36]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
#X_train, X_test, y_train, y_test = train_test_split(df['Tweet'], df['Label'], random_state = 0)
count_vect = CountVectorizer()
#X_train_counts = count_vect.fit_transform(X_train)
#tfidf_transformer = TfidfTransformer()
#X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#clf = MultinomialNB().fit(X_train_tfidf, y_train)


# In[37]:


#print(clf.predict(count_vect.transform(["News from Discover Arlington: Mayoral"])))


# <h3> Training a Model on LinearSVC and viewing the output on a plot</h3>

# In[38]:


import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
model = LinearSVC() #LogisticRegression(random_state=0)
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.Label.values, yticklabels=category_id_df.Label.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


# In[39]:


print(model.predict(count_vect.transform(["News from Discover Arlington: Mayoral"])))


# <h3> Detailed view of what went wrong </h3>

# In[28]:


from IPython.display import display
for predicted in category_id_df.category_id:
  for actual in category_id_df.category_id:
    if predicted != actual and conf_mat[actual, predicted] >= 10:
      print("'{}' predicted as '{}' : {} examples.".format(id_to_category[actual], id_to_category[predicted], conf_mat[actual, predicted]))
      display(df.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['Label', 'Tweet']])
      print('')


# In[29]:


#model.fit(features, labels)
#N = 2
#for Product, category_id in sorted(category_to_id.items()):
#  indices = np.argsort(model.coef_[category_id])
#  feature_names = np.array(tfidf.get_feature_names())[indices]
#  unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
#  bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
#  print("# '{}':".format(Product))
#  print("  . Top unigrams:\n       . {}".format('\n       . '.join(unigrams)))
#  print("  . Top bigrams:\n       . {}".format('\n       . '.join(bigrams)))


# <b>Recall (Sensitivity)</b> - Recall is the ratio of correctly predicted positive observations to the all observations in actual class.<br> 
# <b>F1 score</b> - F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account.<br>
# The f1-score gives you the harmonic mean of precision and recall. The scores corresponding to every class will tell you the accuracy of the classifier in classifying the data points in that particular class compared to all other classes. <br>
# <b>Support</b> is the number of samples of the true response that lie in that class.
# 

# In[30]:


from sklearn import metrics
print(metrics.classification_report(y_test, y_pred, target_names=df['Label'].unique()))


# <h3> Testing Other Models such as Logistic Regression, Random Forest Classifier and Naive Bayes. </h3>

# In[31]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
import seaborn as sns
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()

