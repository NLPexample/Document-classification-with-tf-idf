# example based on code by Ahmed Kachkach
# https://cloud.google.com/blog/products/gcp/problem-solving-with-ml-automatic-document-classification

import os
import pandas as pd
from io import StringIO
import numpy as np

#%%
# raw data available here: http://mlg.ucd.ie/datasets/bbc.html
# click ">> Download raw text files" below "Dataset: BBC"

# this loop imports the raw data
df = pd.DataFrame()
folders = ["business", "entertainment", "politics", "sport", "tech"]
for x in folders:
    files = []
    path = r"data\bbc raw\bbc\{}".format(x)
    for file in os.listdir(path):
        with open(os.path.join(path, file)) as f:
            read_data = f.read()
            files.append(read_data)    
        
    temp = pd.DataFrame(files)
    temp["category"]=x
    
    df= df.append(temp)

df.columns = ['article', 'category']
df['category_id'] = df['category'].factorize()[0]

category_id_df = df[['category', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'category']].values)

# separating out the headline (the first line) from the article content (below the first line)
df['title'] = df['article'].str.split('\n\n').str.get(0)
df['content'] = df['article'].str.split('\n\n').str.get(1)

# count the document types
my_tab = pd.crosstab(index=df["category"], columns="count")
# we have a fairly even distribution of article type, so the analysis will not need balancing


df.sample(5, random_state=0)

#%%
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

features = tfidf.fit_transform(df.content).toarray()
labels = df.category_id
features.shape

#%%
# using the chi squared test to see the terms most correlated with the categories. The results look reasonable
from sklearn.feature_selection import chi2

N = 3
for category, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# '{}':".format(category))
  print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))

#%%

#==============================================================================
# from sklearn.manifold import TSNE
# 
# # Sampling a subset of our dataset because t-SNE is computationally expensive
# SAMPLE_SIZE = int(len(features) * 0.3)
# np.random.seed(0)
# indices = np.random.choice(range(len(features)), size=SAMPLE_SIZE, replace=False)
# projected_features = TSNE(n_components=2, random_state=0).fit_transform(features[indices])
# colors = ['pink', 'green', 'midnightblue', 'orange', 'darkgrey']
# for category, category_id in sorted(category_to_id.items()):
#     points = projected_features[(labels[indices] == category_id).values]
#     plt.scatter(points[:, 0], points[:, 1], s=30, c=colors[category_id], label=category)
# plt.title("tf-idf feature vector for each article, projected on 2 dimensions.",
#           fontdict=dict(fontsize=15))
# plt.legend()
#==============================================================================


#%%
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import cross_val_score


models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
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
#%%

#sns.boxplot(x='model_name', y='accuracy', data=cv_df)
#sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
#              size=8, jitter=True, edgecolor="gray", linewidth=2)

from sklearn.model_selection import train_test_split

model = LogisticRegression(random_state=0)

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)


#which terms contribute the most to a document being classified in each of the categories?
model.fit(features, labels)

from sklearn.feature_selection import chi2

N = 5
for category, category_id in sorted(category_to_id.items()):
  indices = np.argsort(model.coef_[category_id])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
  bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
  print("# '{}':".format(category))
  print("  . Top unigrams:\n       . {}".format('\n       . '.join(unigrams)))
  print("  . Top bigrams:\n       . {}".format('\n       . '.join(bigrams)))
#%%
texts = ["Hooli stock price soared after a dip in PiedPiper revenue growth.",
         "Captain Tsubasa scores a magnificent goal for the Japanese team.",
         "Merryweather mercenaries are sent on another mission, as government oversight groups call for new sanctions.",
         "Beyoncé releases a new album, tops the charts in all of south-east Asia!",
         "You won't guess what the latest trend in data analysis is!"]
text_features = tfidf.transform(texts)
predictions = model.predict(text_features)
for text, predicted in zip(texts, predictions):
  print('"{}"'.format(text))
  print("  - Predicted as: '{}'".format(id_to_category[predicted]))
  print("")
  #%%
