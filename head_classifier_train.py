import os
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# https://github.com/bdrad/rad_classify
from rad_classify import EndToEndPreprocessor


seed = 42

GRID_SEARCH = False
DATA_DIR = './data_cv_brainmri_multiclass'

# GET DATA
TEST_SIZE = 0.15
data_raw = pd.read_csv(DATA_DIR+'/ftbrain_prot.csv', header=0, low_memory=False)['__label__protocol __label__order text'].values
train_raw, test_raw = train_test_split(data_raw, test_size = TEST_SIZE, shuffle=True, random_state=seed)

# PREPROCESS
preprocessor = EndToEndPreprocessor(replacement_path="../rad_classify/rad_classify/semantic_dictionaries/clever_replacements",
                                radlex_path="../rad_classify/rad_classify/semantic_dictionaries/radlex_replacements")

# Replace large spaces in text with period to prevent NEGEX of whole string
train_raw = [text.replace(' ' * 4, '.') for text in train_raw]
test_raw = [text.replace(' ' * 4, '.') for text in test_raw]

train = preprocessor.transform(train_raw)
test = preprocessor.transform(test_raw)

df_train = pd.DataFrame({'processed':list(train), 'raw': list(train_raw)})
df_train.to_csv(DATA_DIR+'/brain_mri_train_multiclass_preprocessed.csv',
                index=False, quoting=3,encoding="utf-8-sig", escapechar=" ",na_rep=' ')
df_test = pd.DataFrame({'processed':list(test), 'raw': list(test_raw)})
df_test.to_csv(DATA_DIR+'/brain_mri_test_multiclass_preprocessed.csv',
              index=False, quoting=3,encoding="utf-8-sig", escapechar=" ",na_rep=' ')



#### IMPORTANT STEP!!!
# data in fastText format: data = ["__label__ clinical indication"]
# seperate get labels and text - otherwise you'll be passing the y values in the X
#   new format: y = [label], x = [clinical indication]
y_train = [item.split(' ')[0] for item in train]
x_train = [' '.join(item.split(' ')[2:]) for item in train]

y_test = [item.split(' ')[0] for item in test]
x_test = [' '.join(item.split(' ')[2:]) for item in test]



if GRID_SEARCH:
    grid_params = {
                  'learning_rate': [0.1], # tried 0.05, 0.1, 0.5
                  'n_estimators': [100], # tried 50, 100, 300, 500
                  'objective': ['binary:logistic'], # tried 'multiclass:softmax'
                  'booster': ['gbtree'],
                  'max_depth': [8],
                  'colsample_bytree': [0.8], # tried 0.3, 0.5, 0.8, 1.0
                  'gamma': [1]
                  }

    # drop these labels cause they are too infrequent and mess up gridsearch
    drop_indices = np.concatenate([np.argwhere(np.asarray(y_train) == '__label__tmj'),
                                   np.argwhere(np.asarray(y_train) == '__label__orbits')])[:,0]
    y_train_cv = np.delete(y_train, drop_indices, axis=0)
    x_train_piped_cv = np.delete(x_train_piped.toarray(), drop_indices, axis=0)

    model = XGBClassifier(random_state=seed, n_jobs=-1, nthreads=10)
    model_gridsearch = GridSearchCV(model, grid_params,
                                        cv=StratifiedKFold(n_splits=5, random_state=seed),
                                        scoring='neg_log_loss', verbose=2)
    model_gridsearch.fit(x_train_piped_cv, y_train_cv)
    score = model_gridsearch.best_score_
    params = {**params, **model_gridsearch.best_params_}




model = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('xgb', XGBClassifier(**params))
                 ])

model.fit(x_train, y_train)

cats = model.classes_

# prediction on test set
probs = model.predict_proba(x_test)
max_indices = probs.argmax(axis=1)
preds = cats[max_indices]
probs = [probas[idx] for idx, probas in zip(max_indices, probs)]


### OUTPUT RESULTS
output = """
--------------------------------------------------
BoW Multiclass Test Classification Results (resample = {})
{}
--------------------------------------------------
Test Accuracy: {}
Confusion Matrix:
{}
{}

Classification Report:
{}

Params:
{}
""".format(RESAMPLE,
           datetime.datetime.now(),
           accuracy_score(y_test, preds),
           cats,
           confusion_matrix(y_test, preds),
           classification_report(y_test, preds),
           params)
print(output)
