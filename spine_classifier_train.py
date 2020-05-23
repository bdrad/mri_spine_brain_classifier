import os
import datetime
import numpy as np
import pandas as pd
import fastText
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# https://github.com/bdrad/rad_classify
from rad_classify import EndToEndPreprocessor


DATA_DIR = './data'
OUTPUT_DIR = os.path.join(DATA_DIR, 'spine_output')
seed = 42


# GET DATA
train_raw = pd.read_csv(DATA_DIR+'/spine.train', header=0, quoting=3, low_memory=False).values
val_raw = pd.read_csv(DATA_DIR+'/spine.valid', header=0, quoting=3, low_memory=False).values
val_raw, test_raw = train_test_split(val_raw, test_size=0.5, random_state=seed)

# PREPROCESS
preprocessor = EndToEndPreprocessor(replacement_path="../rad_classify/rad_classify/semantic_dictionaries/clever_replacements",
                                radlex_path="../rad_classify/rad_classify/semantic_dictionaries/radlex_replacements")

train = preprocessor.transform(train_raw)
val = preprocessor.transform(val_raw)
test = preprocessor.transform(test_raw)

df_train = pd.DataFrame({'Labeled Data':list(train)})
df_train.to_csv(DATA_DIR+'/spine_mri_train_preprocess.csv',
                index=False, quoting=3,encoding="utf-8-sig", escapechar=" ",na_rep=' ')
df_val = pd.DataFrame({'Labeled Data':list(val)})
df_val.to_csv(DATA_DIR+'/spine_mri_val_preprocess.csv',
              index=False, quoting=3,encoding="utf-8-sig", escapechar=" ",na_rep=' ')
df_test = pd.DataFrame({'Labeled Data':list(test)})
df_test.to_csv(DATA_DIR+'/spine_mri_test_preprocess.csv',
              index=False, quoting=3,encoding="utf-8-sig", escapechar=" ",na_rep=' ')


# Train
prms = {'epoch': 6,
        'thread': 15,
        'dim': 100,
        'lr': 0.1,
        'wordNgrams': 2,
        'loss': 'softmax'}

print(datetime.date.today())
classifier = fastText.train_supervised(DATA_DIR+'/spine_mri_train_preprocess.csv',
                                       epoch=prms['epoch'],
                                       thread=prms['thread'],
                                       dim=prms['dim'],
                                       lr=prms['lr'],
                                       wordNgrams=prms['wordNgrams'],
                                       loss=prms['loss'])



# Test Set Results
result = classifier.test(DATA_DIR + '/spine_mri_test_preprocess.csv')
labels = results[0] # Label (True: Contrast, False: Routine)
probas = results[1] # confidence score

# get boolean result for label
# Convert confidence score from confidence of the label
#    to a one-way score: confidence of contrast (needed in this format for ROC AUC calculation) 
labelsbool = []
probas_true = []
for (label, proba) in zip(labels, probas):
    label = label[0]
    proba = proba[0]
    if label == '__label__true':
        labelsbool.append(1)
        probas_true.append(proba)
    elif label == '__label__false':
        labelsbool.append(0)
        probas_true.append(1-proba)
probas_true = np.expand_dims(probas_true, axis=1)
labelsbool = np.asarray(labelsbool)



# Convert GT Labels to Boolean
gt_labelsbool = []
for instance in val:
    if instance[9:13] == 'true':
        gt_labelsbool.append(1)
    elif instance[9:14] == 'false':
        gt_labelsbool.append(0)
    else:
        raise Exception('instance improper label: ' + instance)
gt_labelsbool = np.asarray(gt_labelsbool)


accuracy = metrics.accuracy_score(gt_labelsbool, labelsbool)
precision = metrics.precision_score(gt_labelsbool, labelsbool)
recall = metrics.recall_score(gt_labelsbool, labelsbool)
roc_auc = metrics.roc_auc_score(gt_labelsbool, probas_true)
f1_score = metrics.f1_score(gt_labelsbool, labelsbool)
cm = metrics.confusion_matrix(gt_labelsbool, labelsbool)

print('-'*30)
print('Test RESULTS')
print('_'*30)
print(prms)
print_results(*result)
print('ROC AUC: {}'.format(roc_auc))
print('Accuracy: {}'.format(accuracy))
print('Precision: {}'.format(precision))
print('Recall: {}'.format(recall))
print('F1 score: {}'.format(f1_score))
print('Confusion metric:')
print(cm)
