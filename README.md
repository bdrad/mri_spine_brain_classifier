# NLP Classifier of Spine and Head MRI Protocols

Repository for the manuscript: "Development of a web-based automated neuroradiology protocoling automation tool with natural language processing."

## Abstract
### Background
A systematic approach to magnetic resonance imaging (MRI) protocol assignment is essential for the efficient delivery of safe patient care. However, assigning protocols, especially in neuroimaging, can be a time-consuming and error-prone task for radiology trainees and technicians. Advances in natural language processing (NLP) allow for the development of accurate automated protocol assignment.

### Purpose
We developed and evaluated a NLP model that automates protocol assignment, given the clinician indiciation text.

### Methods
We collected 7139 spine MRI protocols and 990 head MRI protocols from a single academic research institution. Protocols were split into training, validation, and test sets. Training and validation sets were used to develop 2 NLP models to classify spine MRI and brain MRI protocols using fastText and XGBoost, respectively.

### Results
The spine MRI model had an accuracy of 83.38% and a receiver operator characteristic area under the curve (ROC AUC) of 0.8873. The head MRI model had an accuracy of 85.43% with a routine protocol ROC AUC of 0.9463 and contrast protocol ROC AUC of 0.9284. Cancer, infectious, and inflammatory related keywords were highly associated with contrast administration. Words related to structural issues for spine MRI and stroke or altered mental status for head MRI were indicative of routine brain MRIs without contrast. Error analysis revealed that modifications to preprocessing and increasing the sample size may improve performance and reduce systematic biases.

### Conclusion
We developed two NLP models that accurately predict spine and head MRI protocol assignment, which could improve radiology workflow efficiency.


# Webapp
A live demonstration is provided at https://barebonesmri.herokuapp.com/.

# How to use this Repo
## Installation Instructions

1. Clone this repository

  `git clone https://github.com/bdrad/mri_spine_brain_classifier.git`


2. Install dependencies (install via pip or conda)

 * numpy
 * pandas
 * sklearn
 * matplotlib
 * xgboost


3. Install Rad Classify (utilized for preprocessing steps & fastText wrapper): https://github.com/bdrad/rad_classify

## Training scripts
`head_classifier_train.py` and `spine_classifier_train.py` contain the training and test set evaluation scripts for their respective classifier.
