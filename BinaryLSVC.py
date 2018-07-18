# import pandas as pd
# import numpy as np
#
# from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import NearMiss
# from sklearn.pipeline import make_pipeline
# from imblearn.pipeline import make_pipeline as make_pipeline_imb
#
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
# from imblearn.metrics import classification_report_imbalanced
#
# data_set = pd.read_csv('processed_data.csv', nrows=1050)
# X = data_set.iloc[:, :-1].values
# y = data_set.iloc[:, -1].values
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#
#
# # Results
# def print_results(headline, true_value, pred_value):
#     print(headline)
#     method = 'weighted'
#     print("Accuracy:{}".format(accuracy_score(true_value, pred_value)))
#     print("Precision:{}".format(precision_score(true_value, pred_value, average=method)))
#     print("Recall: {}".format(recall_score(true_value, pred_value, average=method)))
#     print("F1 Meas:{}".format(f1_score(true_value, pred_value, average=method)))
#
# # Classifier default
# classifier = RandomForestClassifier
#
# # Building normal model
# pipeline = make_pipeline(classifier(random_state=0))
# model = pipeline.fit(X_train, y_train)
# prediction = model.predict(X_test)
#
# print_results('RFC', y_test, prediction)
# print(classification_report(y_test, prediction))
#
# # Building model with SMOTE
# smote_pipeline = make_pipeline_imb(SMOTE(random_state=0), classifier(random_state=0))
# smote_model = smote_pipeline.fit(X_train, y_train)
# smote_prediction = smote_model.predict(X_test)
#
# print_results('RFC + SMOTE', y_test, smote_prediction)
# print(classification_report(y_test, smote_prediction))
#
# # Building model with undersampling
# nearmiss_pipeline = make_pipeline_imb(NearMiss(random_state=0), classifier(random_state=0))
# nearmiss_model = nearmiss_pipeline.fit(X_train, y_train)
# nearmiss_prediction = nearmiss_model.predict(X_test)
#
# print_results('RFC + SMOTE', y_test, nearmiss_prediction)
# print(classification_report(y_test, nearmiss_prediction))
# Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Machine Learning Libs
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.svm import LinearSVC, SVC
from sklearn.exceptions import UndefinedMetricWarning

# Importing Count Vectorised Dataset
dataset = pd.read_csv('processed_data.csv', nrows=800)

# Count the Records of Categories
from collections import Counter
Counter(dataset.aspectCategory)

# Taking Individual CATs in consideration
Micellaneous = dataset.loc[dataset.aspectCategory == 'anecdotes/miscellaneous', :]
Food = dataset.loc[dataset.aspectCategory == 'food', :]
Service = dataset.loc[dataset.aspectCategory == 'service', :]
Ambiance = dataset.loc[dataset.aspectCategory == 'ambience', :]
Price = dataset.loc[dataset.aspectCategory == 'price', :]

# Making Bi-Sets
bi_set_1 = pd.concat([Micellaneous, Food], axis=0).values
bi_set_2 = pd.concat([Micellaneous, Service], axis=0).values
bi_set_3 = pd.concat([Micellaneous, Ambiance], axis=0).values
bi_set_4 = pd.concat([Micellaneous, Price], axis=0).values


def fxn():
    warnings.warn("deprecated", )


def machine(data_set, C=0.5):
    # Preparing Machine for Miscellaneous and Food based on No. of Records
    bi_set_1 = data_set
    X = bi_set_1[:, :-1]
    y = bi_set_1[:, -1]

    warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

    from sklearn.preprocessing import LabelEncoder
    le_y = LabelEncoder()
    y_le = le_y.fit_transform(y)

    keys = []
    for i in range(len(le_y.classes_)):
        keys.append(le_y.classes_[i])
        print(i, '-->', le_y.classes_[i])
    one_key = keys[0][:5] + '+' + keys[1][:5]

    X_train, X_test, y_train, y_test = train_test_split(X, y_le,
                                                        test_size=0.2, random_state=0)

    # ----------------------------------- Lin SVC

    classifier = LinearSVC(C=C, random_state=0)
    classifier.fit(X_train, y_train)
    y_hat = classifier.predict(X_test)

    print(classification_report(y_test, y_hat))
    print("F1 Score:{:.2f}%".format(f1_score(y_test, y_hat, average='weighted') * 100))
    return one_key, f1_score(y_test, y_hat, average='weighted')


f1s = dict()
key, val = machine(data_set = bi_set_1)
f1s[key] = val
key, val = machine(bi_set_2)
f1s[key] = val
key, val = machine(bi_set_3)
f1s[key] = val
key, val = machine(bi_set_4)
f1s[key] = val

print('CATEGORIES\t\t F-Measure')
for k, v in f1s.items():
    print(k, ' \t {:0.2f}%'.format(v*100))
