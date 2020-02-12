# -*- coding: utf-8 -*-
# @Time    : 2019-02-27 18:03
# @Author  : taotao.zhou@zhenai.com
# @File    : test.py.py
# @Software: PyCharm

import itertools
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(i, j, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('true label')
    plt.xlabel('predict label')
    plt.show()
    # plt.savefig('confusion_matrix',dpi=200)



cnf_matrix = np.loadtxt("data/confusion_matrix.txt",dtype=int,delimiter=",")
print(cnf_matrix)

class_names = ['yes', 'no', 'refuse', 'busy', 'other']

# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names,
#                       title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False,
                      title='confusion_matrix')

"""
import pandas as pd
review_file = "data/badcaseDataReview.csv"
review = pd.read_csv(review_file)

predictors = ['text', 'label']

review = review[predictors]

ret = review.sample(frac=1)

ret.to_csv("data/badcaseDataReview.csv", index=False, header=True,encoding="utf_8_sig")
"""