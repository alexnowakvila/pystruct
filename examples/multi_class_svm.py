"""
==============================
Crammer-Singer Multi-Class SVM
==============================

Comparing different solvers on a standard multi-class SVM problem.
"""

from time import time
import numpy as np
import pdb

from sklearn import datasets
from sklearn.datasets import fetch_mldata
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from pystruct.models import MultiClassClf, GeneralizedMultiClassClf
from pystruct.learners import (NSlackSSVM, OneSlackSSVM,
                               SubgradientSSVM, FrankWolfeSSVM, GeneralizedFrankWolfeSSVM)

np.random.seed(0)

###############################################################################
# Load Dataset
###############################################################################

# dat = 'digits'
# dat = 'iris'
dat = 'onedat'
if dat == 'digits':
    # do a binary digit classification
    # digits = fetch_mldata("USPS")
    digits = load_digits()
    X, y = digits.data, digits.target
    #X = X / 255.
    X = X / 16.
    #y = y.astype(np.int) - 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    n_classes = 10
elif dat == 'iris':
    # loading the iris dataset 
    iris = datasets.load_iris() 
    # X -> features, y -> label 
    X, y = iris.data, iris.target
    # dividing X, y into train and test data 
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0) 
    n_classes = 3
elif 'onedat':
    X = np.ones((100, 1))
    n_classes = 100
    prob = np.ones(n_classes) / n_classes
    # add some bias to the first class
    eps = 0.1
    noise = -eps / (n_classes - 1) * np.ones(n_classes)
    noise[2] = eps
    prob += noise
    y = np.random.choice(n_classes, 1000, p=list(prob))
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0) 
    print(prob)
    # pdb.set_trace()
else:
    raise ValueError("Dataset %s not found" % dat)


# epsilon = 0.3
# # add noise 
# for i in range(y_train.shape[0]):
#     if np.random.uniform() < epsilon:
#         y_train[i] = np.random.randint(n_classes)

# for i in range(y_test.shape[0]):
#     if np.random.uniform() < epsilon:
#         y_test[i] = np.random.randint(n_classes)


# we add a constant 1 feature for the bias
X_train_bias = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_test_bias = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

model = MultiClassClf(n_features=X_train_bias.shape[1], n_classes=n_classes)
# n_slack_svm = NSlackSSVM(model, verbose=2, check_constraints=False, C=0.1,
#                          batch_size=100, tol=1e-2)
# one_slack_svm = OneSlackSSVM(model, verbose=2, C=.10, tol=.001)
# subgradient_svm = SubgradientSSVM(model, C=0.1, learning_rate=1.,
#                                   max_iter=500, verbose=1, decay_exponent=1, decay_t0=10)

# method = 'generalized'
method = 'vanilla'

Cs = [1000.]
# Cs = [6.5, 7., 7.5]

for C in Cs:

    fw_bc_svm = FrankWolfeSSVM(model, C=C, max_iter=500, check_dual_every=50, line_search=False, verbose=True)
    # fw_batch_svm = FrankWolfeSSVM(model, C=.1, max_iter=50, batch_mode=True)

    gmodel = GeneralizedMultiClassClf(n_features=X_train_bias.shape[1], n_classes=n_classes)
    gfw_bc_svm = GeneralizedFrankWolfeSSVM(gmodel, C=C, max_iter=500, check_dual_every=50, line_search=False, verbose=True)


    if method == 'generalized':
        ############################################################################
        #  Generalized Method
        ############################################################################

        start = time()
        gfw_bc_svm.fit(X_train_bias, y_train)
        y_pred = np.hstack(gfw_bc_svm.predict(X_test_bias))
        time_fw_bc_svm = time() - start
        print("Score with maxminsvm: %f , C=%f (took %f seconds)" %
            (np.mean(y_pred == y_test), C, time_fw_bc_svm))
        print(gfw_bc_svm.w)
    elif method == 'vanilla':
        ############################################################################
        #  Vanilla Method
        ############################################################################

        start = time()
        fw_bc_svm.fit(X_train_bias, y_train)
        y_pred = np.hstack(fw_bc_svm.predict(X_test_bias))
        time_fw_bc_svm = time() - start
        print("Score with cssvm: %f , C=%f (took %f seconds)" %
            (np.mean(y_pred == y_test), C, time_fw_bc_svm))
        print(fw_bc_svm.w)
        import pdb; pdb.set_trace()
    # start = time()
    # fw_batch_svm.fit(X_train_bias, y_train)
    # y_pred = np.hstack(fw_batch_svm.predict(X_test_bias))
    # time_fw_batch_svm = time() - start
    # print("Score with pystruct frankwolfe batch ssvm: %f (took %f seconds)" %
    #       (np.mean(y_pred == y_test), time_fw_batch_svm))
