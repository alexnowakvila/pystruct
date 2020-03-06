"""
===============================
OCR Letter sequence recognition
===============================
This example illustrates the use of a chain CRF for optical character
recognition. The example is taken from Taskar et al "Max-margin markov random
fields".

Each example consists of a handwritten word, that was presegmented into
characters.  Each character is represented as a 16x8 binary image. The task is
to classify the image into one of the 26 characters a-z. The first letter of
every word was ommited as it was capitalized and the task does only consider
small caps letters.

We compare classification using a standard linear SVM that classifies
each letter individually with a chain CRF that can exploit correlations
between neighboring letters (the correlation is particularly strong
as the same words are used during training and testsing).

The first figures shows the segmented letters of four words from the test set.
In set are the ground truth (green), the prediction using SVM (blue) and the
prediction using a chain CRF (red).

The second figure shows the pairwise potentials learned by the chain CRF.
The strongest patterns are "y after l" and "n after i".

There are obvious extensions that both methods could benefit from, such as
window features or non-linear kernels. This example is more meant to give a
demonstration of the CRF than to show its superiority.
"""
import numpy as np
import pdb
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC

from pystruct.datasets import load_letters
from pystruct.models import ChainCRF, GeneralizedChainCRF
# from pystruct.learners import FrankWolfeSSVM

from pystruct.learners import (NSlackSSVM, OneSlackSSVM,
                               SubgradientSSVM, FrankWolfeSSVM, GeneralizedFrankWolfeSSVM)

abc = "abcdefghijklmnopqrstuvwxyz"
np.random.seed(0)
letters = load_letters()
X, y, folds = letters['data'], letters['labels'], letters['folds']
# we convert the lists to object arrays, as that makes slicing much more
# convenient
X, y = np.array(X), np.array(y)
X_train, X_test = X[folds == 1], X[folds != 1]
y_train, y_test = y[folds == 1], y[folds != 1]
X_train = X_train
y_train = y_train
C = 5.

# Train linear SVM
svm = LinearSVC(dual=False, C=C)
# flatten input
svm.fit(np.vstack(X_train), np.hstack(y_train))

score_train = svm.score(np.vstack(X_train), np.hstack(y_train))
score_test = svm.score(np.vstack(X_test), np.hstack(y_test))
print("Test score with linear SVM: %f %f" % (score_train, score_test))


n_states = len(np.unique(np.hstack([yy.ravel() for yy in y_train])))
# define loss
class_weight = np.ones(n_states)
Loss = np.dot(np.expand_dims(class_weight, 1), np.ones((1, n_states)))
np.fill_diagonal(Loss, 0.0)

mod = 'generalized'
# mod = 'vanilla'

# sort by length
# lengths = np.array([y_train[i].shape[0] for i in range(y_train.shape[0])])


if mod == 'generalized':
    # Train linear chain CRF
    model = GeneralizedChainCRF(Loss=Loss)
    # pdb.set_trace()
    gssvm = GeneralizedFrankWolfeSSVM(model=model, C=C, check_dual_every=1, max_iter=100, verbose=True, X_test=X_test, Y_test=y_test)
    gssvm.fit(X_train, y_train)
    train_score = gssvm.score(X_train, y_train)
    test_score = gssvm.score(X_test, y_test)
    print("Train / Test score with gchain CRF: %f %f" % (train_score, test_score))
else:
    # Train linear chain CRF
    model = ChainCRF()
    # pdb.set_trace()
    ssvm = FrankWolfeSSVM(model=model, C=C, check_dual_every=10, max_iter=100, verbose=True)
    ssvm.fit(X_train, y_train)
    train_score = ssvm.score(X_train, y_train)
    test_score = ssvm.score(X_test, y_test)
    print("Train / Test score with chain CRF: %f %f" % (train_score, test_score))

# plot some word sequenced
n_words = 4
rnd = np.random.RandomState(1)
selected = rnd.randint(len(y_test), size=n_words)
max_word_len = max([len(y_) for y_ in y_test[selected]])
fig, axes = plt.subplots(n_words, max_word_len, figsize=(10, 10))
fig.subplots_adjust(wspace=0)
for ind, axes_row in zip(selected, axes):
    y_pred_svm = svm.predict(X_test[ind])
    y_pred_chain = ssvm.predict([X_test[ind]])[0]
    for i, (a, image, y_true, y_svm, y_chain) in enumerate(
            zip(axes_row, X_test[ind], y_test[ind], y_pred_svm, y_pred_chain)):
        a.matshow(image.reshape(16, 8), cmap=plt.cm.Greys)
        a.text(0, 3, abc[y_true], color="#00AA00", size=25)
        a.text(0, 14, abc[y_svm], color="#5555FF", size=25)
        a.text(5, 14, abc[y_chain], color="#FF5555", size=25)
        a.set_xticks(())
        a.set_yticks(())
    for ii in range(i + 1, max_word_len):
        axes_row[ii].set_visible(False)

plt.matshow(ssvm.w[26 * 8 * 16:].reshape(26, 26))
plt.colorbar()
plt.title("Transition parameters of the chain CRF.")
plt.xticks(np.arange(25), abc)
plt.yticks(np.arange(25), abc)
plt.show()
