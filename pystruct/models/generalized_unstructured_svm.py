import numpy as np
import pdb
import scipy.special as sp

from .base import StructuredModel
from .utils import crammer_singer_joint_feature


class GeneralizedMultiClassClf(StructuredModel):
    """Formulate linear multiclass SVM in C-S style in CRF framework.

    Inputs x are simply feature arrays, labels y are 0 to n_classes.

    Notes
    ------
    No bias / intercept is learned. It is recommended to add a constant one
    feature to the data.

    It is also highly recommended to use n_jobs=1 in the learner when using
    this model. Trying to parallelize the trivial inference will slow
    the infernce down a lot!

    Parameters
    ----------
    n_features : int
        Number of features of inputs x.
        If None, it is inferred from data.

    n_classes : int, default=None
        Number of classes in dataset.
        If None, it is inferred from data.

    class_weight : None, or array-like
        Class weights. If an array-like is passed, it must have length
        n_classes. None means equal class weights.

    rescale_C : bool, default=False
        Whether the class-weights should be used to rescale C (liblinear-style)
        or just rescale the loss.
    """
    def __init__(self, n_features=None, n_classes=None, class_weight=None,
                 rescale_C=False, Loss=None):
        # one weight-vector per class
        self.n_states = n_classes
        self.n_features = n_features
        self.rescale_C = rescale_C
        self.class_weight = class_weight
        self.inference_calls = 0
        self._set_size_joint_feature()
        self._set_class_weight()
        if Loss is None:
            self.Loss = np.dot(np.expand_dims(self.class_weight, 1), np.ones((1, self.n_states)))
            np.fill_diagonal(self.Loss, 0.0)
        else:
            self.Loss = Loss

    def _set_size_joint_feature(self):
        if None not in [self.n_states, self.n_features]:
            self.size_joint_feature = self.n_states * self.n_features

    def initialize(self, X, Y):
        n_features = X.shape[1]
        if self.n_features is None:
            self.n_features = n_features
        elif self.n_features != n_features:
            raise ValueError("Expected %d features, got %d"
                             % (self.n_features, n_features))

        n_classes = len(np.unique(np.hstack([y.ravel() for y in Y])))
        if self.n_states is None:
            self.n_states = n_classes
        elif self.n_states != n_classes:
            raise ValueError("Expected %d classes, got %d"
                             % (self.n_states, n_classes))
        self._set_size_joint_feature()
        self._set_class_weight()

    def output_embedding(self, X, Y):
        n_samples = Y.shape[0]
        mu = np.zeros((Y.shape[0], self.n_states))
        for i in range(Y.shape[0]):
            mu[i, Y[i]] = 1
        return mu

    def __repr__(self):
        return ("%s(n_features=%d, n_classes=%d)"
                % (type(self).__name__, self.n_features, self.n_states))

    def joint_feature(self, x, y, y_true=None):
        """Compute joint feature vector of x and y.

        Feature representation joint_feature, such that the energy of the configuration
        (x, y) and a weight vector w is given by np.dot(w, joint_feature(x, y)).

        Parameters
        ----------
        x : nd-array, shape=(n_features,)
            Input sample features.

        y : int
            Class label. Between 0 and n_classes.

        y_true : int
            True class label. Needed if rescale_C==True.


        Returns
        -------
        p : ndarray, shape (size_joint_feature,)
            Feature vector associated with state (x, y).
        """
        # put feature vector in the place of the weights corresponding to y
        result = np.zeros((self.n_states, self.n_features))
        result[y, :] = x
        if self.rescale_C:
            if y_true is None:
                raise ValueError("rescale_C is true, but no y_true was passed"
                                 " to joint_feature.")
            result *= self.class_weight[y_true]

        return result.ravel()

    def batch_joint_feature(self, X, Y, Y_true=None):
        result = np.zeros((self.n_states, self.n_features))
        if self.rescale_C:
            if Y_true is None:
                raise ValueError("rescale_C is true, but no y_true was passed"
                                 " to joint_feature.")
            for l in range(self.n_states):
                mask = Y == l
                class_weight = self.class_weight[Y_true[mask]][:, np.newaxis]
                result[l, :] = np.sum(X[mask, :] * class_weight, axis=0)
        else:
            # if we don't have class weights, we can use our efficient
            # implementation
            assert(X.shape[0] == Y.shape[0])
            assert(X.shape[1] == self.n_features)
            crammer_singer_joint_feature(X, Y, result)
        return result.ravel()

    def mean_joint_feature(self, x, q):
        """Compute mean joint feature vector of x and y.

        Feature representation joint_feature, such that the energy of the configuration
        (x, y) and a weight vector w is given by np.dot(w, joint_feature(x, y)).

        Parameters
        ----------
        x : nd-array, shape=(n_features,)
            Input sample features.

        q : ndarray, shape=(n_classes)
            Probability vector


        Returns
        -------
        p : ndarray, shape (size_joint_feature,)
            Mean feature vector.
        """
        # put feature vector in the place of the weights corresponding to y
        result = np.ones((self.n_states, self.n_features))
        result = result * x.reshape(1, -1)
        result = result * q.reshape(-1, 1)
        if self.rescale_C:
            raise NotImplementedError
        return result.ravel()

    def batch_mean_joint_feature(self, X, Q, Y_true=None):
        """
        Returns
        -------
        result: ndarray, shape=(n_states * n_features)
            Sum of mean joint features for all data points in the batch
        """
        result = np.zeros(self.size_joint_feature)
        if self.rescale_C:
            raise NotImplementedError
        else:
            # if we don't have class weights, we can use our efficient
            # implementation
            assert(X.shape[0] == Q.shape[0])
            assert(X.shape[1] == self.n_features)
            for i in range(X.shape[0]):
                result += self.mean_joint_feature(X[i], Q[i])
        return result

    def inference(self, x, w, relaxed=None, return_energy=False):
        """Inference for x using parameters w.

        Finds armin_y np.dot(w, joint_feature(x, y)), i.e. best possible prediction.

        For an unstructured multi-class model (this model), this
        can easily done by enumerating all possible y.

        Parameters
        ----------
        x : ndarray, shape (n_features,)
            Input sample features.

        w : ndarray, shape=(size_joint_feature,)
            Parameters of the SVM.

        relaxed : ignored

        Returns
        -------
        y_pred : int
            Predicted class label.
        """
        self.inference_calls += 1
        scores = np.dot(w.reshape(self.n_states, -1), x)
        if return_energy:
            return np.argmax(scores), np.max(scores)
        return np.argmax(scores)

    def batch_inference(self, X, w, relaxed=None):
        scores = np.dot(X, w.reshape(self.n_states, -1).T)
        return np.argmax(scores, axis=1)

    def oracle(self, scores):
        """Computes the result of the oracle q_hat and the resulting energy given the scores.

        Returns
        -------
        q_hat: ndarray, shape (n_classes)
            Probability with highest error.
        """
        ind, en = np.argsort(scores)[::-1], np.sort(scores)[::-1]
        en = [en[:(j+1)].sum() / (j + 1) - 1 / (j + 1) for j in range(self.n_states)]
        # en = [en[j] / (j + 1) - 1 / (j + 1) for j in range(self.n_states)]
        maxx = np.argmax(en)
        inds = ind[:(maxx + 1)]
        val = np.sum(en[:(maxx + 1)]) + 1
        q_hat = np.zeros_like(scores)
        q_hat[inds] = 1 / (maxx.astype(float) + 1.0)
        # pdb.set_trace()
        return q_hat, val

    def general_loss_oracle(self, scores, mu0, 
                        max_iter = 10, check_dual_gap=False, eta=1.):
        scores = np.expand_dims(scores, 1)
        nu = np.expand_dims(mu0, 1)
        y = np.argmin(np.dot(self.Loss, mu0), 0)
        p = np.expand_dims(np.zeros(self.n_states), 1)
        p[y] = 1
        # pdb.set_trace()
        # nu = np.ones((self.n_states, 1)) / self.n_states
        # eta predicted by theory
        # eta = 1 / (2 * np.linalg.norm(self.Loss, ord=2) * np.log(self.n_states))
        mu_avg = np.zeros((self.n_states, 1))
        q_avg = np.zeros((self.n_states, 1))
        for k in range(max_iter):
            q = sp.softmax(-eta * np.dot(self.Loss, nu) + np.log(p + 1e-3) + 1)
            mu = sp.softmax(eta * np.dot(self.Loss.T, p) + eta * scores + np.log(nu + 1e-3) + 1)
            p = sp.softmax(-eta * np.dot(self.Loss, mu) + np.log(q + 1e-3) + 1)
            nu = sp.softmax(eta * np.dot(self.Loss.T, q) + eta * scores + np.log(mu + 1e-3) + 1)
            q_avg = k * q_avg / (k+1) + q / (k+1) 
            mu_avg = k * mu_avg / (k+1) + mu / (k+1) 
            if k == (max_iter - 1):
                m1 = np.max(np.dot(self.Loss.T, q_avg) + scores)
                m2 = np.min(np.dot(self.Loss, mu_avg) 
                                        + np.dot(scores.T, mu_avg))
                dual_gap = m1 - m2
                # print("dual gap %f", dual_gap)
        en = np.dot(q_avg.T, np.dot(self.Loss, mu_avg) + np.dot(scores.T, mu_avg))
        return mu_avg.ravel(), np.asscalar(en), dual_gap

    def loss_augmented_inference(self, x, mu, w, relaxed=None,
                                 return_energy=False):
        """Loss-augmented inference for x and y using parameters w.

        Minimizes over q_hat:
        np.dot(np.dot(joint_feature(x, -), q_hat), w) + min_y np.dot(loss(y, -), q_hat))

        Parameters
        ----------
        x : ndarray, shape (n_features,)
            Unary evidence / input to augment.

        y : int
            Ground truth labeling relative to which the loss
            will be measured. NOT USEED IN THIS GENERALIZED CASE!

        w : ndarray, shape (size_joint_feature,)
            Weights that will be used for inference.

        Returns
        -------
        q_hat : ndarray, shape (n_classes)
            Probability with highest error
        """
        self.inference_calls += 1
        scores = np.dot(w.reshape(self.n_states, -1), x)
        # other_classes = np.arange(self.n_states) != y
        # else:
        #     scores[other_classes] += self.class_weight[y]
        # q_hat2, en1 = self.oracle(scores)
        q_hat2, en2, err_oracle = self.general_loss_oracle(scores, mu)
        # print("q_hat exact", q_hat1)
        # print("q_hat app", q_hat2)
        # print("en1 %f en2 %f", (en1, en2))
        # pdb.set_trace()
        err_oracle = 0
        return q_hat2, err_oracle


    def batch_loss_augmented_inference(self, X, mu_hats, w, relaxed=None):
        """
        Returns
        -------
        Q_hats: ndarray, shape (batch_size, n_classes)
            Matrix of Probabilities with highest error.
        """
        scores = np.dot(X, w.reshape(self.n_states, -1).T)
        Q_hats = np.zeros((scores.shape[0], self.n_states))
        for j in range(scores.shape[0]):
            # q_hat, _ = self.oracle(scores[j])
            q_hat, _ = self.loss_augmented_inference(X[j], mu_hats[j], w)
            Q_hats[j] = q_hat
        return Q_hats

    def loss(self, y, y_hat):
        return self.Loss[y, y_hat]
        # return self.class_weight[y] * (y != y_hat)

    def batch_loss(self, Y, Y_hat):
        losses = [self.loss(Y[i], Y_hat[i]) for i in range(Y.shape[0])]
        return np.array(losses)
        # return self.class_weight[Y] * (Y != Y_hat)

    def cond_loss(self, y, q):
        cond_loss = np.dot(self.Loss, np.expand_dims(q, 1))
        return cond_loss[y]
        # return self.class_weight[y] * (1 - q[y])

    def batch_cond_loss(self, Y, Q):
        # not implemented for class_weight different
        cond_losses = [self.cond_loss(Y[i], Q[i]) for i in range(Y.shape[0])]
        return np.array(cond_losses)
        # return self.class_weight[Y] * (1 - Q[np.arange(0, Y.shape[0]), Y.astype(int)])
        # return 1 - Q[np.arange(0, Y.shape[0]), Y.astype(int)]

    def Bayes_risk(self, q):
        cond_loss = np.dot(self.Loss, np.expand_dims(q, 1))
        opt = np.min(cond_loss)
        return opt

    def batch_Bayes_risk(self, Q):
        bayes_risks = [self.Bayes_risk(Q[i]) for i in range(Q.shape[0])]
        return np.array(bayes_risks)
