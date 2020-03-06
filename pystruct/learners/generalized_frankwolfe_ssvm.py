######################
# Authors:
#   Xianghang Liu <xianghangliu@gmail.com>
#   Andreas Mueller <amueller@ais.uni-bonn.de>
#
# License: BSD 3-clause
#
# Implements structured SVM as described in Joachims et. al.
# Cutting-Plane Training of Structural SVMs

import warnings
from time import time
import numpy as np
import pdb;
from sklearn.utils import check_random_state

from pystruct.learners.ssvm import BaseSSVM
from pystruct.utils import generalized_find_constraint


class GeneralizedFrankWolfeSSVM(BaseSSVM):
    """Structured SVM solver using Block-coordinate Frank-Wolfe.

    This implementation is somewhat experimental. Use with care.

    References
    ----------
    * Lacoste-Julien, Jaggi, Schmidt, Pletscher:
        Block-Coordinate Frank-Wolfe Optimization for Structural SVMs,xi
        JMLR 2013

    With batch_mode=False, this implements the online (block-coordinate)
    version of the algorithm (BCFW)
    BCFW is an attractive alternative to subgradient methods, as no
    learning rate is needed and a duality gap guarantee is given.

    Parameters
    ----------
    model : StructuredModel
        Object containing the model structure. Has to implement
        `loss`, `inference` and `loss_augmented_inference`.

    max_iter : int, default=1000
        Maximum number of passes over dataset to find constraints.

    C : float, default=1
        Regularization parameter. Corresponds to 1 / (lambda * n_samples).

    verbose : int
        Verbosity.

    n_jobs : int, default=1
        Number of parallel processes. Currently only n_jobs=1 is supported.

    show_loss_every : int, default=0
        How often the training set loss should be computed.
        Zero corresponds to never.

    tol : float, default=1e-3
        Convergence tolerance on the duality gap.

    logger : logger object, default=None
        Pystruct logger for storing the model or extracting additional
        information.

    batch_mode : boolean, default=False
        Whether to use batch updates. Will slow down learning enormously.

    line_search : boolean, default=True
        Whether to compute the optimum step size in each step.
        The line-search is done in closed form and cheap.
        There is usually no reason to turn this off.

    check_dual_every : int, default=10
        How often the stopping criterion should be checked. Computing
        the stopping criterion is as costly as doing one pass over the dataset,
        so check_dual_every=1 will make learning twice as slow.

    do_averaging : bool, default=True
        Whether to use weight averaging as described in the reference paper.
        Currently this is only supported in the block-coordinate version.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.


    Attributes
    ----------
    w : nd-array, shape=(model.size_joint_feature,)
        The learned weights of the SVM.

    ``loss_curve_`` : list of float
        List of loss values if show_loss_every > 0.

    ``objective_curve_`` : list of float
       Cutting plane objective after each pass through the dataset.

    ``primal_objective_curve_`` : list of float
        Primal objective after each pass through the dataset.

    ``timestamps_`` : list of int
       Total training time stored before each iteration.
    """
    def __init__(self, model, max_iter=1000, C=1.0, verbose=0, n_jobs=1,
                 show_loss_every=0, logger=None, batch_mode=False,
                 line_search=True, check_dual_every=10, tol=.001,
                 do_averaging=True, sample_method='perm', random_state=None, X_test=None, Y_test=None):

        if n_jobs != 1:
            warnings.warn("FrankWolfeSSVM does not support multiprocessing"
                          " yet. Ignoring n_jobs != 1.")

        if sample_method not in ['perm', 'rnd', 'seq']:
            raise ValueError("sample_method can only be perm, rnd, or seq")

        BaseSSVM.__init__(self, model, max_iter, C, verbose=verbose,
                          n_jobs=n_jobs, show_loss_every=show_loss_every,
                          logger=logger)
        self.tol = tol
        self.batch_mode = batch_mode
        self.line_search = line_search
        self.check_dual_every = check_dual_every
        self.do_averaging = do_averaging
        self.sample_method = sample_method
        self.random_state = random_state
        self.X_test = X_test
        self.Y_test = Y_test
        self.oracle_errs = []

    def _calc_dual_gap(self, X, Y, mu, mu_hats, iteration):
        n_samples = len(X)
        # FIXME don't calculate this again
        joint_feature_gt = self.model.batch_joint_feature(X, Y)
        
        # in order to compute the dual gap we need a full pass over the data
        Q_hat = self.model.batch_loss_augmented_inference(X, mu_hats, self.w,
                                                          relaxed=True)
        djoint_feature = joint_feature_gt - self.model.batch_mean_joint_feature(X, Q_hat)
        ls = np.sum(self.model.batch_Bayes_risk(Q_hat))
        ws = djoint_feature * self.C
        # l = np.sum(self.model.batch_Bayes_risk(mu)) / n_samples
        l_rescaled = self.l * n_samples * self.C
        # l_rescaled = l * n_samples * self.C
        dual_val = -0.5 * np.sum(self.w ** 2) + l_rescaled
        w_diff = self.w - ws
        dual_gap = w_diff.T.dot(self.w) - l_rescaled + ls * self.C
        # both must be the same
        primal_val = dual_val + dual_gap
        primal_val = 0.5 * np.sum(self.w ** 2) + self.C * ls - ws.T.dot(self.w)
        # import pdb; pdb.set_trace()
        return dual_val, dual_gap, primal_val

    def _gfrank_wolfe_bc(self, X, Y):
        """Generalized Block-Coordinate Frank-Wolfe learning.
        """
        n_samples = len(X)
        
        w = self.w.copy()
        w_mat = np.zeros((n_samples, self.model.size_joint_feature))
        mu = self.model.output_embedding(X, Y)
        mu_hats = mu
        l_mat = np.zeros(n_samples)
        l = 0.0
        k = 0

        rng = check_random_state(self.random_state)
        for iteration in range(self.max_iter):
            # print("Full pass: ", iteration)
            perm = np.arange(n_samples)
            if self.sample_method == 'perm':
                rng.shuffle(perm)
            elif self.sample_method == 'rnd':
                perm = rng.randint(low=0, high=n_samples, size=n_samples)
            oracle_err_avg = 0
            for j in range(n_samples):
                i = perm[j]
                x, y = X[i], Y[i]
                ###############################################################
                #  Find constraint
                ###############################################################
                mu_hat, oracle_err = self.model.loss_augmented_inference(x, mu_hats[i], w, relaxed=True)
                delta_joint_feature = self.model.joint_feature(x, y) - self.model.mean_joint_feature(x, mu_hat)
                loss = self.model.Bayes_risk(mu_hat)
                # mu_hat, delta_joint_feature, slack, loss, oracle_err = generalized_find_constraint(self.model, x, y, mu_hats[i], w)
                oracle_err_avg += oracle_err
                ###############################################################
                #  Compute ws and ls
                ###############################################################
                ws = delta_joint_feature * self.C
                ls = loss / n_samples

                # no line search
                gamma = 2.0 * n_samples / (k + 2.0 * n_samples)

                w -= w_mat[i]
                w_mat[i] = (1.0 - gamma) * w_mat[i] + gamma * ws
                w += w_mat[i]
                # if iteration > 10:
                #     pdb.set_trace()
                mu_hats[i] = mu_hat
                mu[i] = (1.0 - gamma) * mu[i] + gamma * mu_hat
                l_mat_i = self.model.Bayes_risk(mu[i]) / n_samples
                l = l + l_mat_i - l_mat[i] 
                l_mat[i] = l_mat_i

                if self.do_averaging:
                    rho = 2. / (k + 2.)
                    self.w = (1. - rho) * self.w + rho * w
                    self.l = (1. - rho) * self.l + rho * l
                else:
                    self.w = w
                    self.l = l
                k += 1
            self.oracle_errs.append(oracle_err_avg / n_samples)
            # if (self.check_dual_every != 0) and (iteration % self.check_dual_every == 0):
            #     score_train = self.score(X, Y)
            #     score_test = self.score(self.X_test, self.Y_test)
            #     print("Error oracles %f", self.oracle_errs[-1])
            #     print("SCORES %f", score_train, score_test)
            if (self.check_dual_every != 0) and (iteration % self.check_dual_every == 0):
                dual_val, dual_gap, primal_val = self._calc_dual_gap(X, Y, mu, mu_hats, iteration)
                self.primal_objective_curve_.append(primal_val)
                self.objective_curve_.append(dual_val)
                self.timestamps_.append(time() - self.timestamps_[0])
                if self.verbose > 0:
                    print("dual: %f, dual_gap: %f, primal: %f, Iteration: %d"
                          % (dual_val, dual_gap, primal_val, iteration))
            if self.logger is not None:
                self.logger(self, iteration)

    def fit(self, X, Y, constraints=None, initialize=True):
        """Learn parameters using (block-coordinate) Frank-Wolfe learning.

        Parameters
        ----------
        X : iterable
            Traing instances. Contains the structured input objects.
            No requirement on the particular form of entries of X is made.

        Y : iterable
            Training labels. Contains the strctured labels for inputs in X.
            Needs to have the same length as X.

        contraints : ignored

        initialize : boolean, default=True
            Whether to initialize the model for the data.
            Leave this true except if you really know what you are doing.
        """
        if initialize:
            self.model.initialize(X, Y)
        self.objective_curve_, self.primal_objective_curve_ = [], []
        self.timestamps_ = [time()]
        self.w = getattr(self, "w", np.zeros(self.model.size_joint_feature))
        self.l = getattr(self, "l", 0)
        try:
            if self.batch_mode:
                self._frank_wolfe_batch(X, Y)
            else:
                self._gfrank_wolfe_bc(X, Y)
        except KeyboardInterrupt:
            pass
        if self.verbose:
            print("Calculating final objective.")
        self.timestamps_.append(time() - self.timestamps_[0])
        # self.primal_objective_curve_.append(self._objective(X, Y))
        # self.objective_curve_.append(self.objective_curve_[-1])
        if self.logger is not None:
            self.logger(self, 'final')
        return self
