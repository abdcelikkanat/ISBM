"""
Collapsed Gibbs Sampler (CGS) for the Infinite Relational Model.

Reference: Ishiguro, Sato, Ueda (2014) "Collapsed Variational Bayes Inference
of Infinite Relational Model", Section 3, Equations (15)-(24).

Sufficient statistics
---------------------
n[k, l]  – # of positive links (x=1) between domain-1 cluster k and domain-2 cluster l
Nm[k, l] – # of negative links (x=0) between domain-1 cluster k and domain-2 cluster l
m1[k]    – # of objects in domain-1 cluster k
m2[l]    – # of objects in domain-2 cluster l
"""

import numpy as np
from scipy.special import gammaln


def _wandb_log(step, metrics):
    """Log to wandb if it is initialised, silently skip otherwise."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except ImportError:
        pass


class CGS_IRM:
    """
    Collapsed Gibbs Sampler for the two-domain binary Infinite Relational Model.

    Generative model (Eq. 6-9):
        theta_{k,l} ~ Beta(a, b)
        z1_i        ~ CRP(alpha1)      i = 1 ... N1
        z2_j        ~ CRP(alpha2)      j = 1 ... N2
        x_{i,j}     ~ Bernoulli(theta_{z1_i, z2_j})

    Parameters
    ----------
    alpha1, alpha2 : float
        CRP concentration parameters.
    a, b : float
        Symmetric Beta prior hyperparameters for link probabilities.
    n_iter : int
        Total Gibbs sweeps.
    burnin : int
        Sweeps discarded before collecting samples.
    seed : int or None
    """

    def __init__(self, alpha1=1.0, alpha2=1.0, a=1.0, b=1.0,
                 n_iter=500, burnin=250, seed=None, wandb_log=False):
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.a = a
        self.b = b
        self.n_iter = n_iter
        self.burnin = burnin
        self.seed = seed
        self.wandb_log = wandb_log

    # ------------------------------------------------------------------
    def fit(self, X):
        """
        Run CGS on binary relation matrix X of shape (N1, N2).

        Stores
        ------
        z1_, z2_             : final (compacted) cluster assignments
        z1_samples_, z2_samples_ : post-burnin samples
        ll_trace_            : per-sweep pseudo log-likelihood
        """
        rng = np.random.default_rng(self.seed)
        X   = np.asarray(X, dtype=np.float64)
        N1, N2 = X.shape

        # -- Initialise: each object in its own singleton cluster ------
        z1 = np.arange(N1, dtype=int)
        z2 = np.arange(N2, dtype=int)
        K  = max(N1, N2)           # upper bound on cluster indices

        m1  = np.zeros(K, dtype=float)
        m2  = np.zeros(K, dtype=float)
        n   = np.zeros((K, K), dtype=float)
        Nm  = np.zeros((K, K), dtype=float)

        for i in range(N1):
            m1[i] += 1
        for j in range(N2):
            m2[j] += 1

        # n[k, l] = Σ_i I(z1_i=k) Σ_j I(z2_j=l) x_{i,j}
        for i in range(N1):
            for j in range(N2):
                n [z1[i], z2[j]] += X[i, j]
                Nm[z1[i], z2[j]] += 1.0 - X[i, j]

        self.z1_samples_ = []
        self.z2_samples_ = []
        self.ll_trace_   = []

        for it in range(self.n_iter):

            # ---- Sweep domain 1 (Eq. 20) ----------------------------
            for i in rng.permutation(N1):
                k_old = z1[i]

                # Remove i's contribution to n, Nm, m1
                m1[k_old] -= 1
                np.add.at(n [k_old], z2, -X[i, :])
                np.add.at(Nm[k_old], z2, -(1.0 - X[i, :]))

                # n+(1,i,k) for any candidate k is the same:
                # n_plus[l] = Σ_j I(z2_j=l) x_{i,j}  (Eq. 18 – deterministic given z2)
                n_plus  = np.zeros(K, dtype=float)
                N_plus  = np.zeros(K, dtype=float)
                np.add.at(n_plus,  z2, X[i, :])
                np.add.at(N_plus,  z2, 1.0 - X[i, :])

                active2 = np.where(m2 > 0)[0]

                # Build log-prob for existing clusters (Eq. 20 top)
                existing1 = np.where(m1 > 0)[0]
                log_p = []
                ids   = []
                for k in existing1:
                    nk  = n [k, active2]
                    Nmk = Nm[k, active2]
                    np_ = n_plus [active2]
                    Np_ = N_plus [active2]
                    lp  = (np.log(m1[k])
                           + np.sum(
                               gammaln(self.a + nk + np_)
                             + gammaln(self.b + Nmk + Np_)
                             - gammaln(self.a + self.b + nk + Nmk + np_ + Np_)
                             - gammaln(self.a + nk)
                             - gammaln(self.b + Nmk)
                             + gammaln(self.a + self.b + nk + Nmk)))
                    log_p.append(lp);  ids.append(k)

                # New cluster (Eq. 20 bottom)
                np_ = n_plus [active2]
                Np_ = N_plus [active2]
                lp_new = (np.log(self.alpha1)
                          + np.sum(
                              gammaln(self.a + np_)
                            + gammaln(self.b + Np_)
                            - gammaln(self.a + self.b + np_ + Np_)
                            - gammaln(self.a) - gammaln(self.b)
                            + gammaln(self.a + self.b)))
                log_p.append(lp_new);  ids.append(-1)

                log_p = np.array(log_p)
                log_p -= log_p.max()
                p = np.exp(log_p);  p /= p.sum()
                chosen = ids[rng.choice(len(ids), p=p)]

                if chosen == -1:
                    chosen = int(np.where(m1 == 0)[0][0])

                z1[i]         = chosen
                m1[chosen]   += 1
                np.add.at(n [chosen], z2,  X[i, :])
                np.add.at(Nm[chosen], z2,  1.0 - X[i, :])

            # ---- Sweep domain 2 (Eq. 24, symmetric) -----------------
            for j in rng.permutation(N2):
                l_old = z2[j]

                m2[l_old] -= 1
                np.add.at(n [:, l_old], z1, -X[:, j])
                np.add.at(Nm[:, l_old], z1, -(1.0 - X[:, j]))

                n_plus  = np.zeros(K, dtype=float)
                N_plus  = np.zeros(K, dtype=float)
                np.add.at(n_plus,  z1, X[:, j])
                np.add.at(N_plus,  z1, 1.0 - X[:, j])

                active1   = np.where(m1 > 0)[0]
                existing2 = np.where(m2 > 0)[0]
                log_p = [];  ids = []
                for l in existing2:
                    nl  = n [active1, l]
                    Nml = Nm[active1, l]
                    np_ = n_plus [active1]
                    Np_ = N_plus [active1]
                    lp  = (np.log(m2[l])
                           + np.sum(
                               gammaln(self.a + nl + np_)
                             + gammaln(self.b + Nml + Np_)
                             - gammaln(self.a + self.b + nl + Nml + np_ + Np_)
                             - gammaln(self.a + nl)
                             - gammaln(self.b + Nml)
                             + gammaln(self.a + self.b + nl + Nml)))
                    log_p.append(lp);  ids.append(l)

                np_ = n_plus [active1]
                Np_ = N_plus [active1]
                lp_new = (np.log(self.alpha2)
                          + np.sum(
                              gammaln(self.a + np_)
                            + gammaln(self.b + Np_)
                            - gammaln(self.a + self.b + np_ + Np_)
                            - gammaln(self.a) - gammaln(self.b)
                            + gammaln(self.a + self.b)))
                log_p.append(lp_new);  ids.append(-1)

                log_p = np.array(log_p)
                log_p -= log_p.max()
                p = np.exp(log_p);  p /= p.sum()
                chosen = ids[rng.choice(len(ids), p=p)]

                if chosen == -1:
                    chosen = int(np.where(m2 == 0)[0][0])

                z2[j]         = chosen
                m2[chosen]   += 1
                np.add.at(n [:, chosen], z1,  X[:, j])
                np.add.at(Nm[:, chosen], z1,  1.0 - X[:, j])

            # Pseudo log-likelihood
            ll = self._pseudo_ll(n, Nm, m1, m2)
            self.ll_trace_.append(ll)

            if self.wandb_log:
                k1_eff = int((m1 > 0).sum())
                k2_eff = int((m2 > 0).sum())
                _wandb_log(it, {
                    "cgs/pseudo_ll":    ll,
                    "cgs/K1_active":    k1_eff,
                    "cgs/K2_active":    k2_eff,
                    "cgs/phase":        "burnin" if it < self.burnin else "sampling",
                })

            if it >= self.burnin:
                self.z1_samples_.append(_compact(z1))
                self.z2_samples_.append(_compact(z2))

        self.z1_ = _compact(z1)
        self.z2_ = _compact(z2)
        self.n_  = n
        self.Nm_ = Nm
        self.m1_ = m1
        self.m2_ = m2
        return self

    def predict(self, X, i_test, j_test):
        """Marginal test log-likelihood using final sample."""
        z1, z2 = self.z1_, self.z2_
        log_ll = 0.0
        for i, j in zip(i_test, j_test):
            k, l   = z1[i], z2[j]
            a_post = self.a  + self.n_ [k, l]
            b_post = self.b  + self.Nm_[k, l]
            theta  = a_post / (a_post + b_post)
            xij    = X[i, j]
            log_ll += (xij * np.log(theta + 1e-300)
                       + (1 - xij) * np.log(1 - theta + 1e-300))
        return log_ll / max(len(i_test), 1)

    # ------------------------------------------------------------------
    def _pseudo_ll(self, n, Nm, m1, m2):
        a, b = self.a, self.b
        act1 = np.where(m1 > 0)[0]
        act2 = np.where(m2 > 0)[0]
        nk   = n [np.ix_(act1, act2)]
        Nmk  = Nm[np.ix_(act1, act2)]
        return float(np.sum(
            gammaln(a + nk) + gammaln(b + Nmk)
            - gammaln(a + b + nk + Nmk)
            - gammaln(a) - gammaln(b) + gammaln(a + b)))


# ------------------------------------------------------------------
def _compact(z):
    """Re-index labels to 0, 1, 2, ... with no gaps."""
    mapping = {}
    out = np.empty_like(z)
    for i, k in enumerate(z):
        if k not in mapping:
            mapping[k] = len(mapping)
        out[i] = mapping[k]
    return out