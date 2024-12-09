import numpy as np
from tqdm.auto import tqdm


def softmax(logits: np.ndarray) -> np.ndarray:
    # Subtract the max value for numerical stability
    logits_sub_max = logits - np.max(logits, axis=-1, keepdims=True)

    # Clip values to avoid extreme underflows
    logits_clipped = np.clip(logits_sub_max, -700, None)

    # Compute softmax
    numer = np.exp(logits_clipped)
    denom = np.sum(numer, axis=-1, keepdims=True) + 1e-10  # Add epsilon to prevent division by zero
    return numer / denom


def logsumexp(logits: np.ndarray) -> np.ndarray:
    """
    Compute log-sum-exp in a numerically stable way.

    Args:
        logits (np.ndarray): Input array.

    Returns:
        np.ndarray: Log-sum-exp computed along the specified axis.
    """
    # Subtract the max value for numerical stability
    max_logits = np.max(logits, axis=-1, keepdims=True)
    logits_sub_max = logits - max_logits

    # Clip values to avoid extreme underflows
    logits_clipped = np.clip(logits_sub_max, -700, None)

    # Compute log-sum-exp
    sum_exp = np.sum(np.exp(logits_clipped), axis=-1, keepdims=True) + 1e-10  # Add epsilon
    log_sum_exp = np.log(sum_exp) + max_logits
    return log_sum_exp


def log_normal_pdf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray, covariance_type: str) -> np.ndarray:
    """
    Compute the log of the Gaussian PDF in a numerically stable way.

    Args:
        x (np.ndarray): NxD array of data points.
        mu (np.ndarray): 1xD or KxD array of means for each Gaussian.
        sigma (np.ndarray): Covariance representation:
                            - Diagonal: 1xD or KxD array (variances).
                            - Full: 1xDxD or KxDxD array (covariance matrices).
        covariance_type (str): "diagonal" or "full" to specify the covariance type.

    Returns:
        log_pdf (np.ndarray): Nx1 array of log probabilities for each data point.
    """
    N, D = x.shape
    diff = x - mu

    # Thresholding to avoid underflow in (diff ** 2)
    diff = np.maximum(np.abs(diff), 1e-10) * np.sign(diff)

    if covariance_type == "diagonal":
        # Regularize sigma and calculate log determinant
        sigma = np.clip(sigma, 1e-10, None)
        log_det_sigma = np.sum(np.log(sigma))

        # Compute quadratic form (independent variances)
        quad_form = np.sum((diff ** 2) / sigma, axis=1)

    elif covariance_type == "full":
        # Regularize full covariance matrix
        D = sigma.shape[1]
        sigma += 1e-10 * np.eye(D)

        # Compute log determinant and inverse
        log_det_sigma = np.log(np.linalg.det(sigma))
        sigma_inv = np.linalg.inv(sigma)

        # Compute quadratic form (correlated features)
        quad_form = np.einsum("ni,ij,nj->n", diff, sigma_inv, diff)

    else:
        raise ValueError("Unsupported covariance_type. Must be 'diagonal' or 'full'.")

    # Shared computation for both cases
    log_pdf = -0.5 * (log_det_sigma + quad_form + D * np.log(2 * np.pi))
    return log_pdf


class SemiSupervisedGMM:
    def __init__(self, K=2, max_iter=10000, abs_tol=1e-6, rel_tol=1e-6, covariance_type="diagonal"):
        """
        Args:
            K: number of clusters
            max_iter: maximum number of iterations
            abs_tol: absolute tolerance for convergence
            rel_tol: relative tolerance for convergence
            covariance_type: "diagonal" or "full"
        """
        self.K = K
        self.max_iter = max_iter
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.covariance_type = covariance_type
        
        self.pi = None
        self.mu = None
        self.sigma = None

    def _init_components(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize the parameters using labeled data.
        """
        N, D = X.shape
        unique, counts = np.unique(y, return_counts=True)
        assert len(unique) == self.K, "Number of classes must match K."

        # Initialize class priors with Laplace smoothing
        self.pi = (counts + 1) / (np.sum(counts) + len(counts))

        # Initialize mean and covariance
        self.mu = np.empty((self.K, D))
        if self.covariance_type == "diagonal":
            self.sigma = np.empty((self.K, D))
        elif self.covariance_type == "full":
            self.sigma = np.empty((self.K, D, D))

        for k in range(self.K):
            members = X[y == k]  # Data points for class k
            self.mu[k] = np.mean(members, axis=0)  # Compute mean

            if len(members) == 1:
                # Handle single-sample case
                if self.covariance_type == "diagonal":
                    self.sigma[k] = np.var(X, axis=0) + 1e-6  # Use global variance
                elif self.covariance_type == "full":
                    self.sigma[k] = np.cov(X, rowvar=False) + 1e-6 * np.eye(D)  # Use global covariance
                else:
                    raise AttributeError
            else:
                # Compute variance or covariance for multiple samples
                if self.covariance_type == "diagonal":
                    self.sigma[k] = np.var(members, axis=0) + 1e-6  # Regularize diagonal
                elif self.covariance_type == "full":
                    self.sigma[k] = np.cov(members, rowvar=False) + 1e-6 * np.eye(D)  # Regularize full
                else:
                    raise AttributeError

    def _ll_joint(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the log-likelihood joint for each data point and component.
        """
        ll = np.empty((X.shape[0], self.K))
        for k in range(self.K):
            ll[:, k] = np.log(self.pi[k]) + log_normal_pdf(X, self.mu[k], self.sigma[k], self.covariance_type)
        return ll

    def _E_step(self, X: np.ndarray) -> np.ndarray:
        """
        E-step: Compute the posterior probabilities (soft assignments).
        """
        ll = self._ll_joint(X)
        return softmax(ll)
    
    def _M_step(self, X: np.ndarray, gamma: np.ndarray):
        """
        Perform the M-step to update the parameters of the Gaussian Mixture Model.

        Args:
            X (np.ndarray): NxD array of data points, where N is the number of samples and D is the dimensionality.
            gamma (np.ndarray): NxK array of posterior probabilities (soft assignments), where K is the number of components.

        Updates:
            self.pi: Updated mixture weights (priors), a 1D array of length K.
            self.mu: Updated means, a KxD array where each row is the mean of a component.
            self.sigma: Updated covariance matrices. Shape depends on `covariance_type`:
                        - "diagonal": KxD array (diagonal covariances for each component).
        """
        N, D = X.shape
        gamma_sum = np.sum(gamma, axis=0) + 1e-10  # Sum over data points for each component (regularized)

        # Scale X to avoid underflow issues
        scaling_factor = np.max(np.abs(X), axis=0) + 1e-10
        X_scaled = X / scaling_factor

        # Update priors (pi)
        self.pi = gamma_sum / N

        # Update means (mu) relative to scaled data
        self.mu = np.dot(gamma.T, X_scaled) / gamma_sum[:, None]
        self.mu *= scaling_factor  # Rescale means back to the original scale

        # Update covariances
        if self.covariance_type == "diagonal":
            self.sigma = np.empty((self.K, D))
            for k in range(self.K):
                gamma_k = np.clip(gamma[:, k], 1e-10, None)  # Clip gamma to avoid underflow
                diff = X_scaled - (self.mu[k] / scaling_factor)
                
                # Apply a minimum threshold to diff to prevent underflow
                diff_thresholded = np.maximum(np.abs(diff), 1e-10) * np.sign(diff)

                # Compute weighted variance
                weighted_diff2 = gamma_k[:, None] * (diff_thresholded ** 2)
                self.sigma[k] = np.sum(weighted_diff2, axis=0) / gamma_sum[k]
                self.sigma[k] *= scaling_factor ** 2  # Rescale covariances back
                self.sigma[k] += 1e-10  # Regularize
        elif self.covariance_type == "full":
            self.sigma = np.empty((self.K, D, D))
            for k in range(self.K):
                gamma_k = np.clip(gamma[:, k], 1e-10, None)
                diff = X_scaled - (self.mu[k] / scaling_factor)
                
                # Apply a minimum threshold to diff to prevent underflow
                diff_thresholded = np.maximum(np.abs(diff), 1e-10) * np.sign(diff)

                # Compute weighted covariance
                self.sigma[k] = (
                    np.einsum("ni,nj->ij", gamma_k[:, None] * diff_thresholded, diff_thresholded) / gamma_sum[k]
                )
                self.sigma[k] *= scaling_factor[:, None] * scaling_factor[None, :]  # Rescale full covariances
                self.sigma[k] += 1e-10 * np.eye(D)  # Regularize
        else:
            raise ValueError

    def fit(self, Xl: np.ndarray, yl: np.ndarray, Xu: np.ndarray = None):
        """
        Fit the model using the EM algorithm.
        """
        self._init_components(Xl, yl)
        N_labeled = Xl.shape[0]
        X_combined = np.vstack([Xl, Xu]) if Xu is not None else Xl

        gamma_labeled = np.zeros((N_labeled, self.K))
        gamma_labeled[np.arange(N_labeled), yl] = 1

        pbar = tqdm(range(self.max_iter))
        prev_loss = None
        for it in pbar:
            if Xu is not None:
                gamma_unlabeled = self._E_step(Xu)
                gamma_combined = np.vstack([gamma_labeled, gamma_unlabeled])
            else:
                gamma_combined = gamma_labeled

            # M-step
            self._M_step(X_combined, gamma_combined)

            # Compute loss
            joint_ll = self._ll_joint(X_combined)
            loss = -np.sum(logsumexp(joint_ll))
            if prev_loss is not None:
                diff = np.abs(prev_loss - loss)
                if diff < self.abs_tol and diff / np.abs(prev_loss) < self.rel_tol:
                    break
            prev_loss = loss
            pbar.set_description(f"Iter {it}, Loss: {loss:.4f}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict soft class assignments (posterior probabilities).
        """
        return self._E_step(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict hard class assignments.
        """
        return np.argmax(self.predict_proba(X), axis=1)
