# File: harmony.py
# harmonypy - A data alignment algorithm.
# Copyright (C) 2018  Ilya Korsunsky
#               2019  Kamil Slowikowski <kslowikowski@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import logging

# create logger
logger = logging.getLogger('harmonypy')
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def get_device(device=None):
    """Get the appropriate device for PyTorch operations."""
    if device is not None:
        return torch.device(device)

    # Check for available accelerators
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def run_harmony(
    data_mat: np.ndarray,
    meta_data: pd.DataFrame,
    vars_use,
    theta=None,
    lamb=None,
    sigma=0.1,
    nclust=None,
    tau=0,
    block_size=0.05,
    max_iter_harmony=10,
    max_iter_kmeans=20,
    epsilon_cluster=1e-5,
    epsilon_harmony=1e-4,
    alpha=0.2,
    verbose=True,
    random_state=0,
    device=None
):
    """Run Harmony batch effect correction.

    This is a PyTorch implementation matching the R package formulas.
    Supports CPU and GPU (CUDA, MPS) acceleration.

    Parameters
    ----------
    data_mat : np.ndarray
        PCA embedding matrix (cells x PCs or PCs x cells)
    meta_data : pd.DataFrame
        Metadata with batch variables (cells x variables)
    vars_use : str or list
        Column name(s) in meta_data to use for batch correction
    theta : float or list, optional
        Diversity penalty parameter(s). Default is 2 for each batch.
    lamb : float or list, optional
        Ridge regression penalty. Default is 1 for each batch.
        If -1, lambda is estimated automatically (matches R package).
    sigma : float, optional
        Kernel bandwidth for soft clustering. Default is 0.1.
    nclust : int, optional
        Number of clusters. Default is min(N/30, 100).
    tau : float, optional
        Protection against overcorrection. Default is 0.
    block_size : float, optional
        Proportion of cells to update in each block. Default is 0.05.
    max_iter_harmony : int, optional
        Maximum Harmony iterations. Default is 10.
    max_iter_kmeans : int, optional
        Maximum k-means iterations per Harmony iteration. Default is 20.
    epsilon_cluster : float, optional
        K-means convergence threshold. Default is 1e-5.
    epsilon_harmony : float, optional
        Harmony convergence threshold. Default is 1e-4.
    alpha : float, optional
        Alpha parameter for lambda estimation (when lamb=-1). Default is 0.2.
    verbose : bool, optional
        Print progress messages. Default is True.
    random_state : int, optional
        Random seed for reproducibility. Default is 0.
    device : str, optional
        Device to use ('cpu', 'cuda', 'mps'). Default is auto-detect.

    Returns
    -------
    Harmony
        Harmony object with corrected data in Z_corr attribute.
    """
    N = meta_data.shape[0]
    if data_mat.shape[1] != N:
        data_mat = data_mat.T

    assert data_mat.shape[1] == N, \
       "data_mat and meta_data do not have the same number of cells"

    if nclust is None:
        nclust = int(min(round(N / 30.0), 100))

    if isinstance(sigma, float) and nclust > 1:
        sigma = np.repeat(sigma, nclust)

    if isinstance(vars_use, str):
        vars_use = [vars_use]

    # Create batch indicator matrix (one-hot encoded)
    phi = pd.get_dummies(meta_data[vars_use]).to_numpy().T.astype(np.float32)
    phi_n = meta_data[vars_use].describe().loc['unique'].to_numpy().astype(int)

    # Theta handling - default is 2 (matches R package)
    if theta is None:
        theta = np.repeat([2] * len(phi_n), phi_n).astype(np.float32)
    elif isinstance(theta, (float, int)):
        theta = np.repeat([theta] * len(phi_n), phi_n).astype(np.float32)
    elif len(theta) == len(phi_n):
        theta = np.repeat([theta], phi_n).astype(np.float32)
    else:
        theta = np.asarray(theta, dtype=np.float32)

    assert len(theta) == np.sum(phi_n), \
        "each batch variable must have a theta"

    # Lambda handling (matches R package)
    lambda_estimation = False
    if lamb is None:
        lamb = np.repeat([1] * len(phi_n), phi_n).astype(np.float32)
        lamb = np.insert(lamb, 0, 0).astype(np.float32)
    elif lamb == -1:
        lambda_estimation = True
        lamb = np.zeros(1, dtype=np.float32)
    elif isinstance(lamb, (float, int)):
        lamb = np.repeat([lamb] * len(phi_n), phi_n).astype(np.float32)
        lamb = np.insert(lamb, 0, 0).astype(np.float32)
    elif len(lamb) == len(phi_n):
        lamb = np.repeat([lamb], phi_n).astype(np.float32)
        lamb = np.insert(lamb, 0, 0).astype(np.float32)
    else:
        lamb = np.asarray(lamb, dtype=np.float32)
        if len(lamb) == np.sum(phi_n):
            lamb = np.insert(lamb, 0, 0).astype(np.float32)

    # Number of items in each category
    N_b = phi.sum(axis=1)
    Pr_b = (N_b / N).astype(np.float32)

    if tau > 0:
        theta = theta * (1 - np.exp(-(N_b / (nclust * tau)) ** 2))

    # Get device
    device_obj = get_device(device)

    if verbose:
        logger.info(f"Running Harmony (PyTorch on {device_obj})")
        logger.info("  Parameters:")
        logger.info(f"    max_iter_harmony: {max_iter_harmony}")
        logger.info(f"    max_iter_kmeans: {max_iter_kmeans}")
        logger.info(f"    epsilon_cluster: {epsilon_cluster}")
        logger.info(f"    epsilon_harmony: {epsilon_harmony}")
        logger.info(f"    nclust: {nclust}")
        logger.info(f"    block_size: {block_size}")
        if lambda_estimation:
            logger.info(f"    lamb: dynamic (alpha={alpha})")
        else:
            logger.info(f"    lamb: {lamb[1:]}")
        logger.info(f"    theta: {theta}")
        logger.info(f"    sigma: {sigma[:5]}..." if len(sigma) > 5 else f"    sigma: {sigma}")
        logger.info(f"    verbose: {verbose}")
        logger.info(f"    random_state: {random_state}")
        logger.info(f"  Data: {data_mat.shape[0]} PCs × {N} cells")
        logger.info(f"  Batch variables: {vars_use}")

    # Set random seeds
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    # Ensure data_mat is a proper numpy array
    if hasattr(data_mat, 'values'):
        data_mat = data_mat.values
    data_mat = np.asarray(data_mat, dtype=np.float32)

    ho = Harmony(
        data_mat, phi, Pr_b, sigma.astype(np.float32),
        theta, lamb, alpha, lambda_estimation,
        max_iter_harmony, max_iter_kmeans,
        epsilon_cluster, epsilon_harmony, nclust, block_size, verbose,
        random_state, device_obj,
        max_correction_ratio=0.5, momentum=0.3
    )

    return ho


class Harmony:
    """Harmony class for batch effect correction using PyTorch.

    Supports CPU and GPU acceleration.
    """

    def __init__(
            self, Z, Phi, Pr_b, sigma, theta, lamb, alpha, lambda_estimation,
            max_iter_harmony, max_iter_kmeans,
            epsilon_kmeans, epsilon_harmony, K, block_size, verbose,
            random_state, device,
            max_correction_ratio=0.5, momentum=0.3
    ):
        self.device = device

        # Convert to PyTorch tensors on device
        # Store with underscore prefix internally, expose as properties returning NumPy arrays
        self._Z_corr = torch.tensor(Z, dtype=torch.float32, device=device)
        self._Z_orig = torch.tensor(Z, dtype=torch.float32, device=device)

        # Simple L2 normalization
        self._Z_cos = F.normalize(self._Z_orig, p=2, dim=0)

        # Batch indicators
        self._Phi = torch.tensor(Phi, dtype=torch.float32, device=device)
        self._Pr_b = torch.tensor(Pr_b, dtype=torch.float32, device=device)

        self.N = self._Z_corr.shape[1]
        self.B = Phi.shape[0]
        self.d = self._Z_corr.shape[0]

        # Build batch index for fast ridge correction
        self._batch_index = []
        for b in range(self.B):
            idx = torch.where(self._Phi[b, :] > 0)[0]
            self._batch_index.append(idx)

        # Create Phi_moe with intercept
        ones = torch.ones(1, self.N, dtype=torch.float32, device=device)
        self._Phi_moe = torch.cat([ones, self._Phi], dim=0)

        self.window_size = 2
        self.epsilon_kmeans = epsilon_kmeans
        self.epsilon_harmony = epsilon_harmony

        self._lamb = torch.tensor(lamb, dtype=torch.float32, device=device)
        self.alpha = alpha
        self.lambda_estimation = lambda_estimation
        self._sigma = torch.tensor(sigma, dtype=torch.float32, device=device)
        self.block_size = block_size
        self.K = K
        self.max_iter_harmony = max_iter_harmony
        self.max_iter_kmeans = max_iter_kmeans
        self.verbose = verbose
        self._theta = torch.tensor(theta, dtype=torch.float32, device=device)

        self.max_correction_ratio = max_correction_ratio
        self.momentum = momentum
        self._prev_correction = None

        self.objective_harmony = []
        self.objective_kmeans = []
        self.objective_kmeans_dist = []
        self.objective_kmeans_entropy = []
        self.objective_kmeans_cross = []
        self.kmeans_rounds = []

        self.allocate_buffers()
        with torch.inference_mode():
            self.init_cluster(random_state)
            self.harmonize(self.max_iter_harmony, self.verbose)

    # =========================================================================
    # Properties - Return NumPy arrays for inspection and tutorials
    # =========================================================================

    @property
    def Z_corr(self):
        """Corrected embedding matrix (N x d). Batch effects removed."""
        return self._Z_corr.cpu().numpy().T

    @property
    def Z_orig(self):
        """Original embedding matrix (N x d). Input data before correction."""
        return self._Z_orig.cpu().numpy().T

    @property
    def Z_cos(self):
        """L2-normalized embedding matrix (N x d). Used for clustering."""
        return self._Z_cos.cpu().numpy().T

    @property
    def R(self):
        """Soft cluster assignment matrix (N x K). R[i,k] = P(cell i in cluster k)."""
        return self._R.cpu().numpy().T

    @property
    def Y(self):
        """Cluster centroids matrix (d x K). Columns are cluster centers."""
        return self._Y.cpu().numpy()

    @property
    def O(self):
        """Observed batch-cluster counts (K x B). O[k,b] = sum of R[k,:] for batch b."""
        return self._O.cpu().numpy()

    @property
    def E(self):
        """Expected batch-cluster counts (K x B). E[k,b] = cluster_size[k] * batch_proportion[b]."""
        return self._E.cpu().numpy()

    @property
    def Phi(self):
        """Batch indicator matrix (N x B). One-hot encoding of batch membership."""
        return self._Phi.cpu().numpy().T

    @property
    def Phi_moe(self):
        """Batch indicator with intercept (N x (B+1)). First column is all ones."""
        return self._Phi_moe.cpu().numpy().T

    @property
    def Pr_b(self):
        """Batch proportions (B,). Pr_b[b] = cells in batch b / total cells."""
        return self._Pr_b.cpu().numpy()

    @property
    def theta(self):
        """Diversity penalty parameters (B,). Higher = more mixing encouraged."""
        return self._theta.cpu().numpy()

    @property
    def sigma(self):
        """Clustering bandwidth parameters (K,). Soft assignment kernel width."""
        return self._sigma.cpu().numpy()

    @property
    def lamb(self):
        """Ridge regression penalty ((B+1),). Regularization for batch correction."""
        return self._lamb.cpu().numpy()

    @property
    def objectives(self):
        """List of objective values for compatibility with evaluator."""
        return self.objective_harmony

    def result(self):
        """Return corrected data as NumPy array."""
        return self._Z_corr.cpu().numpy().T

    def allocate_buffers(self):
        self._scale_dist = torch.zeros((self.K, self.N), dtype=torch.float32, device=self.device)
        self._dist_mat = torch.zeros((self.K, self.N), dtype=torch.float32, device=self.device)
        self._O = torch.zeros((self.K, self.B), dtype=torch.float32, device=self.device)
        self._E = torch.zeros((self.K, self.B), dtype=torch.float32, device=self.device)
        self._W = torch.zeros((self.B + 1, self.d), dtype=torch.float32, device=self.device)
        self._R = torch.zeros((self.K, self.N), dtype=torch.float32, device=self.device)
        self._Y = torch.zeros((self.d, self.K), dtype=torch.float32, device=self.device)
        self._Phi_moe_T = self._Phi_moe.T

        # Precompute reused constants (Change 5)
        self._neg_inv_sigma = (-1.0 / self._sigma).unsqueeze(1)    # (K, 1) for broadcasting
        self._sigma_unsqueezed = self._sigma.unsqueeze(1)           # (K, 1)
        if not self.lambda_estimation:
            self._lamb_diag = torch.diag(self._lamb)                # (B+1, B+1)
        # Precompute Phi overlap pairs for multi-variable batch case
        self._phi_overlap_pairs = []
        if self.B > 1:
            phi_cross = self._Phi @ self._Phi.T
            for b1 in range(self.B):
                for b2 in range(b1 + 1, self.B):
                    if phi_cross[b1, b2] > 0:
                        mask = self._Phi[b1, :] * self._Phi[b2, :]
                        self._phi_overlap_pairs.append((b1, b2, mask))

    def init_cluster(self, random_state):
        # Proposal 1: Batch-mean-corrected KMeans initialization
        # Remove leading-order batch effect (mean shift) before clustering
        global_mean = self._Z_cos.mean(dim=1, keepdim=True)  # (d, 1)
        Z_init = self._Z_cos.clone()
        for b in range(self.B):
            idx = self._batch_index[b]
            if len(idx) > 0:
                batch_mean = self._Z_cos[:, idx].mean(dim=1, keepdim=True)  # (d, 1)
                Z_init[:, idx] -= (batch_mean - global_mean)
        Z_init = F.normalize(Z_init, p=2, dim=0)

        # KMeans needs CPU numpy array
        Z_init_np = Z_init.cpu().numpy()

        logger.info("Computing initial centroids with sklearn.KMeans (batch-mean-corrected)...")
        model = KMeans(n_clusters=self.K, init='k-means++',
                       n_init=1, max_iter=25, random_state=random_state)
        model.fit(Z_init_np.T)
        self._Y = torch.tensor(model.cluster_centers_.T, dtype=torch.float32, device=self.device)
        logger.info("KMeans initialization complete.")

        # Normalize centroids
        self._Y = F.normalize(self._Y, p=2, dim=0)

        # Compute distance matrix: dist = 2 * (1 - Y.T @ Z_cos)
        torch.mm(self._Y.T, self._Z_cos, out=self._dist_mat)
        self._dist_mat.mul_(-2).add_(2)

        # Compute R using F.softmax for numerical stability (Change 1)
        scaled = self._dist_mat * self._neg_inv_sigma
        self._R = F.softmax(scaled, dim=0)

        # Batch diversity statistics
        self._E = torch.outer(self._R.sum(dim=1), self._Pr_b)
        self._O = self._R @ self._Phi.T

        self.compute_objective()
        self.objective_harmony.append(self.objective_kmeans[-1])

    def compute_objective(self):
        # Normalization constant
        norm_const = 2000.0 / self.N

        # Change 6: Avoid K×N temporary with torch.dot on flattened views
        kmeans_error = torch.dot(self._R.reshape(-1), self._dist_mat.reshape(-1)).item()

        # Compute entropy more efficiently: reduce to (K,) then dot with sigma
        entropy_per_cluster = torch.sum(torch.xlogy(self._R, self._R), dim=1)  # (K,)
        _entropy = torch.dot(entropy_per_cluster, self._sigma).item()

        # Cross entropy using generalized KL divergence (matching the original Harmony paper)
        O_clamped = torch.clamp(self._O, min=1e-4)
        E_clamped = torch.clamp(self._E, min=1e-4)
        _cross_entropy = torch.sum(
            self._theta.unsqueeze(0) * (O_clamped * torch.log(O_clamped / E_clamped) - O_clamped + E_clamped)
        ).item()

        # Store with normalization constant
        self.objective_kmeans.append((kmeans_error + _entropy + _cross_entropy) * norm_const)
        self.objective_kmeans_dist.append(kmeans_error * norm_const)
        self.objective_kmeans_entropy.append(_entropy * norm_const)
        self.objective_kmeans_cross.append(_cross_entropy * norm_const)

    def harmonize(self, iter_harmony=10, verbose=True):
        converged = False
        for i in range(1, iter_harmony + 1):
            if verbose:
                logger.info(f"Iteration {i} of {iter_harmony}")

            self.cluster()
            self.moe_correct_ridge()

            converged = self.check_convergence(1)
            if converged:
                if verbose:
                    logger.info(f"Converged after {i} iteration{'s' if i > 1 else ''}")
                break

        if verbose and not converged:
            logger.info("Stopped before convergence")

    def cluster(self):
        rounds = 0
        for i in range(self.max_iter_kmeans):
            # Change 4: In-place Y update
            torch.mm(self._Z_cos, self._R.T, out=self._Y)
            # In-place normalize
            norms = self._Y.norm(dim=0, keepdim=True).clamp_(min=1e-12)
            self._Y.div_(norms)

            # In-place distance: dist = 2 - 2*(Y.T @ Z_cos)
            torch.mm(self._Y.T, self._Z_cos, out=self._dist_mat)
            self._dist_mat.mul_(-2).add_(2)

            # Update R
            self.update_R()

            # Compute objective and check convergence
            self.compute_objective()

            if i > self.window_size:
                if self.check_convergence(0):
                    rounds = i + 1
                    break
            rounds = i + 1

        self.kmeans_rounds.append(rounds)
        self.objective_harmony.append(self.objective_kmeans[-1])

    def update_R(self):
        # Change 1: Use F.softmax for numerical stability
        scaled = self._dist_mat * self._neg_inv_sigma
        self._scale_dist = F.softmax(scaled, dim=0)

        # Create shuffled update order
        update_order = torch.randperm(self.N, device=self.device)

        # Process in blocks
        n_blocks = int(np.ceil(1.0 / self.block_size))
        cells_per_block = int(self.N * self.block_size)

        # Permute matrices
        R_perm = self._R[:, update_order]
        scale_perm = self._scale_dist[:, update_order]
        Phi_perm = self._Phi[:, update_order]

        for blk in range(n_blocks):
            idx_min = blk * cells_per_block
            idx_max = self.N if blk == n_blocks - 1 else (blk + 1) * cells_per_block

            R_block = R_perm[:, idx_min:idx_max]
            scale_block = scale_perm[:, idx_min:idx_max]
            Phi_block = Phi_perm[:, idx_min:idx_max]

            # Remove cells from statistics
            self._E -= torch.outer(R_block.sum(dim=1), self._Pr_b)
            self._O -= R_block @ Phi_block.T

            # Bounded diversity penalty via clamped log-ratio (Proposal 2)
            O_clamped = torch.clamp(self._O, min=1.0)
            E_clamped = torch.clamp(self._E, min=1e-4)
            log_ratio = torch.log(E_clamped / O_clamped)
            log_ratio = log_ratio.clamp(-2.0, 2.0)
            ratio_powered = torch.exp(self._theta.unsqueeze(0) * log_ratio)  # (K, B)
            R_block_new = scale_block * (ratio_powered @ Phi_block)
            R_block_sum = R_block_new.sum(dim=0)
            R_block_sum = torch.clamp(R_block_sum, min=1e-8)
            R_block_new = R_block_new / R_block_sum

            # Put cells back
            self._E += torch.outer(R_block_new.sum(dim=1), self._Pr_b)
            self._O += R_block_new @ Phi_block.T

            R_perm[:, idx_min:idx_max] = R_block_new

        # Restore original order via scatter assignment (O(N) vs O(N log N) for argsort)
        self._R[:, update_order] = R_perm

    def check_convergence(self, i_type):
        if i_type == 0:
            if len(self.objective_kmeans) <= self.window_size + 1:
                return False

            w = self.window_size
            obj_old = sum(self.objective_kmeans[-w-1:-1])
            obj_new = sum(self.objective_kmeans[-w:])
            return abs(obj_old - obj_new) / abs(obj_old) < self.epsilon_kmeans

        if i_type == 1:
            if len(self.objective_harmony) < 2:
                return False

            obj_old = self.objective_harmony[-2]
            obj_new = self.objective_harmony[-1]
            return (obj_old - obj_new) / abs(obj_old) < self.epsilon_harmony

        return True

    def moe_correct_ridge(self):
        """Ridge regression correction for batch effects with per-PC adaptive weighting.
        
        Optimized: eliminates (K, B+1, N) tensor, uses Cholesky solve,
        and exploits one-hot structure of Phi for covariance and RHS computation.
        """
        K, N, B, d = self.K, self.N, self.B, self.d
        B1 = B + 1

        # Change 2: Compute covariance analytically from O and R_sums
        R_sums = self._R.sum(dim=1)  # (K,)

        cov_all = torch.zeros(K, B1, B1, dtype=torch.float32, device=self.device)
        cov_all[:, 0, 0] = R_sums
        cov_all[:, 0, 1:] = self._O          # O is (K, B)
        cov_all[:, 1:, 0] = self._O
        # Diagonal of B×B block: for one-hot Phi, Phi_b * R_k * Phi_b^T = O[k, b]
        for b in range(B):
            cov_all[:, b + 1, b + 1] = self._O[:, b]
        # Cross-terms for multi-variable batches
        for b1, b2, mask in self._phi_overlap_pairs:
            cross_val = self._R @ mask  # (K,)
            cov_all[:, b1 + 1, b2 + 1] = cross_val
            cov_all[:, b2 + 1, b1 + 1] = cross_val

        # Add regularization
        if not self.lambda_estimation:
            cov_all += self._lamb_diag.unsqueeze(0)
        else:
            for k in range(K):
                lamb_vec = find_lambda_torch(self.alpha, self._E[k, :], self.device)
                cov_all[k] += torch.diag(lamb_vec)

        # Change 2b: Compute RHS using _batch_index
        # rhs: (K, B+1, d)
        ZR_T = (self._Z_orig @ self._R.T).T  # (K, d) — intercept term
        rhs = torch.zeros(K, B1, d, dtype=torch.float32, device=self.device)
        rhs[:, 0, :] = ZR_T
        for b in range(B):
            idx = self._batch_index[b]
            rhs[:, b + 1, :] = (self._Z_orig[:, idx] @ self._R[:, idx].T).T

        # Change 3: Cholesky solve instead of explicit inverse
        # Add small jitter for robustness
        cov_all.diagonal(dim1=-2, dim2=-1).add_(1e-8)
        L = torch.linalg.cholesky(cov_all)         # (K, B+1, B+1)
        W_all = torch.cholesky_solve(rhs, L)        # (K, B+1, d)
        W_all[:, 0, :] = 0  # Don't remove intercept

        # Change 2c: Apply correction using _batch_index
        total_correction = torch.zeros(d, N, dtype=torch.float32, device=self.device)
        for b in range(B):
            idx = self._batch_index[b]
            # W_all[:, b+1, :] is (K, d), self._R[:, idx] is (K, N_b)
            # (d, K) @ (K, N_b) -> (d, N_b)
            total_correction[:, idx] = W_all[:, b + 1, :].T @ self._R[:, idx]

        # Cap correction magnitude to preserve biological signal
        correction_norms = total_correction.norm(dim=0, keepdim=True)
        orig_norms = self._Z_orig.norm(dim=0, keepdim=True)
        max_allowed = self.max_correction_ratio * orig_norms
        scale = torch.clamp(max_allowed / (correction_norms + 1e-8), max=1.0)
        total_correction = total_correction * scale

        # Apply momentum blending for stability
        if self.momentum > 0 and self._prev_correction is not None:
            total_correction = (1.0 - self.momentum) * total_correction + self.momentum * self._prev_correction
        self._prev_correction = total_correction.clone()

        self._Z_corr = self._Z_orig - total_correction

        # Update Z_cos
        self._Z_cos = F.normalize(self._Z_corr, p=2, dim=0)


def safe_entropy_torch(x):
    """Compute x * log(x), returning 0 where x is 0 or negative."""
    return torch.xlogy(x, x)


def harmony_pow_torch(A, T):
    """Element-wise power with different exponents per column."""
    result = torch.empty_like(A)
    for c in range(A.shape[1]):
        result[:, c] = torch.pow(A[:, c], T[c])
    return result


def find_lambda_torch(alpha, cluster_E, device):
    """Compute dynamic lambda based on cluster expected counts."""
    lamb = torch.zeros(len(cluster_E) + 1, dtype=torch.float32, device=device)
    lamb[1:] = cluster_E * alpha
    return lamb
