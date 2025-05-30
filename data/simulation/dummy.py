import numpy as np
from scipy.special import expit as sigmoid
import igraph as ig
import random
import pandas as pd
from typing import Dict, List, Tuple, Callable, Union
import os
import json
from datetime import datetime
import networkx as nx
import numpy as np

# References:
# https://github.com/xunzheng/notears/blob/master/notears/utils.py

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()

def simulate_dag(d, s0, graph_type):
    """Simulate random DAG with some expected number of edges.

    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF, BP

    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """
    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError('unknown graph type')
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm

def simulate_parameter(B, w_ranges=((-2.0, -0.5), (0.5, 2.0))):
    """Simulate SEM parameters for a DAG.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        w_ranges (tuple): disjoint weight ranges

    Returns:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
    """
    W = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)  # which range
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U
    return W

def simulate_linear_sem(W, n, sem_type, noise_scale=None, discrete_ratio=0.0, max_categories=10):
    """Simulate samples from linear SEM with specified type of noise and mixed continuous/discrete variables.

    For uniform, noise z ~ uniform(-a, a), where a = noise_scale.

    Args:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
        n (int): num of samples, n=inf mimics population risk
        sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones
        discrete_ratio (float): ratio of discrete columns, default 0.3
        max_categories (int): number of categories for discrete columns, default 10

    Returns:
        X (np.ndarray): [n, d] sample matrix with mixed continuous and discrete variables
    """
    def _simulate_single_equation(X, w, scale, is_discrete=False, n_cats=None):
        """X: [n, num of parents], w: [num of parents], x: [n]"""
        if is_discrete:
            # For discrete variables, generate multi-dimensional output first
            if X.shape[1] == 0:
                # No parents - generate random logits
                logits = np.random.normal(scale=scale, size=(n, n_cats))
            else:
                # Generate logits based on parents, the noise is added via sampling
                logits = np.zeros((n, n_cats))
                for k in range(n_cats):
                    if sem_type == 'gaussian':
                        logits[:, k] = X @ w
                    elif sem_type == 'exponential':
                        logits[:, k] = X @ w
                    elif sem_type == 'gumbel':
                        logits[:, k] = X @ w
                    elif sem_type == 'uniform':
                        logits[:, k] = X @ w
                    else:
                        raise ValueError('unsupported sem type for discrete variables')

            # Apply softmax to get probabilities
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
            # Sample from categorical distribution
            return np.array([np.random.choice(n_cats, p=p) for p in probs]).astype(float)

        else:
            if sem_type == 'gaussian':
                z = np.random.normal(scale=scale, size=n)
                x = X @ w + z
            elif sem_type == 'exponential':
                z = np.random.exponential(scale=scale, size=n)
                x = X @ w + z
            elif sem_type == 'gumbel':
                z = np.random.gumbel(scale=scale, size=n)
                x = X @ w + z
            elif sem_type == 'uniform':
                z = np.random.uniform(low=-scale, high=scale, size=n)
                x = X @ w + z
            elif sem_type == 'logistic':
                x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
            elif sem_type == 'poisson':
                x = np.random.poisson(np.exp(X @ w)) * 1.0
            else:
                raise ValueError('unknown sem type')
            return x

    d = W.shape[0]
    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    else:
        if len(noise_scale) != d:
            raise ValueError('noise scale must be a scalar or has length d')
        scale_vec = noise_scale
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    if np.isinf(n):  # population risk for linear gauss SEM
        if sem_type == 'gauss':
            # make 1/d X'X = true cov
            X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
            return X
        else:
            raise ValueError('population risk not available')

    # Sample discrete columns based on ratio
    n_discrete = int(d * discrete_ratio)
    discrete_columns = np.random.choice(d, size=n_discrete, replace=False)

    # empirical risk
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X = np.zeros([n, d])
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        is_discrete = j in discrete_columns
        n_categories = np.random.randint(2, max_categories)
        X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j], is_discrete, n_categories)
    return X

def simulate_nonlinear_sem(B, n, sem_type, noise_scale=None, discrete_ratio=0.3, max_categories=10):
    """Simulate samples from nonlinear SEM with mixed continuous and discrete variables.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        n (int): num of samples
        sem_type (str): mlp, mim, gp, gp-add
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones
        discrete_ratio (float): ratio of discrete columns, default 0.3
        max_categories (int): number of categories for discrete columns, default 10

    Returns:
        X (np.ndarray): [n, d] sample matrix with mixed continuous and discrete variables
    """
    def _simulate_single_equation(X, scale, is_discrete=False, n_cats=None):
        """X: [n, num of parents], x: [n]"""
        pa_size = X.shape[1]
        
        if is_discrete:
            # For discrete variables, generate multi-dimensional output first
            if pa_size == 0:
                # No parents - generate random logits
                logits = np.random.normal(scale=scale, size=(n, n_cats))
            else:
                if sem_type == 'mlp':
                    hidden = 100
                    # Generate separate MLPs for each category
                    logits = np.zeros((n, n_cats))
                    for k in range(n_cats):
                        W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
                        W1[np.random.rand(*W1.shape) < 0.5] *= -1
                        W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
                        W2[np.random.rand(hidden) < 0.5] *= -1
                        logits[:, k] = sigmoid(X @ W1) @ W2
                elif sem_type == 'mim':
                    logits = np.zeros((n, n_cats))
                    for k in range(n_cats):
                        w1 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
                        w1[np.random.rand(pa_size) < 0.5] *= -1
                        w2 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
                        w2[np.random.rand(pa_size) < 0.5] *= -1
                        w3 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
                        w3[np.random.rand(pa_size) < 0.5] *= -1
                        logits[:, k] = np.tanh(X @ w1) + np.cos(X @ w2) + np.sin(X @ w3)
                elif sem_type == 'gp' or sem_type == 'gp-add':
                    from sklearn.gaussian_process import GaussianProcessRegressor
                    logits = np.zeros((n, n_cats))
                    for k in range(n_cats):
                        gp = GaussianProcessRegressor()
                        if sem_type == 'gp':
                            logits[:, k] = gp.sample_y(X, random_state=None).flatten()
                        else:  # gp-add
                            logits[:, k] = sum([gp.sample_y(X[:, i, None], random_state=None).flatten() 
                                              for i in range(X.shape[1])])
                else:
                    raise ValueError('unknown sem type')
            
            # Apply softmax to get probabilities
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
            # Sample from categorical distribution
            return np.array([np.random.choice(n_cats, p=p) for p in probs]).astype(float)
            
        else:
            # Original continuous variable generation
            z = np.random.normal(scale=scale, size=n)
            if pa_size == 0:
                return z
                
            if sem_type == 'mlp':
                hidden = 100
                W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
                W1[np.random.rand(*W1.shape) < 0.5] *= -1
                W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
                W2[np.random.rand(hidden) < 0.5] *= -1
                x = sigmoid(X @ W1) @ W2 + z
            elif sem_type == 'mim':
                w1 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
                w1[np.random.rand(pa_size) < 0.5] *= -1
                w2 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
                w2[np.random.rand(pa_size) < 0.5] *= -1
                w3 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
                w3[np.random.rand(pa_size) < 0.5] *= -1
                x = np.tanh(X @ w1) + np.cos(X @ w2) + np.sin(X @ w3) + z
            elif sem_type == 'gp':
                from sklearn.gaussian_process import GaussianProcessRegressor
                gp = GaussianProcessRegressor()
                x = gp.sample_y(X, random_state=None).flatten() + z
            elif sem_type == 'gp-add':
                from sklearn.gaussian_process import GaussianProcessRegressor
                gp = GaussianProcessRegressor()
                x = sum([gp.sample_y(X[:, i, None], random_state=None).flatten()
                         for i in range(X.shape[1])]) + z
            else:
                raise ValueError('unknown sem type')
            return x

    d = B.shape[0]
    scale_vec = noise_scale * np.ones(d) if noise_scale else np.ones(d)
    X = np.zeros([n, d])
    G = ig.Graph.Adjacency(B.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    
    # Sample discrete columns based on ratio
    n_discrete = int(d * discrete_ratio)
    discrete_columns = np.random.choice(d, size=n_discrete, replace=False)
    
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        is_discrete = j in discrete_columns
        n_categories = np.random.randint(2, max_categories)
        X[:, j] = _simulate_single_equation(X[:, parents], scale_vec[j], is_discrete, n_categories)
    return X

def count_accuracy(B_true, B_est):
    """Compute various accuracy metrics for B_est.

    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition

    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    if (B_est == -1).any():  # cpdag
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError('B_est should take value in {0,1,-1}')
        if ((B_est == -1) & (B_est.T == -1)).any():
            raise ValueError('undirected edge should only appear once')
    else:  # dag
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError('B_est should take value in {0,1}')
        if not is_dag(B_est):
            raise ValueError('B_est should be a DAG')
    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size}

class DataSimulator:
    def __init__(self):
        self.data = None
        self.graph = None
        self.ground_truth = {}
        self.variable_names = None

    def generate_graph(self, n_nodes: int, edge_probability: float = 0.3, variable_names: List[str] = None, graph_type: str = 'ER') -> None:
        """Generate a random directed acyclic graph (DAG) using specified method."""
        self.graph = simulate_dag(n_nodes, int(edge_probability * n_nodes * (n_nodes - 1) / 2), graph_type)
        if variable_names and len(variable_names) == n_nodes:
            self.variable_names = variable_names
            self.graph_dict = {i: name for i, name in enumerate(variable_names)}
        else:
            self.variable_names = [f'X{i+1}' for i in range(n_nodes)]
            self.graph_dict = {i: f'X{i+1}' for i in range(n_nodes)}
        self.ground_truth['graph'] = self.graph_dict
        self.ground_truth['edge_probability'] = edge_probability

    def generate_single_domain_data(self, n_samples: int, noise_scale: float, noise_type: str, 
                                    function_type: Union[str, List[str], Dict[str, str]], 
                                    discrete_ratio: float = 0.0, max_categories: int = 5) -> pd.DataFrame:
        """Generate data for a single domain based on the graph structure."""
        # assert if function_type, noise_type, noise_scale are valid
        print(f"function_type: {function_type}, noise_type: {noise_type}, noise_scale: {noise_scale}")
        assert isinstance(function_type, str) and function_type in ['linear', 'mlp', 'mim', 'gp', 'gp-add']
        assert noise_type in ['gaussian', 'exponential', 'gumbel', 'uniform', 'logistic', 'poisson']
        # if function_type is not linear, then the noise_type must be gaussian
        if function_type != 'linear':
            print(f"When function_type is not linear, noise_type is set to gaussian")
        assert isinstance(noise_scale, float) and noise_scale > 0
        if function_type == 'linear':
            W = simulate_parameter(self.graph)
            data = simulate_linear_sem(W, n_samples, noise_type, noise_scale, discrete_ratio, max_categories)
        else:
            data = simulate_nonlinear_sem(self.graph, n_samples, function_type, noise_scale, discrete_ratio, max_categories)
        
        data_df = pd.DataFrame(data, columns=self.variable_names)
        return data_df
    
    def generate_multi_domain_data(self, n_samples: int, noise_scale: float, noise_type: str, 
                                   function_type: Union[str, List[str], Dict[str, str]], 
                                   discrete_ratio: float = 0.0, max_categories: int = 5, edge_probability: float = 0.3) -> pd.DataFrame:
        """Generate data for a single domain based on the graph structure."""
        # assert if function_type, noise_type, noise_scale are valid
        print(f"function_type: {function_type}, noise_type: {noise_type}, noise_scale: {noise_scale}")
        assert isinstance(function_type, str) and function_type in ['linear', 'mlp', 'mim', 'gp', 'gp-add']
        assert noise_type in ['gaussian', 'exponential', 'gumbel', 'uniform', 'logistic', 'poisson']
        # if function_type is not linear, then the noise_type must be gaussian
        if function_type != 'linear':
            print(f"When function_type is not linear, noise_type is set to gaussian")
        assert isinstance(noise_scale, float) and noise_scale > 0

        # Generate base data for all domains
        if function_type == 'linear':
            W = simulate_parameter(self.graph)
            base_data = []
            for i in range(self.n_domains):
                base_data.append(simulate_linear_sem(W, n_samples, noise_type, noise_scale, discrete_ratio, max_categories))
        else:
            base_data = []
            for i in range(self.n_domains):
                base_data.append(simulate_nonlinear_sem(self.graph, n_samples, function_type, noise_scale, discrete_ratio, max_categories))
        # Find nodes that don't have many connections to simulate confounding with domain
        n_nodes = self.graph.shape[0]
        # Get nodes with fewer connections (potential targets for domain confounding)
        node_connections = np.sum(self.graph, axis=1) + np.sum(self.graph, axis=0)
        # Sort nodes by connection count (ascending)
        less_connected_nodes = np.argsort(node_connections)
        # Select a subset of less connected nodes to be affected by domain
        n_affected = max(3, int(edge_probability * n_nodes))
        affected_nodes = less_connected_nodes[:n_affected]
        
        print(f"Domain index affects variables: {[self.variable_names[i] for i in affected_nodes]}")
        
        # Generate domain-specific effects
        data = []
        N = 10
        step = 10 / self.n_domains
        
        # Calculate correlation matrix for base data before adding domain effects
        base_data_array = np.vstack([d for d in base_data])
        base_correlation_matrix = np.corrcoef(base_data_array.T)
        print(f"Base correlation matrix shape (before domain effects): {base_correlation_matrix.shape}")
        
        # Find highly correlated pairs in base data
        n_vars = base_correlation_matrix.shape[0]
        high_corr_threshold = 0.7
        high_corr_pairs_base = []
        
        for i in range(n_vars):
            for j in range(i+1, n_vars):
                corr = base_correlation_matrix[i, j]
                if abs(corr) > high_corr_threshold:
                    high_corr_pairs_base.append((self.variable_names[i], self.variable_names[j], corr))
        
        if high_corr_pairs_base:
            print(f"Before domain effects - Found {len(high_corr_pairs_base)} highly correlated pairs (|r| > {high_corr_threshold}):")
            for var1, var2, corr in high_corr_pairs_base:
                print(f"  {var1} and {var2}: r = {corr:.4f}")
        else:
            print(f"Before domain effects - No variable pairs with correlation above {high_corr_threshold} threshold")
        
        # Iteratively adjust domain effects until we find new highly correlated pairs
        domain_effect_multiplier = 0.3
        max_attempts = 5
        attempts = 0
        new_high_corr_pairs_found = False
        
        while attempts < max_attempts and not new_high_corr_pairs_found:
            # Try with current domain effect multiplier
            temp_data = []
            for i in range(self.n_domains):
                domain_data = base_data[i].copy()
                # Domain effect increases with domain index to create clear domain separation
                
                if function_type == 'linear':
                    # Linear domain effect
                    domain_effect = (i + 1) * step * domain_effect_multiplier
                    for node_idx in affected_nodes:
                        domain_data[:, node_idx] += domain_effect
                else:
                    # Nonlinear domain effect
                    domain_effect = (i + 1) * step * domain_effect_multiplier
                    for node_idx in affected_nodes:
                        # Quadratic effect: domain_effect * x^2
                        base_values = domain_data[:, node_idx]
                        domain_data[:, node_idx] += domain_effect * np.sign(base_values) * (base_values ** 2)
                
                temp_data.extend(domain_data)
            
            # Calculate correlation matrix after adding domain effects
            temp_data_array = np.vstack(temp_data)
            temp_correlation_matrix = np.corrcoef(temp_data_array.T)
            
            # Find highly correlated pairs after domain effects
            high_corr_pairs_temp = []
            for i in range(n_vars):
                for j in range(i+1, n_vars):
                    corr = temp_correlation_matrix[i, j]
                    if abs(corr) > high_corr_threshold:
                        high_corr_pairs_temp.append((self.variable_names[i], self.variable_names[j], corr))
            
            # Check if we found new highly correlated pairs
            if len(high_corr_pairs_temp) > len(high_corr_pairs_base):
                new_high_corr_pairs_found = True
                data = temp_data
                correlation_matrix = temp_correlation_matrix
                high_corr_pairs = high_corr_pairs_temp
            else:
                # Increase domain effect multiplier and try again
                domain_effect_multiplier *= 1.5
                attempts += 1
        
        # If we couldn't find new correlations, use the last attempt
        if not new_high_corr_pairs_found:
            data = temp_data
            correlation_matrix = temp_correlation_matrix
            high_corr_pairs = high_corr_pairs_temp
        
        # Print correlation information after domain effects
        print(f"Correlation matrix shape (after domain effects): {correlation_matrix.shape}")
        
        if high_corr_pairs:
            print(f"After domain effects - Found {len(high_corr_pairs)} highly correlated pairs (|r| > {high_corr_threshold}):")
            for var1, var2, corr in high_corr_pairs:
                print(f"  {var1} and {var2}: r = {corr:.4f}")
        else:
            print(f"After domain effects - No variable pairs with correlation above {high_corr_threshold} threshold")
        
        # Create the final dataframe with domain index
        data_df = pd.DataFrame(data, columns=self.variable_names)
        data_df['domain_index'] = np.repeat(range(1, 1 + self.n_domains), n_samples)
        return data_df


    def generate_data(self, n_samples: int, noise_scale: float = 1.0, 
                      noise_type: str = 'gaussian', 
                      function_type: Union[str, List[str], Dict[str, str]] = 'linear',
                      n_domains: int = 1, variable_names: List[str] = None,
                      discrete_ratio: float = 0.0, max_categories: int = 5, edge_probability: float = 0.3) -> None:
        """Generate heterogeneous data from multiple domains."""
        if self.graph is None:
            raise ValueError("Generate graph first")

        domain_size = n_samples // n_domains

        self.n_domains = n_domains

        if n_domains == 1:
            domain_df = self.generate_single_domain_data(domain_size, noise_scale, noise_type, function_type, 
                                                          discrete_ratio, max_categories)
        else:
            domain_df = self.generate_multi_domain_data(domain_size, noise_scale, noise_type, function_type, 
                                                       discrete_ratio, max_categories, edge_probability)
    
        self.data = domain_df
        # shuffle the self.data
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        if variable_names is not None:
            if n_domains == 1:
                self.data.columns = variable_names
            else:
                self.data.columns = variable_names + ['domain_index']
            

        self.ground_truth['n_domains'] = n_domains
        self.ground_truth['noise_type'] = noise_type
        self.ground_truth['function_type'] = function_type
        self.ground_truth['discrete_ratio'] = discrete_ratio
        self.ground_truth['max_categories'] = max_categories
 

    def add_measurement_error(self, error_std: float = 0.3, error_rate: float = 0.5) -> None:
        """Randomly sample a subset of columns to add gaussian measurement error."""
        if self.data is None:
            raise ValueError("Generate data first")

        available_cols = [col for col in self.data.columns if col != 'domain_index']
        n_cols = int(error_rate * len(available_cols))
        columns = np.random.choice(available_cols, size=n_cols, replace=False)

        for col in columns:
            if pd.api.types.is_numeric_dtype(self.data[col]):
                self.data[col] += np.random.randn(len(self.data)) * error_std
        
        self.ground_truth['measurement_error'] = {col: error_std for col in columns}
        self.ground_truth['measurement_error_value'] = error_rate
        self.ground_truth['measurement_error_desc'] = f"The measurement error is Gaussian with a standard deviation of {error_std} on {error_rate} of the columns (some are selected, some are not, except domain_index), and the measurement error is added to the original data."

    def add_missing_values(self, missing_rate: float = 0.1) -> None:
        """Introduce missing values to the whole dataframe with a specified missing rate."""
        if self.data is None:
            raise ValueError("Generate data first")

        # Get all columns except domain_index
        columns = [col for col in self.data.columns if col != 'domain_index']
        
        # Create missing value mask for the whole dataframe
        mask = np.random.random(size=self.data[columns].shape) < missing_rate
        
        # Apply mask to create NaN missing values
        self.data[columns] = np.where(mask, np.nan, self.data[columns])
        
        # Record which columns were affected
        affected_columns = [col for col in columns if mask[:, columns.index(col)].any()]
        self.ground_truth['missing_rate'] = {col: missing_rate for col in affected_columns}
        self.ground_truth['missing_rate_value'] = missing_rate
        self.ground_truth['missing_data_desc'] = f"The missing values are randomly sampled with a missing rate of {missing_rate} on all column data (except domain_index), and the missing values are replaced with NaN."

    def generate_dataset(self, n_nodes: int, n_samples: int, edge_probability: float = 0.3,
                         noise_scale: float = 1.0, noise_type: str = 'gaussian',
                         function_type: Union[str, List[str], Dict[str, str]] = 'random', discrete_ratio: float = 0.0, max_categories: int = 5,
                         add_measurement_error: bool = False, add_missing_values: bool = False, n_domains: int = 1, 
                         error_std: float = 0.3, error_rate: float = 0.5, missing_rate: float = 0.1,
                         variable_names: List[str] = None, graph_type: str = 'ER') -> Tuple[Dict[int, str], pd.DataFrame]:
        """
        Generate a complete heterogeneous dataset with various characteristics.
        """
        self.generate_graph(n_nodes, edge_probability, variable_names, graph_type)
        self.generate_data(n_samples, noise_scale, noise_type, function_type, n_domains, variable_names, discrete_ratio, max_categories, edge_probability)
                    
        if add_measurement_error:
            self.add_measurement_error(error_std=error_std, error_rate=error_rate)
        else:
            self.ground_truth['measurement_error'] = None
            self.ground_truth['measurement_error_value'] = None
            self.ground_truth['measurement_error_desc'] = None
        
        if add_missing_values:
            self.add_missing_values(missing_rate=missing_rate)
        else:
            self.ground_truth['missing_rate'] = None
            self.ground_truth['missing_rate_value'] = None
            self.ground_truth['missing_data_desc'] = None
        
        # original (i, j) == 1 (i -> j), here we return the transpose of the graph to be (i, j) == 1 -> (j -> i)
        return self.graph.T, self.data

    def save_simulation(self, output_dir: str = 'simulated_data', prefix: str = 'base') -> None:
        """
        Save the simulated data, graph structure, and simulation settings.
        
        :param output_dir: The directory to save the files
        :param prefix: A prefix for the filenames
        """
        if self.data is None or self.graph is None:
            raise ValueError("No data or graph to save. Generate dataset first.")

        # Create a timestamped folder with specific settings
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        n_nodes = len(self.graph)
        n_samples = len(self.data)
        folder_name = f"{timestamp}_{prefix}_nodes{n_nodes}_samples{n_samples}"
        save_dir = os.path.join(output_dir, folder_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the data as CSV
        data_filename = os.path.join(save_dir, f'{prefix}_data.csv')
        data_to_save = self.data.copy()
        if self.ground_truth.get('n_domains', 1) == 1 and 'domain_index' in data_to_save.columns:
            data_to_save = data_to_save.drop('domain_index', axis=1)
        data_to_save.to_csv(data_filename, index=False)
        print(f"Data saved to {data_filename}")
        
        # Save the graph structure as NPY
        graph_filename = os.path.join(save_dir, f'{prefix}_graph.npy')
        # original (i, j) == 1 (i -> j), here we return the transpose of the graph to be (i, j) == 1 -> (j -> i)
        np.save(graph_filename, self.graph.T)
        print(f"Graph structure saved to {graph_filename}")
        
        # Save the simulation settings as JSON
        config_filename = os.path.join(save_dir, f'{prefix}_config.json')
        config = {
            'n_nodes': n_nodes,
            'n_samples': n_samples,
            'n_domains': self.ground_truth.get('n_domains'),
            'noise_type': self.ground_truth.get('noise_type'),
            'function_type': self.ground_truth.get('function_type'),
            'node_functions': self.ground_truth.get('node_functions'),
            'categorical': self.ground_truth.get('categorical'),
            'measurement_error': self.ground_truth.get('measurement_error'),
            'selection_bias': self.ground_truth.get('selection_bias'),
            'confounding': self.ground_truth.get('confounding'),
            'missing_rate': self.ground_truth.get('missing_rate'),
            'edge_probability': self.ground_truth.get('edge_probability'),
            'discrete_ratio': self.ground_truth.get('discrete_ratio'),
            'max_categories': self.ground_truth.get('max_categories'),
            'missing_rate_value': self.ground_truth.get('missing_rate_value'),
            'measurement_error_value': self.ground_truth.get('measurement_error_value'),
            'missing_data_desc': self.ground_truth.get('missing_data_desc'),
            'measurement_error_desc': self.ground_truth.get('measurement_error_desc'),
        }
        with open(config_filename, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        print(f"Simulation settings saved to {config_filename}")

    def generate_and_save_dataset(self, n_nodes: int, n_samples: int, output_dir: str = 'simulated_data', prefix: str = 'base', **kwargs) -> None:
        """
        Generate a dataset and save the results.
        
        :param n_nodes: Number of nodes in the graph
        :param n_samples: Number of samples in the dataset
        :param output_dir: The directory to save the files
        :param prefix: A prefix for the filenames
        :param kwargs: Additional arguments for the generate_dataset method
        """
        self.generate_dataset(n_nodes, n_samples, **kwargs)
        self.save_simulation(output_dir, prefix)

def test_mixed_data_generation():
    """
    Test case that generates a dataset with mixed data types and various data quality issues:
    - Linear Gaussian base relationships
    - Mix of continuous and discrete variables
    - Missing values
    - Measurement/observation errors
    """
    simulator = DataSimulator()
    
    # Generate complex mixed dataset
    graph, data = simulator.generate_dataset(
        # Basic parameters
        n_nodes=5,
        n_samples=500,
        edge_probability=0.3,
        
        # Linear Gaussian base
        function_type='linear',
        noise_type='gaussian',
        
        # Add discrete variables
        discrete_ratio=0.3,  # 30% of variables will be discrete
        max_categories=5,
        
        # Add data quality issues
        add_missing_values=True,
        add_measurement_error=True,
    )
    
    print("Test dataset generated with mixed data specifications")
    return graph, data


class TimeSeriesSimulator:
    """
    A class for generating time series data with causal relationships.
    Uses CausalNex's data generators to create stationary dynamic networks.
    """
    
    def __init__(self):
        """Initialize the TimeSeriesSimulator."""
        self.graph = None
        self.data = None
        self.ground_truth = {
            'summary_adjacency': None,
            'lagged_adjacency': None,
            'config': None
        }
        
    def generate_time_series_data(self, 
                                  num_nodes: int = 10, 
                                  lag: int = 3, 
                                  degree_inter: float = 3.0, 
                                  degree_intra: float = 2.0,
                                  sample_size: int = 2000, 
                                  noise_type: str = 'linear-gauss',
                                  seed: int = None) -> pd.DataFrame:
        """
        Generate time series data with causal relationships.
        
        Args:
            num_nodes: Number of variables in the time series
            lag: Maximum time lag for causal relationships
            degree_inter: Average number of inter-time-slice edges per node
            degree_intra: Average number of intra-time-slice edges per node
            sample_size: Number of time points to generate
            noise_type: Type of noise to use ('linear-gauss', 'linear-exp', 'linear-gumbel')
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame containing the generated time series data
        """
        from causalnex.structure.data_generators import gen_stationary_dyn_net_and_df
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Calculate weight parameters based on network size
        if degree_inter == 0:
            degree_inter = 1.0
            
        if degree_intra:
            w_min_intra = round(1/degree_intra, 1)
            w_max_intra = round(1/degree_intra, 1)
        else:
            w_min_intra = 0.4
            w_max_intra = 0.4
            
        w_min_inter = min(round(1/degree_inter, 1), 
                          round(1/np.sqrt(num_nodes), 1), 
                          round(1/np.sqrt(num_nodes * lag), 2))
        w_max_inter = min(round(1/degree_inter, 1), 
                          round(1/np.sqrt(num_nodes), 1), 
                          round(1/np.sqrt(num_nodes * lag), 2))
        
        # Generate the network and data
        graph_net, df, intra_nodes, inter_nodes = gen_stationary_dyn_net_and_df(
            num_nodes=num_nodes,
            n_samples=sample_size,
            p=lag,
            degree_intra=degree_intra,
            degree_inter=degree_inter,
            w_min_intra=w_min_intra,
            w_max_intra=w_max_intra,
            w_min_inter=w_min_inter,
            w_max_inter=w_max_inter,
            w_decay=1.0,
            sem_type=noise_type
        )
        
        # Extract only the intra-time nodes (current time slice)
        df = df[intra_nodes]
        
        # Simplify column names
        df.columns = [el.split('_')[0] for el in df.columns]
        
        # Extract the ground truth graph
        graph_true = self._get_graph(graph_net, intra_nodes)
        summary_adj, lagged_adj = self._dict_to_adjacency_matrix(graph_true, num_nodes, lag)
        
        # Store results
        function_type, noise_type = noise_type.split('-')
        if noise_type == 'gauss':
            noise_type = 'gaussian'
        elif noise_type == 'exp':
            noise_type = 'exponential'
        elif noise_type == 'gumbel':
            noise_type = 'gumbel'
            
        self.graph = graph_net
        self.data = df
        self.ground_truth = {}
        self.ground_truth['graph'] = graph_true
        self.ground_truth['summary_adjacency'] = summary_adj
        self.ground_truth['lagged_adjacency'] = lagged_adj
        self.ground_truth['config'] = {
            "n_nodes": num_nodes,
            "lag": lag,
            "degree_inter": degree_inter,
            "degree_intra": degree_intra,
            "w_min_intra": w_min_intra,
            "w_max_intra": w_max_intra,
            "w_min_inter": w_min_inter,
            "w_max_inter": w_max_inter,
            "noise_type": noise_type,
            "function_type": function_type,
            "n_samples": sample_size,
            "seed": seed
        }
        return df
    
    def _get_graph(self, sm, data):
        """
        Extract the graph structure from the structural model.
        
        Args:
            sm: Structural model from CausalNex
            data: Data nodes
            
        Returns:
            Dictionary representing the graph structure
        """
        graph_dict = dict()
        for name in data:
            node = int(name.split('_')[0])
            graph_dict[node] = []

        for c, e in sm.edges:
            node_e = int(e.split('_')[0])
            node_c = int(c.split('_')[0])
            tc = int(c.partition("lag")[2])
            te = int(e.partition("lag")[2])
            lag = -1 * tc
            graph_dict[node_e].append((node_c, lag))
                
        return graph_dict
    
    def _dict_to_adjacency_matrix(self, result_dict, num_nodes, lookback_period):
        """
        Convert the graph dictionary to adjacency matrices.
        
        Args:
            result_dict: Dictionary representing the graph structure
            num_nodes: Number of nodes in the graph
            lookback_period: Maximum time lag
            
        Returns:
            Tuple of (summary adjacency matrix, lagged adjacency matrix)
        """
        adj_matrix = np.zeros((lookback_period, num_nodes, num_nodes))
        lagged_adj_matrix = np.zeros((lookback_period + 1, num_nodes, num_nodes))
        
        nodes = list(result_dict.keys())
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}

        for target, causes in result_dict.items():
            target_idx = node_to_idx[target]
            for cause, lag in causes:
                cause_idx = node_to_idx[cause]
                lag_index = -lag 
                lagged_adj_matrix[lag_index, target_idx, cause_idx] = 1
                    
        summary_adj_matrix = np.any(lagged_adj_matrix, axis=0).astype(int)

        return summary_adj_matrix, lagged_adj_matrix
    
    def save_simulation(self, output_dir: str = 'simulated_time_series', prefix: str = 'ts'):
        """
        Save the generated time series data and ground truth to files.
        
        Args:
            output_dir: Directory to save the files
            prefix: Prefix for the filenames
        """
        if self.data is None:
            raise ValueError("Generate data first")
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save the data
        data_filename = os.path.join(output_dir, f"{prefix}_{timestamp}_data.csv")
        self.data.to_csv(data_filename, index=False)
        
        # Save the summary adjacency matrix
        summary_filename = os.path.join(output_dir, f"{prefix}_{timestamp}_graph.npy")
        np.save(summary_filename, self.ground_truth['summary_adjacency'])
        
        # Save the lagged adjacency matrix
        lagged_filename = os.path.join(output_dir, f"{prefix}_{timestamp}_lagged.npy")
        np.save(lagged_filename, self.ground_truth['lagged_adjacency'])
        
        # Save the configuration
        config_filename = os.path.join(output_dir, f"{prefix}_{timestamp}_config.json")
        with open(config_filename, 'w') as f:
            json.dump(self.ground_truth['config'], f, indent=2, default=str)
            
        print(f"Time series data saved to {data_filename}")
        print(f"Summary adjacency matrix saved to {summary_filename}")
        print(f"Lagged adjacency matrix saved to {lagged_filename}")
        print(f"Configuration saved to {config_filename}")
    
    def generate_and_save_time_series(self, 
                                     num_nodes: int = 10, 
                                     lag: int = 3, 
                                     degree_inter: float = 3.0, 
                                     degree_intra: float = 2.0,
                                     sample_size: int = 2000, 
                                     noise_type: str = 'linear-gauss',
                                     output_dir: str = 'simulated_time_series', 
                                     prefix: str = 'ts',
                                     seed: int = None) -> None:
        """
        Generate time series data and save the results.
        
        Args:
            num_nodes: Number of variables in the time series
            lag: Maximum time lag for causal relationships
            degree_inter: Average number of inter-time-slice edges per node
            degree_intra: Average number of intra-time-slice edges per node
            sample_size: Number of time points to generate
            noise_type: Type of noise to use ('linear-gauss', 'linear-exp', 'linear-gumbel')
            output_dir: Directory to save the files
            prefix: Prefix for the filenames
            seed: Random seed for reproducibility
        """
        self.generate_time_series_data(
            num_nodes=num_nodes,
            lag=lag,
            degree_inter=degree_inter,
            degree_intra=degree_intra,
            sample_size=sample_size,
            noise_type=noise_type,
            seed=seed
        )
        self.save_simulation(output_dir, prefix)


def test_time_series_generation():
    """
    Test case that generates a time series dataset with various configurations.
    """
    ts_simulator = TimeSeriesSimulator()
    
    # Generate a simple time series dataset
    data = ts_simulator.generate_time_series_data(
        num_nodes=5,
        lag=3,
        degree_inter=2.0,
        degree_intra=1.0,
        sample_size=500,
        noise_type='linear-gauss',
        seed=42
    )
    
    print(f"Generated time series data with shape: {data.shape}")
    print(f"Summary adjacency matrix shape: {ts_simulator.ground_truth['summary_adjacency'].shape}")
    print(f"Lagged adjacency matrix shape: {ts_simulator.ground_truth['lagged_adjacency'].shape}")
    
    return ts_simulator.graph, data



# Generate pure simulated data using base simulator
# base_simulator = DataSimulator()

# 1. Linear Gaussian data (simple)
# base_simulator.generate_and_save_dataset(function_type='linear', n_nodes=5, n_samples=1000, edge_probability=0.3)

# 2. Non-linear Gaussian data
# base_simulator.generate_and_save_dataset(function_type='polynomial', n_nodes=10, n_samples=2000, edge_probability=0.4)

# 3. Linear Non-Gaussian data
# base_simulator.generate_and_save_dataset(function_type='linear', noise_type='uniform', n_nodes=15, n_samples=3000, edge_probability=0.3)

# 4. Discrete data
# base_simulator.generate_and_save_dataset(function_type='neural_network', n_nodes=8, n_samples=1500, edge_probability=0.35, 
#                                          add_categorical=True)

# 5. Mixed data (complex)
# base_simulator.generate_and_save_dataset(function_type=['linear', 'polynomial', 'neural_network'], n_nodes=20, n_samples=5000, edge_probability=0.25)

# 6. Heterogeneous data
# base_simulator.generate_and_save_dataset(function_type='linear', n_nodes=10, n_samples=1000, edge_probability=0.3, n_domains=5)




if __name__ == "__main__":
    # import numpy as np
    # import pandas as pd
    # import sys
    # sys.path.insert(0, 'causal-learn')
    # from causallearn.search.ConstraintBased.PC import pc
    # from causallearn.search.ConstraintBased.FCI import fci
    # from causallearn.search.ConstraintBased.CDNOD import cdnod
    # from causallearn.utils.GraphUtils import GraphUtils
    # from causallearn.search.FCMBased import lingam
    # from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion
    # from causallearn.graph.SHD import SHD
    # import sklearn.metrics

    # from causallearn.graph.GeneralGraph import GeneralGraph
    # from causallearn.graph.GraphNode import GraphNode
    # from causallearn.graph.Edge import Edge
    # from causallearn.graph.Endpoint import Endpoint
    # from causallearn.utils.DAG2CPDAG import dag2cpdag

    # def array2cpdag(adj_array):
    #     g = GeneralGraph([])
    #     node_map = {}
    #     num_nodes = adj_array.shape[0]
        
    #     # Create nodes
    #     for i in range(num_nodes):
    #         node_name = f"X{i+1}"
    #         node_map[node_name] = GraphNode(node_name)
    #         g.add_node(node_map[node_name])
        
    #     # Create edges
    #     for i in range(num_nodes):
    #         for j in range(num_nodes):
    #             if adj_array[i, j] == 1:
    #                 node1 = node_map[f"X{i+1}"]
    #                 node2 = node_map[f"X{j+1}"]
    #                 edge = Edge(node1, node2, Endpoint.TAIL, Endpoint.ARROW)
    #                 g.add_edge(edge)
        
    #     truth_cpdag = dag2cpdag(g)
    #     return truth_cpdag

    # def evaluate_algorithms():
    #     results = {'PC': [], 'CDNOD': []} #  'FCI': [], 'CDNOD': [], 'LiNGAM': []}
        
    #     for _ in range(1):
    #         base_simulator = DataSimulator()
    #         graph, data = base_simulator.generate_dataset(
    #             function_type='linear',
    #             n_nodes=10,
    #             n_samples=10000,
    #             edge_probability=0.15,
    #             noise_type='gaussian',
    #             n_domains=2
    #         )
    #         # Convert data to DataFrame
    #         df = pd.DataFrame(data)
    #         df_wo_domain = df.drop(columns=['domain_index']).values
    #         c_indx = df['domain_index'].values.reshape(-1, 1)
    #         print('graph: ', graph)
            
    #         # # PC Algorithm
    #         pc_graph = pc(df_wo_domain)
    #         pc_shd = SHD(array2cpdag(graph), pc_graph.G).get_shd()
    #         print('pc_graph: ', pc_graph.G.graph)

    #         adj = AdjacencyConfusion(array2cpdag(graph), pc_graph.G)
    #         pc_precision = adj.get_adj_precision()
    #         pc_recall = adj.get_adj_recall()
    #         pc_f1 = 2 * pc_precision * pc_recall / (pc_precision + pc_recall)
    #         results['PC'].append((pc_shd, pc_precision, pc_recall, pc_f1))
    #         print(results['PC'])

    #         # # FCI Algorithm
    #         # fci_graph, _ = fci(df_wo_domain)
    #         # fci_shd = SHD(array2cpdag(graph), fci_graph).get_shd()
    #         # adj = AdjacencyConfusion(array2cpdag(graph), fci_graph)
    #         # fci_precision = adj.get_adj_precision()
    #         # fci_recall = adj.get_adj_recall()
    #         # fci_f1 = 2 * fci_precision * fci_recall / (fci_precision + fci_recall)
    #         # results['FCI'].append((fci_shd, fci_precision, fci_recall, fci_f1))
            
    #         # CDNOD Algorithm
    #         cdnod_graph = cdnod(df_wo_domain, c_indx, alpha=0.01)
    #         print(cdnod_graph.G.graph)
    #         cdnod_graph.G.remove_node(GraphNode(f'X{len(cdnod_graph.G.nodes)}'))
    #         print(cdnod_graph.G.graph)
    #         cdnod_shd = SHD(array2cpdag(graph), cdnod_graph.G).get_shd()
    #         adj = AdjacencyConfusion(array2cpdag(graph), cdnod_graph.G)
    #         cdnod_precision = adj.get_adj_precision()
    #         cdnod_recall = adj.get_adj_recall()
    #         cdnod_f1 = 2 * cdnod_precision * cdnod_recall / (cdnod_precision + cdnod_recall)
    #         results['CDNOD'].append((cdnod_shd, cdnod_precision, cdnod_recall, cdnod_f1))
            
    #         # # Drop the last node of the estimated graph
    #         # cdnod_graph_dropped = cdnod_graph.G
    #         # cdnod_shd_dropped = SHD(array2cpdag(graph), cdnod_graph_dropped).get_shd()
    #         # adj = AdjacencyConfusion(array2cpdag(graph), cdnod_graph_dropped)
    #         # cdnod_precision_dropped = adj.get_adj_precision()
    #         # cdnod_recall_dropped = adj.get_adj_recall()
    #         # cdnod_f1_dropped = 2 * cdnod_precision_dropped * cdnod_recall_dropped / (cdnod_precision_dropped + cdnod_recall_dropped)
    #         # results['CDNOD_dropped'] = results.get('CDNOD_dropped', []) + [(cdnod_shd_dropped, cdnod_precision_dropped, cdnod_recall_dropped, cdnod_f1_dropped)]
            
    #         # # LiNGAM Algorithm
    #         # model = lingam.DirectLiNGAM()
    #         # model.fit(df_wo_domain)
    #         # inferred_flat_lingam = np.where(model.adjacency_matrix_ != 0, 1, 0)
    #         # lingam_shd = np.sum(graph.flatten() != inferred_flat_lingam.flatten())
    #         # lingam_precision = sklearn.metrics.precision_score(graph.flatten(), inferred_flat_lingam.flatten())
    #         # lingam_recall = sklearn.metrics.recall_score(graph.flatten(), inferred_flat_lingam.flatten())
    #         # lingam_f1 = sklearn.metrics.f1_score(graph.flatten(), inferred_flat_lingam.flatten())
    #         # results['LiNGAM'].append((lingam_shd, lingam_precision, lingam_recall, lingam_f1))

    #         # 1. evluation difference between cpdag and dag
    #         #    pc/ges/fci/cdnod: cpdag - cpdag
    #         #    lingam/notears: dag - dag
    #         # 2. cdnod (fisherz) -- pc (fisherz)
    #         # 3. post-processing: pc/ges/fci/cdnod -> --? -> dag -> evaluation()
    #         # 4. x-y-z => x->y<-z, keep it there
    #     return results

    # # Use the base simulator to generate data

    # # Evaluate algorithms
    # results = evaluate_algorithms()

    # # Calculate average performance
    # avg_results = {}
    # for alg, metrics in results.items():
    #     shds, precisions, recalls, f1s = zip(*metrics)
    #     avg_results[alg] = {
    #         'avg_shd': np.mean(shds),
    #         'avg_precision': np.mean(precisions),
    #         'avg_recall': np.mean(recalls),
    #         'avg_f1': np.mean(f1s)
    #     }

    # print("Average performance for each algorithm over 10 simulations:")
    # for alg, avg_metrics in avg_results.items():
    #     print(f"{alg}: SHD={avg_metrics['avg_shd']}, Precision={avg_metrics['avg_precision']}, Recall={avg_metrics['avg_recall']}, F1={avg_metrics['avg_f1']}")

    graph, data = test_mixed_data_generation()
    print(graph)
    print(data)