import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union, List

import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from algorithm.wrappers.base import CausalDiscoveryAlgorithm
from algorithm.evaluation.evaluator import GraphEvaluator
from castle.algorithms import NotearsNonlinear


class NOTEARSNonlinear(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'lambda1': 0.01,  # L1 regularization parameter
            'lambda2': 0.01,  # L2 regularization parameter
            'max_iter': 100,
            'h_tol': 1e-8,
            'rho_max': 1e+16,
            'w_threshold': 0.3,
            'hidden_layers': (10, 1),  # Default architecture with one hidden layer
            'bias': True,
            'model_type': 'mlp',  # 'mlp' or 'sob'
            'device_type': 'auto',  # Device type ('cpu', 'gpu', or 'auto')
            'device_ids': 0,
            # Additional parameters for Sobolev model
            'expansions': 10  # Only used when model_type='sob'
        }
        self._params.update(params)
        # Automatically decide device_type if set to 'auto'
        if self._params.get('device_type', 'cpu') == 'auto':
            try:
                import torch
                self._params['device_type'] = 'gpu' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                self._params['device_type'] = 'cpu'

    @property
    def name(self):
        return "NOTEARS-MLP"

    def get_params(self):
        return self._params
    
    def get_primary_params(self):
        self._primary_param_keys = ['lambda1', 'lambda2', 'max_iter', 'w_threshold']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}
    
    def get_secondary_params(self):
        self._secondary_param_keys = ['hidden_layers', 'model_type', 'device_type',
                                      'h_tol', 'rho_max', 'bias', 'device_ids', 'expansions']
        return {k: v for k, v in self._params.items() if k in self._secondary_param_keys}

    def fit(self, data: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, Dict]:
        """
        Fit the NOTEARS-MLP algorithm to data.
        
        Args:
            data: Data matrix as numpy array or pandas DataFrame
            
        Returns:
            adj_matrix: Estimated adjacency matrix
            info: Additional information from the algorithm
            model: Fitted model object
        """
        # Check and remove domain_index if it exists
        if isinstance(data, pd.DataFrame) and 'domain_index' in data.columns:
            data = data.drop(columns=['domain_index'])
            
        if isinstance(data, pd.DataFrame):
            node_names = list(data.columns)
            data = data.values
        else:
            node_names = [f"X{i}" for i in range(data.shape[1])]
            data = np.array(data)


        # Initialize NOTEARS-MLP from Castle
        all_params = {**self.get_primary_params(), **self.get_secondary_params()}

        model = NotearsNonlinear(**all_params)
        
        # Fit the model
        model.learn(data)
        
        # Get the adjacency matrix
        adj_matrix = model.causal_matrix.T

        # Prepare additional information
        info = {
            'model': model,
            'weight_causal_matrix': model.weight_causal_matrix
        }

        return adj_matrix, info, model

    def test_algorithm(self):
        # Generate some sample data
        np.random.seed(42)
        n_samples = 1000
        X1 = np.random.normal(0, 1, n_samples)
        X2 = 0.5 * X1 + np.random.normal(0, 0.5, n_samples)
        X3 = 0.3 * X1 + 0.7 * X2 + np.random.normal(0, 0.3, n_samples)
        X4 = 0.6 * X2 + np.random.normal(0, 0.4, n_samples)
        X5 = 0.4 * X3 + 0.5 * X4 + np.random.normal(0, 0.2, n_samples)
        
        df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5})

        print("Testing NOTEARS-MLP algorithm with pandas DataFrame:")

        # Ground truth graph
        gt_graph = np.array([
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0]
        ])

        # Initialize lists to store metrics
        f1_scores = []
        precisions = []
        recalls = []
        shds = []

        # Run the algorithm 10 times
        for _ in range(1):
            adj_matrix, info, _ = self.fit(df)
            evaluator = GraphEvaluator()
            metrics = evaluator.compute_metrics(gt_graph, adj_matrix)
            f1_scores.append(metrics['f1'])
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            shds.append(metrics['shd'])

        # Calculate average and standard deviation
        avg_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        avg_precision = np.mean(precisions)
        std_precision = np.std(precisions)
        avg_recall = np.mean(recalls)
        std_recall = np.std(recalls)
        avg_shd = np.mean(shds)
        std_shd = np.std(shds)

        print("\nAverage Metrics over 10 runs:")
        print(f"F1 Score: {avg_f1:.4f} ± {std_f1:.4f}")
        print(f"Precision: {avg_precision:.4f} ± {std_precision:.4f}")
        print(f"Recall: {avg_recall:.4f} ± {std_recall:.4f}")
        print(f"SHD: {avg_shd:.4f} ± {std_shd:.4f}")

if __name__ == "__main__":
    notears_mlp = NOTEARSNonlinear()
    notears_mlp.test_algorithm()
