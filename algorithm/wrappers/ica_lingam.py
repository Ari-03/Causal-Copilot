import numpy as np
import pandas as pd
from typing import Dict, Tuple

# use the local causal-learn package
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
causal_learn_dir = os.path.join(root_dir, 'externals', 'causal-learn')
if not os.path.exists(causal_learn_dir):
    raise FileNotFoundError(f"Local causal-learn directory not found: {causal_learn_dir}, please git clone the submodule of causal-learn")
algorithm_dir = os.path.join(root_dir, 'algorithm')
sys.path.append(root_dir)
sys.path.insert(0, causal_learn_dir)



from causallearn.search.FCMBased.lingam.ica_lingam import ICALiNGAM as CLICALiNGAM

from algorithm.wrappers.base import CausalDiscoveryAlgorithm
from algorithm.evaluation.evaluator import GraphEvaluator

class ICALiNGAM(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'random_state': None,
            'max_iter': 1000
        }
        self._params.update(params)

    @property
    def name(self):
        return "ICALiNGAM"

    def get_params(self):
        return self._params

    def get_primary_params(self):
        self._primary_param_keys = ['max_iter']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}

    def get_secondary_params(self):
        self._secondary_param_keys = ['random_state']
        return {k: v for k, v in self._params.items() if k in self._secondary_param_keys}

    def fit(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict, CLICALiNGAM]:
        # Check and remove domain_index if it exists
        if 'domain_index' in data.columns:
            data = data.drop(columns=['domain_index'])
            
        node_names = list(data.columns)
        data_values = data.values

        # Run ICALiNGAM algorithm
        model = CLICALiNGAM(**self.get_primary_params(), **self.get_secondary_params())
        model.fit(data_values)

        # Convert the graph to adjacency matrix
        adj_matrix = self.convert_to_adjacency_matrix(model.adjacency_matrix_)

        # Prepare additional information
        info = {
            'causal_order': model.causal_order_
        }
        return adj_matrix, info, model

    def convert_to_adjacency_matrix(self, adjacency_matrix: np.ndarray) -> np.ndarray:
        adj_matrix = np.where(adjacency_matrix != 0, 1, 0)
        return adj_matrix

    def test_algorithm(self):
        # Generate sample data with linear relationships
        np.random.seed(42)
        n_samples = 1000
        X1 = np.random.uniform(0, 1, n_samples)
        X2 = 0.5 * X1 + np.random.uniform(0, 1, n_samples)
        X3 = 0.3 * X1 + 0.7 * X2 + np.random.uniform(0, 1, n_samples)
        X4 = 0.6 * X2 + np.random.uniform(0, 1, n_samples)
        X5 = 0.4 * X3 + 0.5 * X4 + np.random.uniform(0, 1, n_samples)
        
        df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5})

        print("Testing ICALiNGAM algorithm with pandas DataFrame:")
        params = {
            'random_state': 42,
            'max_iter': 1000
        }
        adj_matrix, info, _ = self.fit(df)
        print("Adjacency Matrix:")
        print(adj_matrix)
        print("Causal Order:")
        print(info['causal_order'])

        # Ground truth graph
        gt_graph = np.array([
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0]
        ])

        # Use GraphEvaluator to compute metrics
        evaluator = GraphEvaluator()
        metrics = evaluator.compute_metrics(gt_graph, adj_matrix)

        print("\nMetrics:")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"SHD: {metrics['shd']:.4f}") 