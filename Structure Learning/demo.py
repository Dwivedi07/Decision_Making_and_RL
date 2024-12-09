import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import sys
import csv
import networkx as nx
import math
import random
import copy
import time
from scipy.special import gammaln 
from matplotlib import pyplot as plt



class BayesianNetworkLearner:
    def __init__(self, nodes, data):
        self.Nodes = nodes  # List of nodes in the network
        self.data = pd.DataFrame(data, columns=nodes)  # Data used for learning as a pandas DataFrame
        self.DAG = None  # This will store the final directed acyclic graph
        self.BS = []  # Stores the Bayesian scores during learning

    def BayesianScore(self):
        # Your function that computes the Bayesian score of the current DAG
        return random.random()  # Placeholder for the actual Bayesian score calculation

    def conditional_independence_test(self, X, Y, Z=[]):
        """
        Chi-squared conditional independence test for X and Y given Z.
        - X: the first node (feature/variable)
        - Y: the second node (feature/variable)
        - Z: the set of conditioning nodes (features/variables)
        """
        # If no conditioning set Z, just do a simple chi-square test for independence between X and Y
        if len(Z) == 0:
            contingency_table = pd.crosstab(self.data[X], self.data[Y])
            chi2, p, dof, ex = chi2_contingency(contingency_table)
            return p > 0.05  # Return True if p-value is greater than significance level (0.05)

        # If there is a conditioning set Z, we'll have to use stratification (conditional on Z)
        for value in self.data[Z].drop_duplicates().values:
            subset_data = self.data[(self.data[Z] == value).all(axis=1)]
            contingency_table = pd.crosstab(subset_data[X], subset_data[Y])
            
            if contingency_table.empty:
                continue  # If we don't have enough data, skip this conditioning

            chi2, p, dof, ex = chi2_contingency(contingency_table)
            if p <= 0.05:
                return False  # If any subset shows dependence, return False

        return True  # Return True if all subsets show independence

    def max_min_parents_children(self):
        """
        Implements the MMPC phase. This will find parents and children for each node using a conditional independence test.
        Returns an undirected skeleton (edges between related nodes).
        """
        skeleton = set()  # Set to store undirected edges

        # Iterate over all pairs of nodes to identify dependencies
        for i in range(len(self.Nodes)):
            for j in range(i + 1, len(self.Nodes)):
                node_i = self.Nodes[i]
                node_j = self.Nodes[j]

                # Conditional independence test with an empty conditioning set (Z = [])
                if not self.conditional_independence_test(node_i, node_j, Z=[]):
                    # If node_i and node_j are dependent, add an edge between them
                    skeleton.add((node_i, node_j))

        return skeleton

    def hill_climbing_phase(self):
        """
        Implements the hill-climbing phase, starting from the skeleton produced by MMPC.
        This will refine the structure by maximizing the Bayesian score.
        """
        best_DAG = self.DAG.copy()
        best_score = self.BayesianScore()

        improved = True
        while improved:
            improved = False
            for i in range(len(self.Nodes)):
                for j in range(i + 1, len(self.Nodes)):
                    node_i = self.Nodes[i]
                    node_j = self.Nodes[j]

                    # Try adding, removing, or reversing edges
                    possible_operations = []

                    if not self.DAG.has_edge(node_i, node_j) and not self.DAG.has_edge(node_j, node_i):
                        possible_operations.append(("add", node_i, node_j))
                    if self.DAG.has_edge(node_i, node_j):
                        possible_operations.append(("remove", node_i, node_j))
                        possible_operations.append(("reverse", node_i, node_j))

                    for operation, src, dest in possible_operations:
                        # Apply the operation and compute new score
                        if operation == "add":
                            self.DAG.add_edge(src, dest)
                        elif operation == "remove":
                            self.DAG.remove_edge(src, dest)
                        elif operation == "reverse":
                            self.DAG.remove_edge(src, dest)
                            self.DAG.add_edge(dest, src)

                        new_score = self.BayesianScore()

                        # If the score improved, keep the modification
                        if new_score > best_score:
                            best_score = new_score
                            best_DAG = self.DAG.copy()
                            improved = True
                        else:
                            # Revert the change if it did not improve the score
                            if operation == "add":
                                self.DAG.remove_edge(src, dest)
                            elif operation == "remove":
                                self.DAG.add_edge(src, dest)
                            elif operation == "reverse":
                                self.DAG.remove_edge(dest, src)
                                self.DAG.add_edge(src, dest)

        # Store the final DAG and score
        self.DAG = best_DAG
        return best_DAG, best_score

    def max_min_hill_climbing(self):
        """
        Full MMHC algorithm that combines the MMPC and hill-climbing phases.
        """
        print("Starting MMPC phase...")
        skeleton = self.max_min_parents_children()

        # Initialize the DAG with the skeleton
        self.DAG = nx.DiGraph()
        self.DAG.add_edges_from(skeleton)
        
        print("Starting Hill-Climbing phase...")
        best_DAG, best_score = self.hill_climbing_phase()
        
        print(f"Final Bayesian score: {best_score}")
        return best_DAG

def write_gph(outputfilepath,G):
    with open(outputfilepath, 'w') as f:
        for edge in G.edges():
            f.write("{}, {}\n".format(edge[0], edge[1]))

def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]

    df = pd.read_csv(inputfilename)
    nodes = df.columns.tolist()

    data = df.iloc[:].to_numpy()
    # nodes = ['A', 'B', 'C', 'D']  # List of nodes
    # data = np.random.randint(0, 2, size=(1000, len(nodes)))  # Dummy binary dataset

    bn_learner = BayesianNetworkLearner(nodes, data)
    best_DAG = bn_learner.max_min_hill_climbing()
    write_gph(outputfilename, best_DAG)


if __name__ == '__main__':
    main()

# # Example usage:
# df = pd.read_csv(example/example.csv)
# nodes = df.columns.tolist()

# data = df.iloc[:].to_numpy()
# # nodes = ['A', 'B', 'C', 'D']  # List of nodes
# # data = np.random.randint(0, 2, size=(1000, len(nodes)))  # Dummy binary dataset

# bn_learner = BayesianNetworkLearner(nodes, data)
# best_DAG = bn_learner.max_min_hill_climbing()

