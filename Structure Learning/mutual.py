import numpy as np
import pandas as pd
from scipy.stats import entropy
import networkx as nx

def joint_prob(df, col1, col2):
    joint_dist = pd.crosstab(df[col1], df[col2], normalize=False)  #counts the occurance combined
    return joint_dist.values


def marginal_prob(df, col):
    marginal_dist = df[col].value_counts(normalize=False)
    return marginal_dist.values

# Function to compute mutual information I(X; Y)
def mutual_information(df, col1, col2):
    joint = joint_prob(df, col1, col2)
    marginal_x = marginal_prob(df, col1)
    marginal_y = marginal_prob(df, col2)

    # Compute mutual information I(X; Y)
    mi = 0
    for i in range(joint.shape[0]):
        for j in range(joint.shape[1]):
            if joint[i, j] > 0:  # Avoid division by zero
                mi += joint[i, j] * np.log(joint[i, j] / (marginal_x[i] * marginal_y[j]))
    
    return mi

# Function to compute mutual information for each pair of columns
def compute_mutual_info(df):
    columns = df.columns
    # print(columns)
    mi_matrix = pd.DataFrame(index=columns, columns=columns)

    for col1 in columns:
        for col2 in columns:
            if col1 != col2:
                mi_matrix.loc[col1, col2] = mutual_information(df, col1, col2)
            else:
                mi_matrix.loc[col1, col2] = np.nan  

    return mi_matrix


def top_pairs(path,k):
    df = pd.read_csv(path)
    
    mi_matrix = compute_mutual_info(df)
    print(mi_matrix)
    mi_pairs = mi_matrix.stack().reset_index()
    mi_pairs.columns = ['Variable 1', 'Variable 2', 'Mutual Information']
    
   
    top_pairs = mi_pairs.sort_values(by='Mutual Information', ascending=False).head(k)
    


    return top_pairs.to_numpy()





