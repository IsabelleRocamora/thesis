# -*- coding: utf-8 -*-
"""
Created July 2024

@author: isabelle rocamora
"""

import pandas as pd
import numpy as np
from scipy import sparse
from gbssl import CAMLP
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import pairwise_distances


#%% FUNCTIONS

# Function to extract a mutual K-Nearest Neighbors (KNN) graph
def extractKNNGraph(knn, k):
	nrow, _ = knn.shape
	G = sparse.lil_matrix((nrow,nrow))

	for i in range(nrow):
		for j in knn[i,1:k+1]:
			G[i,j]=1

	G_trans = G.transpose()
	# Create a mutual KNN graph
	mKNN = np.minimum(G.todense(),G_trans.todense())
	return sparse.lil_matrix(mKNN)


# Function to perform graph classification using CAMLP
def graph_classification(id_labeled, cl_labeled, distMatrix):
    knn = np.argsort(distMatrix,axis=1)
    G = extractKNNGraph(knn, 20) # Extract mKNN graph with 20 neighbors
    nrow, _ = G.shape
    camlp = CAMLP(graph=G)       # Integrate mKNN graph in CAMLP model
    camlp.fit(np.array(id_labeled),np.array(cl_labeled))  # Fit the CAMLP model
    prob_cl = camlp.predict_proba(np.arange(nrow)) # Predict class probabilities
    return prob_cl, camlp


# Function to train the model and evaluate its performance
def train_and_evaluate(data, label, id_to_train, id_to_test):
    data_numpy = data.to_numpy()

    # Standardize the data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data_numpy)

    # Train model
    positions = [data.index.get_loc(id_) for id_ in id_to_train]
    cl_labeled = label.loc[id_to_train].label_num.to_numpy()
    distMatrix = pairwise_distances(normalized_data)  # Compute distance between samples
    prob_cl, model = graph_classification(positions, cl_labeled, distMatrix) # Create graph and propagate labels
    pred = np.argmax(prob_cl, axis=1)

    # Evaluate model on test set
    positions_test = [data.index.get_loc(id_) for id_ in id_to_test]
    f1 = f1_score(label.loc[id_to_test].label_num, pred[positions_test], average="weighted")
    
    return f1, pred   
    

#%% VARIABLES
path_data = "data/data_for_moraine_classif.xlsx"
path_best_ids = "data/30_best_simulations_ids.xlsx"
path_ela = "data/ela_data.xlsx"
folder_save = "results"


#%% LOAD DATA
# Load geomorphological characteristics data 
data = pd.read_excel(path_data, index_col="IdSort")
data.sort_index(inplace=True) # Sort the DataFrame by index


# Load ela characteristics data  
data_ela = pd.read_excel(path_ela, index_col="IdSort")
data_ela.sort_index(inplace=True) # Sort the DataFrame by index


# Load best IDs from an Excel file
xlsx_file = pd.ExcelFile(path_best_ids)
# Initialize arrays to store train and test IDs
list_train_ids = np.zeros((30, 30))
list_test_ids = np.zeros((30, 102))
n = 0
# Iterate over each sheet in the Excel file
for sheet_name in xlsx_file.sheet_names:
    # Read the sheet into a DataFrame
    df = xlsx_file.parse(sheet_name, header=0)
    # Convert the string representation of IDs to a list of integers for training IDs
    list_str = df.loc[1][0][1:-1].split(', ')
    list_int = [eval(i) for i in list_str]
    list_train_ids[n] = list_int
    # Convert the string representation of IDs to a list of integers for testing IDs
    list_str = df.loc[2][0][1:-1].split(', ')
    list_int = [eval(i) for i in list_str]
    list_test_ids[n] = list_int
    n = n + 1
    
# Define columns to be added from ELA data
col_to_add = ["ela_bm", "ela_melm", "ela_thar", "D_ela_bm_bassin", "D_ela_melm_bassin", "D_ela_thar_bassin"]


#%% PREPROCESS
# Join ELA data to the geomorphological data
data = data.join(data_ela[col_to_add])

# Create a mask for unknown labels
mask_unknow = np.where(data.label == "unknow", False, True)

# Factorize labels to convert them to numerical values
label = data[["label"]].copy()
label.loc[:,'label_num'] = pd.factorize(label['label'])[0]

# Create an empty DataFrame to store results
df = pd.DataFrame()


#%% PROCESS ADD ELA
# List of columns to add one by one for evaluation
add_col = ["", "ela_bm", "ela_melm", "ela_thar", "D_ela_bm_bassin", "D_ela_melm_bassin", "D_ela_thar_bassin"]
for ela in add_col :
    col_to_remove = data.columns[~data.columns.isin(['len3D_True', 'DaltCrete', 
                                                     'DistDraina', "DistGlacier_norm", 
                                                     "median_height", 'median_width', 
                                                     'curv_max', 'slope_crest', ela])] #len3D_True, length_3D
    # Drop columns that are not in the list
    new_data = data.drop(columns=col_to_remove)
    
    
    # Initialize array to store F1 scores for each train-test split
    list_f1 = np.zeros((list_train_ids.shape[0]))

    for i in range(list_f1.shape[0]):
        # Get training and testing IDs for the current iteration
        train_ids = list_train_ids[i]
        test_ids = list_test_ids[i]
        
        # Evaluate the model with the current train-test split
        f1, pred = train_and_evaluate(new_data, label, train_ids, test_ids)
        list_f1[i] = f1
        
        # Store the mean and standard deviation of F1 scores in the DataFrame
        df[ela] = np.array((np.mean(list_f1), np.std(list_f1)))
        
    
#%% PROCESS BOOTSTRAP
# List of columns for bootstrap feature removal
col_bootstrap = ['len3D_True', 'DaltCrete', 'DistDraina', "DistGlacier_norm", 
                 "median_height", 'median_width', 'curv_max', 'slope_crest']

for feature in col_bootstrap :
    # Determine columns to keep by excluding the current feature
    col_to_remove = data.columns[~data.columns.isin(col_bootstrap)] #len3D_True, length_3D
    new_data = data.drop(columns=col_to_remove)
    new_data = new_data.drop(columns=feature)
    
    
    # Initialize array to store F1 scores for each train-test split
    list_f1 = np.zeros((list_train_ids.shape[0]))

    for i in range(list_f1.shape[0]):
        # Get training and testing IDs for the current iteration
        train_ids = list_train_ids[i]
        test_ids = list_test_ids[i]
        
        # Evaluate the model with the current train-test split
        f1, pred = train_and_evaluate(new_data, label, train_ids, test_ids)
        list_f1[i] = f1
        
        # Store the mean and standard deviation of F1 scores in the DataFrame
        df[feature] = np.array((np.mean(list_f1), np.std(list_f1)))
    

#%% SAVE RESULTS
# Rename columns and index in the DataFrame for clarity
df.rename(columns={'': 'ref'}, inplace=True)
df.rename_axis("f1_score", inplace=True)
df.rename(index={0: 'mean', 1: 'std'}, inplace=True)

# Save the results to an Excel file
df.to_excel(f'{folder_save}/f1-score_ela_bootstrap.xlsx')

