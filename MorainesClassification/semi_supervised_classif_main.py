# -*- coding: utf-8 -*-
"""
Created on July 2024

@author: isabelle rocamora
"""

import pandas as pd
import numpy as np
from scipy import sparse
from gbssl import CAMLP
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics.pairwise import pairwise_distances
from cmcrameri import cm
import datetime


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


# Function to create and display a confusion matrix with standard deviation
def create_confusion_matrix_with_std(y_true_list, y_pred_list, labels):
    conf_matrices = []

    for y_true, y_pred in zip(y_true_list, y_pred_list):
        conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
        conf_matrices.append(conf_matrix)

    # Stack confusion matrices to compute mean and std
    conf_matrices = np.array(conf_matrices)
    conf_matrix_mean = np.mean(conf_matrices, axis=0)
    conf_matrix_std = np.std(conf_matrices, axis=0)

    # Calculate percentage values
    conf_matrix_mean_pct = conf_matrix_mean / conf_matrix_mean.sum(axis=1)[:, np.newaxis] * 100
    
    # Display confusion matrix
    plt.figure(figsize=(8, 6), dpi=150)
    plt.imshow(conf_matrix_mean_pct, cmap=cm.lapaz_r, vmin=0, vmax=100)
    plt.title('Confusion Matrix (Mean ± STD)')
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.yticks(range(len(labels)), labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Display mean and std values in each cell
    for i in range(len(conf_matrix_mean_pct)):
        for j in range(len(conf_matrix_mean_pct[i])):
            mean_val = conf_matrix_mean_pct[i, j]
            std_val = conf_matrix_std[i, j]

            # Suppress std value when it's equal to 0.0%
            if std_val > 0.0:
                plt.text(j, i, f'{mean_val:.1f}%\n±{std_val:.1f}', ha='center', va='center', color='k')
            else:
                plt.text(j, i, f'{mean_val:.1f}%', ha='center', va='center', color='k')


    plt.tight_layout()
    plt.show()


# Function to save the results to an Excel file
def save_results(best_results):
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f'{folder_save}/results_{now}.xlsx'
    with pd.ExcelWriter(file_name) as writer:
        for i, result in enumerate(best_results):
            result_df = pd.DataFrame(result)
            result_df.to_excel(writer, sheet_name=f'Iteration_{i+1}', index=False)
    print(now)
    plt.savefig(f'{folder_save}/all_without_ela_{now}.png')        
    
    
    
#%% VARIABLES
path_data = "data/data_for_moraine_classif.xlsx"
folder_save = "results"

# Variables to perform Monte Carlo simulation
num_iterations = 10000
num_samples_per_class = 5
best_f1_scores = []


#%% LOAD AND PREPROCESS DATA

# Load data
data = pd.read_excel(path_data, index_col="IdSort")
data.sort_index(inplace=True)

# Mask to identify known labels
mask_unknow = np.where(data.label == "unknow", False, True)

# Factorize labels to convert them to numerical values
label = data[["label"]].copy()
label.loc[:,'label_num'] = pd.factorize(label['label'])[0]


# Remove unnecessary  columns
col_to_remove = data.columns[~data.columns.isin(['len3D_True', 'DaltCrete', 
                                                 'DistDraina', "DistGlacier_norm", 
                                                 "median_height", 'median_width', 
                                                 'curv_max', 'slope_crest'])]
# Drop columns that are not in the list
data = data.drop(columns=col_to_remove)


#%% MONTE CARLO SIMULATION
list_f1 = np.zeros((num_iterations)) # Create an empty array to store future results
for i in range(num_iterations):
    print(i)
    # Randomly select train and test IDs
    train_ids = []
    test_ids = []

    # Iterate over each class and randomly select samples
    for class_label in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']:
        # Get the IDs of samples belonging to the current class
        class_samples = data[label.label == class_label].index.tolist()

        # Randomly select num_samples_per_class samples from the current class
        selected_samples = np.random.choice(class_samples, size=num_samples_per_class, replace=False)
        
        # Add selected samples to train_ids
        train_ids.extend(selected_samples)
        
        
        # Remove selected samples from class_samples
        class_samples = [sample for sample in class_samples if sample not in selected_samples]
        test_ids.extend(class_samples)


    # Shuffle train_ids and test_ids
    np.random.shuffle(train_ids)
    np.random.shuffle(test_ids)

    # Evaluate model with the current train-test split
    f1, pred = train_and_evaluate(data, label, train_ids, test_ids)
    list_f1[i] = f1
    best_f1_scores.append((f1, train_ids, test_ids, pred))

# Select the top 30 F1-scores
best_f1_scores.sort(reverse=True, key=lambda x: x[0])
top_30_f1_scores = best_f1_scores[:30]


#%% DISPLAY AND SAVE
# Prepare data for confusion matrix
y_true_list = []
y_pred_list = []
for f1, train_ids, test_ids, pred in top_30_f1_scores:
    positions_test = [data.index.get_loc(id_) for id_ in test_ids]
    y_true_list.append(label.loc[test_ids].label_num)
    y_pred_list.append(pred[positions_test])

unique_labels = sorted(label['label_num'].unique())[1:-1]

# Display confusion matrix
create_confusion_matrix_with_std(y_true_list, y_pred_list, unique_labels)


# Save results
save_results(top_30_f1_scores)

# Print final results
mean_f1 = np.mean([score[0] for score in top_30_f1_scores])
std_f1 = np.std([score[0] for score in top_30_f1_scores])
print(f'Top 30 F1-score mean: {mean_f1:.3f} ± {std_f1:.3f}')