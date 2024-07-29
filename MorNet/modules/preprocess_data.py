# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 14:07:05 2021

@author: Isabelle
"""

import numpy as np
import rasterio as rio
from sklearn.utils import shuffle
from .utils import normalize
import sys


#%% LOAD DATA
def load_data (param):
    """
    Load already normalized data
    """
    x_train = np.load(param.path_data + "x_train.npy")
    x_test = np.load(param.path_data + "x_test.npy")
    x_valid = np.load(param.path_data + "x_valid.npy")
    y_train = np.load(param.path_data + "y_train.npy")
    y_test = np.load(param.path_data + "y_test.npy")
    y_valid = np.load(param.path_data + "y_valid.npy")
    
    # checks that datasets are 4-dimensional
    if x_train.ndim != 4 :
        x_train = np.expand_dims(x_train,-1)

    if x_test.ndim != 4 :
        x_test = np.expand_dims(x_test,-1)

    if x_valid.ndim != 4 :
        x_valid = np.expand_dims(x_valid,-1)

    return x_train, x_valid, y_train, y_valid, x_test, y_test



#%% EXTRACTION DATASET
# NEGATIVE PATCHES WITH COORDS
def extractNegPatches(x_coord, y_coord, data, params):
    """
    Extract patches knowing their coordinates
    """
    nrow, ncol = data[:,:,0].shape
    
    if (x_coord - params.border) < 0 or (y_coord - params.border) < 0:
        return None
    if (x_coord + params.border) >=nrow or (y_coord + params.border) >= ncol:
        return None
    begin_i = x_coord - params.border
    begin_j = y_coord - params.border
    end_i = x_coord + params.border + 1
    end_j = y_coord + params.border + 1

    patch = data[begin_i:end_i , begin_j:end_j , :]

    if np.any(patch<0):
        return None
    else:
        return patch


def extractCoords(coord, data, params):
    """
    Run through the list of patch coordinates to give them to the patch extraction function
    """
    patches = []
    for i in range(coord.shape[1]):
        patch = extractNegPatches(coord[0][i],coord[1][i], data, params)
        if patch is not None:
            patches.append(patch)
    return np.array(patches)


# NEGATIVE PATCHES WITHOUT COORDS
def sampleNegative(neg_patchs, pos_training_size, pos_valid_size, pos_test_size):
    """
    From the list of all possible negative patches, randomly distribute the patches between the three datasets.
    Ensuring that the number of negative training and validation patches is the same as the number of positive patches
    and setting the number of negative test patches to 8%, i.e. 538509 samples
    """
    # Define neg_test size
    neg_test_size = 538509

    # Shuffle the patches
    neg_samples = shuffle(neg_patchs)

    # Extract patchs
    neg_training = neg_samples[0:pos_training_size]
    neg_valid = neg_samples[pos_training_size:(pos_training_size+pos_valid_size)]
    neg_test = neg_test = neg_samples[(pos_training_size + pos_valid_size):(pos_training_size + pos_valid_size + neg_test_size)]

    return neg_training, neg_valid, neg_test
    

def extractSpecific(nb, stack_norm, gt, params):
    """
    Reinforced training by selecting 80% of negative patches within 1 km of the main drainage network
    """
    nodata = -99999
    mnt = rio.open(params.path_data + params.dict_infosbands["MNT"][-1]).read(params.dict_infosbands["MNT"][0])
    dist = rio.open(params.path_data + params.dict_infosbands["DistDrainage"][-1]).read(params.dict_infosbands["DistDrainage"][0])

    dist = np.where(mnt==nodata, 99999, dist)
    dist = np.where((dist <= 1000) & (gt == 0), 1, dist)
    dist = np.where(mnt==nodata, -1, dist)
    dist = np.where((dist > 1000) & (gt == 0), 2, dist)

    neg_samples_inf800 = extractPatches([1], stack_norm, dist, params)
    neg_samples_sup800 = extractPatches([2], stack_norm, dist, params)

    nb_train_80 = int(np.round(nb[0] * 0.8))
    nb_valid_80 = int(np.round(nb[1] * 0.8))
    nb_test_80 = int(np.round(nb[2] * 0.8))
    neg_training_80, neg_valid_80, neg_test_80 = sampleNegative(neg_samples_inf800,
                                                                nb_train_80, 
                                                                nb_valid_80,
                                                                nb_test_80)
    nb_train_20 = nb[0] - nb_train_80
    nb_valid_20 = nb[1] - nb_valid_80
    nb_test_20 = nb[2] - nb_test_80
    neg_training_20, neg_valid_20, neg_test_20 = sampleNegative(neg_samples_sup800,
                                                                nb_train_20, 
                                                                nb_valid_20,
                                                                nb_test_20)

    neg_training = np.concatenate([neg_training_80, neg_training_20],axis=0)
    neg_valid = np.concatenate([neg_valid_80, neg_valid_20],axis=0)
    neg_test = np.concatenate([neg_test_80, neg_test_20],axis=0)
    
    return neg_training, neg_valid, neg_test


# POSITIVE PATCHES
def extractPatch(x_coord, y_coord, data, params):
    """
    Extract patches knowing their coordinates and checks their 
    validity (no nodata and the correct patch size)
    """
    if len(params.stack_info) == 1:
        nrow, ncol = data.shape
    else:
        nrow, ncol = data[:,:,0].shape
    
    if (x_coord - params.border) < 0 or (y_coord - params.border) < 0:
        return None
    if (x_coord + params.border) >=nrow or (y_coord + params.border) >= ncol:
        return None
    begin_i = x_coord - params.border
    begin_j = y_coord - params.border
    end_i = x_coord + params.border + 1
    end_j = y_coord + params.border + 1

    if len(params.stack_info) == 1:
        patch = data[begin_i:end_i , begin_j:end_j ]
    else:
        patch = data[begin_i:end_i , begin_j:end_j , :]

    if np.any(patch<0):
        return None
    else:
        return patch


def extractPatches(id_list, data, gt, params):
    """
    Finds the pixel coordinates corresponding to a ground truth identifier
    """
    patches = []
    for el in id_list:
        x, y = np.where(gt == el)
        for i in range(len(x)):
            patch = extractPatch(x[i],y[i], data, params)
            if patch is not None:
                patches.append(patch)
    return np.array(patches)



#%% NORMALIZATION
def normalize_data(params, ROI_size):
    """
    Pre-processing of input data: normalization and creation of a stack of different input data bands
    """
    stack_norm = np.zeros( (len(params.list_bands), ROI_size[0], ROI_size[1]) )
    i = 0
    for key, bands in params.stack_info.items():
        print(key)
        data = rio.open(params.path_data + bands[-1]).read(bands[0])
        if bands[-3] == "log":
            data = 10 * np.log10(data + 10**(-6))
        stack_norm[i,:,:] = normalize(data, bands)
        i = i+1
        
    stack_norm = np.moveaxis(stack_norm, 0, -1)
    return stack_norm



#%% MAIN FUNCTION
def formation_dataset (params, id_train, id_test, id_valid):
    """
    Main function for creating datasets in two ways:
        - by supplying it with a list of known coordinates for the position of the patch's central pixel
        - by asking it to create the datasets from scratch
    in the second case, it is also possible to specify whether or not you want a reinforced training session.
    """

    # Load ground true
    gt = rio.open(params.path_data + "ROI_couche_positive.tif").read(1)
    ROI_size = [gt.shape[0], gt.shape[1]]

    # Load and normalize data
    stack_norm = normalize_data(params, ROI_size)
    
    if params.form_dataset == "coords" :
        # Load coords for patches
        coords = np.load(params.path_data + "index_dataset.npy", allow_pickle=True)
        coords_train = np.vstack([coords[0], coords[1]])
        coords_valid = np.vstack([coords[2], coords[3]])
        coords_test = np.vstack([coords[6], coords[7]])

        # Extract positives patches
        pos_training = extractPatches(id_train, stack_norm, gt, params)
        pos_valid = extractPatches(id_valid, stack_norm, gt, params)
        pos_test = extractPatches(id_test, stack_norm, gt, params)

        # Extract negative patches
        neg_training = extractCoords(coords_train, stack_norm, params)
        neg_valid = extractCoords(coords_valid, stack_norm, params)
        neg_test = extractCoords(coords_test, stack_norm, params)

    if params.form_dataset == "complete" :
        # Extract positives patches
        pos_training = extractPatches(id_train, stack_norm, gt, params)
        pos_valid = extractPatches(id_valid, stack_norm, gt, params)
        pos_test = extractPatches(id_test, stack_norm, gt, params)

        # Extract negative patches
        if params.specific_form == "yes" :
            nb = [np.shape(pos_training)[0], np.shape(pos_test)[0], np.shape(pos_valid)[0]]
            neg_training, neg_valid, neg_test = extractSpecific(nb, stack_norm, gt, params)

        else :
            # Extract negatives patchs
            neg_samples = extractPatches([0], stack_norm, gt, params)
            neg_training, neg_valid, neg_test = sampleNegative(
                                                            neg_samples,
                                                            pos_training.shape[0], 
                                                            pos_valid.shape[0])


    # Concatenate dataset
    y_train = np.concatenate( [np.ones(pos_training.shape[0]), np.zeros(neg_training.shape[0])], axis=0)
    x_train = np.concatenate([pos_training, neg_training],axis=0)

    y_valid = np.concatenate( [np.ones(pos_valid.shape[0]), np.zeros(neg_valid.shape[0])], axis=0)
    x_valid = np.concatenate([pos_valid, neg_valid],axis=0)

    y_test = np.concatenate( [np.ones(pos_test.shape[0]), np.zeros(neg_test.shape[0])], axis=0)
    x_test = np.concatenate([pos_test, neg_test],axis=0)

    # Removes the band used to identify the location of nodata values
    x_train = x_train[:,:,:,1:]
    x_valid = x_valid[:,:,:,1:]
    x_test = x_test[:,:,:,1:]

    # Save dataset (optionnal)
    """
    np.save(params.path_data + "x_train.npy", x_train)
    np.save(params.path_data + "x_valid.npy", x_valid)
    np.save(params.path_data + "x_test.npy", x_test)
    np.save(params.path_data + "y_train.npy", y_train)
    np.save(params.path_data + "y_valid.npy", y_valid)
    np.save(params.path_data + "y_test.npy", y_test)
    """

    return x_train, x_valid, y_train, y_valid, x_test, y_test