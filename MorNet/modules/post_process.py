# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 11:16:05 2022

@author: isabelle rocamora
"""
import rasterio as rio
import numpy as np
from .utils_new import getBatch, normalize
import sys


#%% FUNCTIONS
def predict_proba(model, images, params):
    """
    Run model predictions in "test" mode (Launches model without training)
    Returns the prediction as a probability output
    """
    # 1 BRANCH
    if params.nb_branch == 1 :
        iterations = images.shape[0] / params.batch_map
        predictions = []
        if images.shape[0] % params.batch_map != 0:
            iterations = iterations + 1
        for i in range ( int(iterations)):
            batch_image = getBatch(images, i, params.batch_map)
            pred_batch = model.predict(batch_image)
            predictions.append(pred_batch)

    # 2 BRANCH
    elif params.nb_branch == 2:
        images_1, images_2 = images
        iterations = images_1.shape[0] / params.batch_map
        predictions = []
        if images_1.shape[0] % params.batch_map != 0:
            iterations = iterations + 1
        for i in range ( int(iterations)):
            batch_1 = getBatch(images_1, i, params.batch_map)
            batch_2 = getBatch(images_2, i, params.batch_map)
            pred_batch, pred_1, pred_2 = model.predict([batch_1, batch_2])
            predictions.append(pred_batch)
    
    # 3 BRANCH
    else :
        images_1, images_2, images_3 = images
        iterations = images_1.shape[0] / params.batch_map
        predictions = []
        if images_1.shape[0] % params.batch_map != 0:
            iterations = iterations + 1
        for i in range ( int(iterations)):
            batch_1 = getBatch(images_1, i, params.batch_map)
            batch_2 = getBatch(images_2, i, params.batch_map)
            batch_3 = getBatch(images_3, i, params.batch_map)
            pred_batch, pred_1, pred_2, pred_3 = model.predict([batch_1, batch_2, batch_3])
            predictions.append(pred_batch)
    return np.concatenate(predictions, axis=0)


def getARow(data, i, border, ncol):
    """
    Extract patches on an entire raster line 
    """
    patch = []
    end_limit = ncol-border
    for j in range(border,end_limit):
        begin_j = j - border
        end_j = j + border + 1
        begin_i = i - border
        end_i = i + border + 1
        patch.append( data[begin_i:end_i , begin_j:end_j , :] )
    return np.array(patch)


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
            data = np.where(data<0, 0, data)
            print(np.min(data), np.max(data))
            data = 10 * np.log10(data + 10**(-6))
        stack_norm[i,:,:] = normalize(data, bands)
        i = i+1
    stack_norm = stack_norm[1:,:,:]
    stack_norm = np.moveaxis(stack_norm, 0, -1)
    return stack_norm


def export_tiff (array, name, origin, res=10, num_crs=32645, no_data=-99999):
    """
    Export prediction map as .tiff file 
    """
    dim=array.shape
    crs=rio.crs.CRS.from_epsg(num_crs)
    transform=rio.transform.from_origin( origin[0], origin[1], res, res)
    new_array = rio.open(name, 'w', driver='GTiff',
                         height = dim[0], width = dim[1],
                         count=1, dtype=array.dtype, nodata=no_data,
                         crs=crs, transform=transform)
    new_array.write(array, 1)
    new_array.close()



#%% MAIN FUNCTION
def create_map (model, params):
    """
    Creation of the prediction map over the entire study area after loading the best model
    """
    ### LOAD AND NORMALIZE DATA
    gt = rio.open(params.path_data + params.stack_info["nodata"][-1])
    epsg = gt.read_crs()
    origin = gt.get_transform() 
    ROI_size = gt.read(1).shape

    gt = gt.read(1)
    stack_norm = normalize_data(params, ROI_size)
    print(np.shape(stack_norm))
    
    #### RESTORE model with learnt weights
    model.load_weights(params.path_results +"best_model"+params.name)
    print("MODEL RESTORED")
    
    #### PREDICTIONS
    classifRaster = np.zeros(stack_norm[:,:,0].shape)
    nrow, ncol = stack_norm[:,:,0].shape

    end_limit = nrow-params.border
    for i in range(params.border, end_limit):

        # Display progress bar
        tot = end_limit - params.border
        chaine = str(round( ((i-params.border)/tot)*100, 0)) + '%'
        sys.stdout.write('\r' + chaine)
        
        # Extract data and predict
        if params.nb_branch == 1 :
            tempData = getARow(stack_norm, i, params.border, ncol)
            rowClassif = predict_proba(model, tempData, params)
            classifRaster[i,params.border:(ncol-params.border)] = rowClassif[:,-1]

        elif params.nb_branch == 2 :
            tempData = getARow(stack_norm, i, params.border, ncol)
            tempMNT = tempData[:,:,:,:params.nb_bands[0]]
            tempS2 = tempData[:,:,:,params.nb_bands[0]:]
            rowClassif = predict_proba(model, [tempMNT, tempS2], params)
            classifRaster[i,params.border:(ncol-params.border)] = rowClassif[:,-1]
        else :
            lim = params.nb_bands[0] + params.nb_bands[1]
            tempData = getARow(stack_norm, i, params.border, ncol)
            tempMNT = tempData[:,:,:,:params.nb_bands[0]]
            tempSAR = tempData[:,:,:,params.nb_bands[0]:lim]
            tempS2 = tempData[:,:,:,-params.nb_bands[2]:]
            rowClassif = predict_proba(model, [tempMNT, tempSAR, tempS2], params)
            classifRaster[i,params.border:(ncol-params.border)] = rowClassif[:,-1]
    
      
    ### SAVE ###
    export_tiff(classifRaster, params.path_results + "classif_results" + params.name + ".tif", 
                (origin[0], origin[3]), num_crs=epsg.to_epsg())    
    gt = None

def evaluation (model, images, labels, params):
    """
    Calculates the f1-score on the test dataset using the best model
    """
    
    #### RESTORE model with learnt weights
    model.load_weights(params.path_results +"best_model"+params.name)
    print("MODEL RESTORED")

    #### PREDICTIONS
    if params.nb_branch == 1 :
        pred = np.argmax(model.predict(images), axis=-1)

    elif params.nb_branch == 2 :
        images_1 = images[:,:,:,:params.nb_bands[0]]
        images_2 = images[:,:,:,params.nb_bands[0]:]
        pred = np.argmax(model.predict([images_1, images_2])[0], axis=-1)
        
    else :
        lim = params.nb_bands[0] + params.nb_bands[1]
        images_1 = images[:,:,:,:params.nb_bands[0]]
        images_2 = images[:,:,:,params.nb_bands[0]:lim]
        images_3 = images[:,:,:,-params.nb_bands[2]:]
        pred = np.argmax(model.predict([images_1, images_2, images_3])[0], axis=-1)

    # COMPUTE f1-score
    f1 = f1_score(labels, pred, average=None)

    return f1
