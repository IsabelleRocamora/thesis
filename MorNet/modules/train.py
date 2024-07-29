# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 14:51:29 2022

@author: isabelle rocamora
"""

import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from .utils import getBatch
from time import time

#%% MAIN FUNCTION
def run (model, x_train, y_train, x_valid, y_valid, , x_test, y_test, params):
    """ Train and test model"""

    # Cost and optimizer functions
    loss_fct = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    
    train_acc = tf.keras.metrics.Accuracy(name='train_acc')
    best_fmeasure = 0
    metrics_train = np.ones( (params.nb_epoch, 6))*-10
    
    for epoch in range (params.nb_epoch):
        print("epoch n°", epoch + 1)
        t0 = time()
        # preprocess
        if params.nb_branch == 2:
            x_train[0], x_train[1], y_train = shuffle(x_train[0], x_train[1], y_train)
        elif params.nb_branch == 3:
            x_train[0], x_train[1], x_train[2], y_train = shuffle(x_train[0], x_train[1], x_train[2], y_train)
        else :
            x_train, y_train = shuffle(x_train, y_train)

        # training
        train_loss = train_step(x_train, y_train, model, loss_fct, optimizer, train_acc, params)
        
        # validation
        pred_valid = np.argmax(test_step(x_valid, model, params), axis=-1)
        f1_valid = f1_score(y_valid, pred_valid, average=None)
        f1 = f1_score(y_valid, pred_valid, average="weighted")
        best_fmeasure = keep_best_model(f1, best_fmeasure, model, params.path_results, params.name)

        # metrics storage
        metrics_train[epoch,:] = np.array(( epoch, train_loss, train_acc.result().numpy(),
                                            f1_valid[0], f1_valid[1], f1))

        if params.optimize == "no" :
            # test and others metrics
            pred_test = np.argmax(test_step(x_test, model, params), axis=-1)
            pred_train = np.argmax(test_step(x_train, model, params), axis=-1)
            f1_test = f1_score(y_test, pred_test, average=None)
            metrics_f1[epoch, :] = np.array(( epoch, f1[0], f1[1], 
                                            f1_valid[0], f1_valid[1],
                                            f1_test[0], f1_test[1]))


        t1 = time()
        print("durée epoch (s):", t1-t0)
    
    # save metrics    
    np.savetxt(params.path_results + "/metrics_train" + params.name + ".csv", metrics_train, 
               delimiter=";", fmt="%1.3f", header="epoch, loss, acc")

    if params.optimize == "no" :
        np.savetxt(params.path_results + "/metrics_f1_score" + params.name + ".csv", metrics_f1, 
               delimiter=";", fmt="%1.3f", 
               header="epoch, F1 train pos, F1 train neg, F1 valid pos, F1 valid neg, F1 test pos, F1 test neg")
    

#%% FUNCTIONS
def keep_best_model (f_measure, best_fmeasure, model, path_results, name):
    """ Keep the best model by checking f1 score"""
    if f_measure > best_fmeasure:
        best_fmeasure = f_measure
        model.save_weights( path_results + "best_model" + name)
        print("-----> MODEL SAVED")
    return best_fmeasure


#@tf.function
def train_step(images, labels, model, loss_fct, optimizer, train_acc, params):
    """ Training """
    # 1 BRANCH
    if params.nb_branch == 1:
        tot_loss = 0.0
        iterations = images.shape[0] / params.batch_size
        if images.shape[0] % params.batch_size != 0:
            iterations += 1
        for i in range(int(iterations)):
            batch_images = getBatch(images, i, params.batch_size)
            batch_labels = getBatch(labels, i, params.batch_size)
            with tf.GradientTape() as tape :
                predictions = model(batch_images, training=True)
                loss = loss_fct(batch_labels, predictions)
                gradients = tape.gradient(loss, model.trainable_variables)
                gradients = [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(model.trainable_variables, gradients)]
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                tot_loss = tot_loss + loss
                train_acc(batch_labels, tf.math.argmax(predictions,axis=1))
    # 2 BRANCH
    elif params.nb_branch == 2:
        images_1, images_2 = images
        a = 0.5
        tot_loss = 0.0
        iterations = images_1.shape[0] / params.batch_size
        if images_1.shape[0] % params.batch_size != 0:
            iterations += 1
        for i in range(int(iterations)):
            batch_MNT = getBatch(images_1, i, params.batch_size)
            batch_S2 = getBatch(images_2, i, params.batch_size)
            batch_labels = getBatch(labels, i, params.batch_size)
            with tf.GradientTape() as tape :
                pred, pred_1, pred_2 = model([batch_MNT, batch_S2], training=True)
                loss = loss_fct(batch_labels, pred) + a*loss_fct(batch_labels, pred_1) + a*loss_fct(batch_labels, pred_2)
                gradients = tape.gradient(loss, model.trainable_variables)
                gradients = [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(model.trainable_variables, gradients)]
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                tot_loss = tot_loss + loss
                train_acc(batch_labels, tf.math.argmax(pred,axis=1))
    
    # 3 BRANCH
    else :
        images_1, images_2, images_3 = images
        a = params.cte_loss
        tot_loss = 0.0
        iterations = images_1.shape[0] / params.batch_size
        if images_1.shape[0] % params.batch_size != 0:
            iterations += 1
        for i in range(int(iterations)):
            batch_MNT = getBatch(images_1, i, params.batch_size)
            batch_SAR = getBatch(images_2, i, params.batch_size)
            batch_S2 = getBatch(images_3, i, params.batch_size)
            batch_labels = getBatch(labels, i, params.batch_size)
            with tf.GradientTape() as tape :
                pred, pred_1, pred_2, pred_3 = model([batch_MNT, batch_SAR, batch_S2], training=True)
                loss = loss_fct(batch_labels, pred) + a*loss_fct(batch_labels, pred_1) + a*loss_fct(batch_labels, pred_2) + a*loss_fct(batch_labels, pred_3)
                gradients = tape.gradient(loss, model.trainable_variables)
                gradients = [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(model.trainable_variables, gradients)]
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                tot_loss = tot_loss + loss
                train_acc(batch_labels, tf.math.argmax(pred,axis=1))

    return (tot_loss / iterations)

        
def test_step (images, model, params):
    """ Launches model without training (default training=False)"""
    # 1 BRANCH
    if params.nb_branch == 1:
        iterations = images.shape[0] / params.batch_size
        predictions = []
        if images.shape[0] % params.batch_size != 0:
            iterations = iterations + 1
        for i in range ( int(iterations)):
            batch_image = getBatch(images, i, params.batch_size)
            pred_batch = model(batch_image)
            predictions.append(pred_batch.numpy())

    # 2 BRANCH
    elif params.nb_branch == 2:
        images_1, images_2 = images
        iterations = images_1.shape[0] / params.batch_size
        predictions = []
        if images_1.shape[0] % params.batch_size != 0:
            iterations = iterations + 1
        for i in range ( int(iterations)):
            batch_1 = getBatch(images_1, i, params.batch_size)
            batch_2 = getBatch(images_2, i, params.batch_size)
            pred_batch, pred_1, pred_2 = model([batch_1, batch_2])
            predictions.append(pred_batch.numpy())
    
    # 3 BRANCH
    else :
        images_1, images_2, images_3 = images
        iterations = images_1.shape[0] / params.batch_size
        predictions = []
        if images_1.shape[0] % params.batch_size != 0:
            iterations = iterations + 1
        for i in range ( int(iterations)):
            batch_1 = getBatch(images_1, i, params.batch_size)
            batch_2 = getBatch(images_2, i, params.batch_size)
            batch_3 = getBatch(images_3, i, params.batch_size)
            pred_batch, pred_1, pred_2, pred_3 = model([batch_1, batch_2, batch_3])
            predictions.append(pred_batch.numpy())
    
    return np.vstack(predictions)
