# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 17:19:03 2022

@author: isabelle rocamora
"""

from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D, Dropout, BatchNormalization, GlobalMaxPooling2D, Flatten, Concatenate, Add
from tensorflow.keras import Model
import tensorflow as tf

#%% BASICS ELEMENTS (CONV AND FC LAYERS)
class Conv(Model):
    """ 
    Convolution layer with batch normalization and dropout layer
    Inputs : nb_filters = number of filters
             drop_rate = dropout rate
             strides = strides of the convolution along the height and width
    """
    def __init__(self, nb_filters, drop_rate, strides, padding="valid", act="relu", k_size=3): 
        super(Conv, self).__init__()
        self.conv = Conv2D(filters=nb_filters, kernel_size=(k_size,k_size), padding=padding, activation=act, strides=strides)
        self.bn = BatchNormalization()
        self.drop = Dropout(drop_rate)
    
    @tf.function 
    def call(self, inputs, training=False):
        conv = self.conv(inputs)
        conv = self.bn(conv, training=training)
        return self.drop(conv, training=training)

    
class FC(Model):
    """ 
    Fully-connected layer with dropout layer and ReLU activation fonction 
    Inputs : units = number of units
             drop_rate =  dropout rate 
    """
    def __init__(self, units, drop_rate, act='relu'):
        super(FC, self).__init__()
        self.dense = Dense(units, activation=act)
        self.drop = Dropout(drop_rate)
        
    @tf.function 
    def call (self, inputs, training=False):
        d = self.dense(inputs) 
        return self.drop(d, training=training)


    
#%% INTERMEDIATE ELEMENTS (ENCODER AND CLASSIFIER)    
class Encoder(Model):
    """ 
    Encoder block with 3 Conv layers and a pooling or flattened layer
    Inputs : nb_filters = number of filters
             drop_rate = dropout rate
             type_pool = type of pooling function, "GMP" for GlobalMaxPooling2D or "Fla" for Flatten
    """
    def __init__(self, nb_filters, drop_rate, type_pool):
        super(Encoder, self).__init__()
        self.conv1 = Conv(nb_filters=nb_filters, drop_rate=drop_rate, strides=2)
        self.conv2 = Conv(nb_filters=nb_filters, drop_rate=drop_rate, strides=2)
        self.conv3 = Conv(nb_filters=nb_filters, drop_rate=drop_rate, strides=1)
        if type_pool == "GMP":
            self.pool = GlobalMaxPooling2D()
            print("GlobalMaxPooling2D (GMP) layer")
        else:
            self.pool = Flatten()
            print("Flatten layer")

    
    @tf.function
    def call(self, inputs, training=False):
        conv = self.conv1(inputs, training=training)
        conv = self.conv2(conv, training=training)
        conv = self.conv3(conv, training=training)
        return self.pool(conv)


class Classifier(Model):
    """ 
    Classifier block with 2 FC layers and 1 fully-connected with a softmax activation fonction
    Inputs : drop_rate =  dropout rate
    """
    def __init__(self, drop_rate, act='relu'):
        super(Classifier, self).__init__()
        self.d1 = FC(32, drop_rate)
        self.d2 = FC(32, drop_rate)
        self.out = Dense(2, activation='softmax')
        
    @tf.function 
    def call (self, inputs, training=False):
        dense = self.d1(inputs, training=training)
        dense = self.d2(dense, training=training)
        return self.out(dense)


#%% FINAL MODELS
class CNN_1Branch(Model):
    """ 
    Model one branch for mono-source models
    Inputs : nb_filters = number of filters
             drop_rate = dropout rate
    """
    def __init__(self, nb_filters, drop_rate):
        super(CNN_1Branch, self).__init__()
        self.conv1 = Conv(nb_filters=nb_filters, drop_rate=drop_rate, strides=2)
        self.conv2 = Conv(nb_filters=nb_filters, drop_rate=drop_rate, strides=2)
        self.conv3 = Conv(nb_filters=nb_filters, drop_rate=drop_rate, strides=1)
        self.pool = GlobalMaxPooling2D()
        self.final = Classifier(drop_rate = drop_rate)
        
    @tf.function
    def call(self, inputs, training=False):
        conv = self.conv1(inputs, training=training)
        conv = self.conv2(conv, training=training)
        conv = self.conv3(conv, training=training)
        pool = self.pool(conv)
        return self.final(pool, training=training)


class CNN_2Branch(Model):
    """ 
    Model two branch for bi-source models with 2 Encoder blocks and 2 Classifier blocks
    Inputs : nb_filters_1 = number of filters for the 1st branch
             nb_filters_2 = number of filters for the 2nd branch
             drop_rate_1 = dropout rate for the 1st branch
             drop_rate_2 = dropout rate for the 2nd branch
             param = class containing all other necessary parameters like the type of the pooling function (pool_layer)
    """
    def __init__(self, nb_filters_1, nb_filters_2, drop_rate_1, drop_rate_2, params):
        super(CNN_2Branch, self).__init__()
        self.enc1 = Encoder(nb_filters_1, drop_rate_1, params.pool_layer)
        self.enc2 = Encoder(nb_filters_2, drop_rate_2, params.pool_layer)
        if params.fusion_layer == "C":
            self.fusion = Concatenate()
            print("Concatenate function")
        else :
            self.fusion = Add()
            print("Add function")
        self.aux1 = Classifier(drop_rate_2)
        self.aux2 = Classifier(drop_rate_2)
        self.final = Classifier(drop_rate_2)
        
    @tf.function
    def call(self, inputs, training=False):
        inputs_1, inputs_2 = inputs
        feat_enc1 = self.enc1(inputs_1, training=training)
        feat_enc2 = self.enc2(inputs_2, training=training)
        feat_fusion = self.fusion([feat_enc1, feat_enc2])
        pred_1 = self.aux1(feat_enc1, training=training)
        pred_2 = self.aux2(feat_enc2, training=training)
        pred = self.final(feat_fusion, training=training)
        return pred, pred_1, pred_2
    
    
class CNN_3Branch(Model):
    """ Model three branch for multi-source models with 3 Encoder blocks and 3 Classifier blocks
    Inputs : nb_filters_1 = number of filters for the 1st branch
             nb_filters_2 = number of filters for the 2nd branch
             nb_filters_3 = number of filters for the 3rd branch
             drop_rate_1 = dropout rate for the 1st branch
             drop_rate_2 = dropout rate for the 2nd branch
             drop_rate_3 = dropout rate for the 3rd branch
             param = class containing all other necessary parameters like the type of the pooling function (pool_layer) 
    """
    def __init__(self, nb_filters_1, nb_filters_2, nb_filters_3, drop_rate_1, drop_rate_2, drop_rate_3, params):
        super(CNN_3Branch, self).__init__()
        self.enc1 = Encoder(nb_filters_1, drop_rate_1, params.pool_layer)  # MNT
        self.enc2 = Encoder(nb_filters_2, drop_rate_2, params.pool_layer)  # SAR
        self.enc3 = Encoder(nb_filters_3, drop_rate_3, params.pool_layer)  # Opt
        if params.fusion_layer == "C":
            self.fusion = Concatenate()
            print("Concatenate function")
        else :
            self.fusion = Add()
            print("Add function")
        self.aux1 = Classifier(drop_rate_2)
        self.aux2 = Classifier(drop_rate_2)
        self.aux3 = Classifier(drop_rate_2)
        self.final = Classifier(drop_rate_2)

    @tf.function
    def call(self, inputs, training=False):
        inputs_1, inputs_2, inputs_3 = inputs
        feat_enc1 = self.enc1(inputs_1, training=training)
        feat_enc2 = self.enc2(inputs_2, training=training)
        feat_enc3 = self.enc3(inputs_3, training=training)
        feat_fusion = self.fusion([feat_enc1, feat_enc2, feat_enc3])
        pred_1 = self.aux1(feat_enc1, training=training)
        pred_2 = self.aux2(feat_enc2, training=training)
        pred_3 = self.aux3(feat_enc3, training=training)
        pred = self.final(feat_fusion, training=training)
        return pred, pred_1, pred_2, pred_3
