from modules.parameters import Param
from modules.preprocess_data import formation_dataset, load_data
from modules.model import CNN_1Branch, CNN_2Branch, CNN_3Branch
from modules.train import run
from modules.post_process import evaluation, create_map
import sys
import numpy as np
from time import time

if len(sys.argv) < 2 :
    print ("Usage: "+sys.argv[0]+" Config.cfg manquant")
    sys.exit(1)

#%% VARIABLES
configFile = sys.argv[1]
params = Param(configFile)
print("name :", params.name)
print("list_bands :", params.list_bands)
print("number of branch :", params.nb_branch)
print("name config file :", sys.argv[1])
t0 = time()

# ideentifiers of moraines assigned to each dataset
id_train = np.array([1, 2, 4, 5, 16, 17, 18, 24, 25, 26, 27, 29, 30, 32, 43, 44, 46, 47, 48])
id_valid = np.array([3, 15, 28, 31, 45])
id_test = np.setdiff1d(np.array(range(1,60)), np.concatenate([id_train,id_valid],axis=0) )


#%% CREATE DATASETS
if params.form_dataset == "no" :
    x_train, x_valid, y_train, y_valid, x_test, y_test = load_data(params)
else :
    x_train, x_valid, y_train, y_valid, x_test, y_test = formation_dataset(params, id_train, id_test, id_valid)

print("x_train :", np.shape(x_train))
print("x_valid :", np.shape(x_valid))
print("x_test :", np.shape(x_test))


#%% CREATE MODEL
if params.nb_branch == 1:
    print("CNN_1Branch")
    params.drop_rate = float(params.drop_rate[0])
    params.nb_filters = int(params.nb_filters[0])
    model = CNN_1Branch(nb_filters=params.nb_filters, drop_rate=params.drop_rate)
elif params.nb_branch == 2 :
    print("CNN_2Branch")
    drop_rate_1 = float(params.drop_rate[0])
    drop_rate_2 = float(params.drop_rate[1])
    nb_filters_1 = int(params.nb_filters[0])
    nb_filters_2 = int(params.nb_filters[1])
    model = CNN_2Branch(nb_filters_1, nb_filters_2, drop_rate_1, drop_rate_2, params)
else:
    print("CNN_3Branch")
    drop_rate_1 = float(params.drop_rate[0])
    drop_rate_2 = float(params.drop_rate[1])
    drop_rate_3 = float(params.drop_rate[2])
    nb_filters_1 = int(params.nb_filters[0])
    nb_filters_2 = int(params.nb_filters[1])
    nb_filters_3 = int(params.nb_filters[2])
    model = CNN_3Branch(nb_filters_1, nb_filters_2, nb_filters_3, drop_rate_1, drop_rate_2, drop_rate_3, params)
t1 = time()


#%% TRAINING AND VALIDATION
if params.nb_branch == 2 :
    x_train = [ x_train[:,:,:,:params.nb_bands[0]], x_train[:,:,:,params.nb_bands[0]:] ]
    x_valid = [ x_valid[:,:,:,:params.nb_bands[0]], x_valid[:,:,:,params.nb_bands[0]:] ]
    x_test = [ x_test[:,:,:,:params.nb_bands[0]], x_test[:,:,:,params.nb_bands[0]:] ]
elif params.nb_branch == 3 :
    lim = params.nb_bands[0] + params.nb_bands[1]
    x_train = [x_train[:,:,:,:params.nb_bands[0]], x_train[:,:,:,params.nb_bands[0]:lim], x_train[:,:,:,-params.nb_bands[2]:]]
    x_valid = [x_valid[:,:,:,:params.nb_bands[0]], x_valid[:,:,:,params.nb_bands[0]:lim], x_valid[:,:,:,-params.nb_bands[2]:]]
    x_test = [x_test[:,:,:,:params.nb_bands[0]], x_test[:,:,:,params.nb_bands[0]:lim], x_test[:,:,:,-params.nb_bands[2]:]]


run(model, x_train, y_train, x_valid, y_valid, x_test, y_test, params)
del x_train, x_valid, y_train, y_valid
t2 = time()


#%% CREATE MAP
create_map(model, params)
#f1 = evaluation(model, params)

t3 = time()
print("\ndurée totale (en min) :", (t3-t0)/60)
print("\ndurée formation dataset (en min) :", (t1-t0)/60)
print("\ndurée training (en min) :", (t2-t1)/60)
print("\ndurée create map (en min) :", (t3-t2)/60)
