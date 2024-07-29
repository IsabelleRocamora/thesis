import numpy as np

def getBatch (images, i, batch_size):
    """ 
    Returns the batch to be processed
    """
    start_id = i * batch_size
    end_id = min ( (i+1) * batch_size, images.shape[0] )
    batch_images = images[start_id : end_id]
    return batch_images

def normalize(data, infos):
    """
    Gets the type of normalization from the dictionary containing 
    pre-processing info and extracts min and max values accordingly
    to finally proceed to data normalization
    """
    # Gets min and max value
    if infos[-2] == "with_nodata":
        max_data = np.amax(data)
        no_data = np.amin(data)
        min_data = np.amin(data[np.where(data!=no_data)])
        
    elif infos[-2] == "normale" :
        max_data = np.amax(data)
        min_data = np.amin(data)
        
    elif infos[-2] == "specific":
        max_data = infos[2]
        min_data = infos[1]
        
    else :
        print("error : invalid type's normalization -->", infos[-2])

    # Normalize data
    data = (data - min_data) / (max_data - min_data)
    return data

    