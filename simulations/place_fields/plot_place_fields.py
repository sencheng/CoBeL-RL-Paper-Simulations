#generate plot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
import scipy.ndimage as nd
import math
import os
from scipy.spatial.distance import pdist

def classify_place_field(activity_map, smooth=True, resolution=(25,25),
                         size_threshold=0.50, cutoff=0.25) :
    """ Decide whether or not a given spatial activity map contains place fields.
    
    Keyword arguments : 
    activity_map : 2d numpy array of spatial activations
    smooth : apply gaussian smoothing to field
    resolution : resolution of input data
    size_threshold : maximum size of field (proportion of arena) 
    cutoff : proportion of maximum activity level below which to remove activity
    """
    is_field = True
    #find clusters where activity falls off below threshold
    #and calculate a boolean mask of clusters
    activity_map = nd.gaussian_filter(activity_map,sigma=2)
    activity_map = np.where(activity_map<cutoff*np.max(activity_map),0,
                            activity_map)
    mask = activity_map > 0
    
    #label blobs
    labels, nb = nd.label(mask)

    if nb != 0 :
        #find largest cluster
        sizes = []
        for i in range(nb+1) :
            slice_id = nd.find_objects(labels==i)
            #sizes.append(np.shape(activity_map[slice_id[0]]))
            if not slice_id :
                sizes.append(resolution)
            else : sizes.append(np.shape(activity_map[slice_id[0]]))
        pixels = np.prod(sizes, axis=1)
        l = np.argmax(pixels[1:]) + 1        
        if smooth : 
            field = nd.gaussian_filter(np.where(labels==l,activity_map,0),sigma=2)
    
    else :
        field = labels

    if nb == 0 :
        is_field=False
        print("no blobs found -- not place like")
    elif nb > 4 : 
        is_field=False
        print("too many blobs -- not place like")
    elif pixels[l] > size_threshold*pixels[0] :
        non_zero = np.count_nonzero(field)
        fraction_active = non_zero/(np.product(resolution))
        is_field=False
        print("too large, not place like : ", fraction_active)
    
    return is_field, field

def calculate_pdist(fields_sliced, place_indices) :
    """ Calculate pairwise distances between field centers for specified fields.
    
    Keyword arguments :
    fields_sliced : fields for which the pairwise distances must be calculated
    place_indices : (legacy, remove)
    """
    print(place_indices,place_indices.size)
    if place_indices.size != 1 :
        centers = np.zeros((len(place_indices),6,2))
    else : 
        centers = np.zeros((1,6,2))
    for f,i in zip(fields_sliced,range(len(fields_sliced))) :
        for angle,j in zip(f,range(len(f))) :
            centers[i,j,:] = divmod((np.argmax(angle)),25)
    pdist_centers = [pdist(centers[i]) for i in range(len(centers))]
    angles    = np.deg2rad(np.arange(0,360,60,dtype='int32').reshape(-1,1))
    vectors   = np.squeeze(np.array([(np.cos(a),np.sin(a)) for a in angles]))
    return pdist_centers, pdist(vectors)

def place_like(fields) : 
    
    n_units = fields.shape[2]
    mean_fields = np.mean(fields,axis=0).reshape((1,625,n_units))
    stacked = np.vstack((fields,mean_fields))
    
    is_field = np.zeros((7,n_units))
    largest_field  = np.zeros((7,625,n_units))
    print("Place")
    for f,i in zip(np.rollaxis(stacked,2),range(n_units)) :
        print("Unit",i)
        for hd,j in zip(f,range(7)) :
            print("Head direction",j)
            hd = hd.reshape(25,25)
            isit, big = classify_place_field(hd)
            is_field[j,i] = isit
            largest_field[j,:,i] = big.flatten()
            
    #if all fields and the mean are classified, it may be a place like representation
    
    a = np.all(is_field, axis=0)
    ids = np.squeeze(np.array(np.where(a))) #ids where fields are place like
    if ids.shape != () : 
        f = np.array([largest_field[:,:,p] for p in ids])
    else :
        f = np.array([largest_field[:,:,ids]])
    
    if ids.size > 0 :
        pd,v = calculate_pdist(f[:,:6,:],ids)
    else :
        print("no slice found--",ids )
        pd =  [15*np.ones((15,))]
        v =  [15*np.ones((15,))]
    #if there is too much directional modulation, not field
    x_all = np.all(np.array(pd) < 10, axis=1)
    ids = np.where(x_all,ids,-1)
    ids = ids[ids != -1] 
    f = np.array([largest_field[:,:,p] for p in ids])
    if ids.size > 0 :
        pd,v = calculate_pdist(f[:,:6,:],ids)
    else :
        print("no slice found--",ids )
        pd =  [15*np.ones((15,))]
    
    return f,ids,pd,v

    
if __name__ == '__main__' :
    
    data_path = 'data/activations/'
    activation_files = [file for file in os.listdir(data_path)]
    n_place_fields = []
    for a in activation_files : 
        place_fields_run = []
        with open(data_path+a, 'rb') as fp:
            activations = pickle.load(fp)
            for a in activations : 
                f, ids, pd, v = place_like(np.squeeze(np.array(a)))
                place_fields_run.append(len(ids))
        n_place_fields.append(place_fields_run)
    
    n_place_fields = np.array(n_place_fields)
    means = np.mean(n_place_fields, axis=0)
    
    
    trials = np.arange(0,6500,800)

    tick_spacing = 1600
    fig, ax = plt.subplots(figsize=(3,2),dpi=500)
    ax.plot(trials,means,marker='^', c='black')
    ax.set_xlim(0, 6400)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.set_xlabel('Trial')
    ax.set_ylabel('Place Fields [#]')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig('cobel.png', dpi=200, bbox_inches='tight', transparent=True)
    plt.savefig('cobel.svg', dpi=200, bbox_inches='tight', transparent=True)
    