import numpy as np
from skimage.transform import rescale, SimilarityTransform, warp
from skimage.morphology import disk, square

def disk_in_center(array, r):
    # if r<1 use it as percentage of maximum dimension
    if r<.5:
        r = np.floor(np.max(array.shape)*r).astype(int)
    elif r>=.5 and r<1:
        print('r must be <.5 or >=1')        
    else:
        r = int(r)
    d = disk(r, dtype=array.dtype)
    array[array.shape[0]//2 - d.shape[0]//2:array.shape[0]//2 + d.shape[0]//2+1,\
          array.shape[1]//2 - d.shape[1]//2:array.shape[1]//2 + d.shape[1]//2+1] = d
    return array

def square_in_center(array, r):
    if r<.5:
        r = np.floor(np.max(array.shape)*r).astype(int)
    elif r>=.5 and r<1:
        print('r must be <.5 or >=1')        
    else:
        r = int(r)
    d = square(r*2, dtype=array.dtype)
    array[array.shape[0]//2 - d.shape[0]//2:array.shape[0]//2 + d.shape[0]//2,\
          array.shape[1]//2 - d.shape[1]//2:array.shape[1]//2 + d.shape[1]//2] = d
    return array

def add_center_stop(array, r, shift = 0):
    # if r<1 use it as percentage of maximum dimension
    center_stop = np.zeros_like(array)
    if r<=1:
        r = np.floor(np.max(array.shape)*r/2).astype(int)
#     elif r>=.5 and r<1:
#         print('r must be <.5 or >=1')        
    else:
        r = int(r)
    d = disk(r, dtype=array.dtype)
    center_stop[center_stop.shape[0]//2 - d.shape[0]//2:center_stop.shape[0]//2 + d.shape[0]//2+1,\
          center_stop.shape[1]//2 - d.shape[1]//2:center_stop.shape[1]//2 + d.shape[1]//2+1] = d
    tform = SimilarityTransform(translation=(-shift, 0))
    center_stop = warp(center_stop, tform) #np.roll(center_stop, shift, axis = 1)
    array[center_stop.astype(bool)] = 0
    return array

def sq_diagonal(array, which_diagonal = 0):
    # Returns 1D array of diagonal entries in square array.
    diagonal = []
    for i in range(array.shape[0]):
        diagonal.append(array[np.abs(which_diagonal*array.shape[0]-i),i])
    return diagonal

def create_aperture(aperture_radius, array_size, downscale_factor, b = 0, shift = 0):
    # Creat edge smoothed aperture via rescaling high res aperture
    aperture = disk(int(aperture_radius*downscale_factor), dtype='double')
    if b > 0:
        aperture = add_center_stop(aperture, r=b, shift = shift)
    aperture = np.pad(aperture, pad_width = int(aperture_radius*downscale_factor))
    aperture = rescale(aperture, scale = 1/downscale_factor)
    # Insert in to array of zeros.
    a = np.zeros((array_size, array_size), dtype = 'float32')
    a[array_size//2-aperture.shape[0]//2:array_size//2+aperture.shape[0]//2,
      array_size//2-aperture.shape[0]//2:array_size//2+aperture.shape[0]//2] = aperture
    return a

# psf of circular aperture
psf = lambda o,v: (2*jv(o,v)/v)**2 

# psf of circular aperture with center stop
psf_c = lambda o,v,b: (1/(1-b**2))**2*\
                      (2*jv(o,v)/v-b**2*2*jv(o,v*b)/(v*b))**2