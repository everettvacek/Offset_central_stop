import numpy as np
from skimage.transform import rescale, SimilarityTransform, warp
from skimage.morphology import disk

def add_center_stop(array, r, shift = 0):
    # if r<1 use it as percentage of maximum dimension
    center_stop = np.zeros_like(array)
    if r<=1:
        r = np.floor(np.max(array.shape)*r/2).astype(int)
    else:
        r = int(r)
    d = disk(r, dtype=array.dtype)
    center_stop[center_stop.shape[0]//2 - d.shape[0]//2:center_stop.shape[0]//2 + d.shape[0]//2+1,\
          center_stop.shape[1]//2 - d.shape[1]//2:center_stop.shape[1]//2 + d.shape[1]//2+1] = d
    tform = SimilarityTransform(translation=(-shift, 0))
    center_stop = warp(center_stop, tform)
    array[center_stop.astype(bool)] = 0
    return array

def create_aperture(aperture_radius, array_size, downscale_factor, b = 0, shift = 0):
    # Creat edge smoothed aperture via rescaling high res aperture.
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