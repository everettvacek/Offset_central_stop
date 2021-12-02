import numpy as np
from skimage.transform import rescale, SimilarityTransform, warp
from skimage.morphology import disk
from scipy.special import jv

def add_center_stop(array, b, shift = 0):
    '''
    array: aperture array.
    b: fractional radius of central stop.
    shift: shift in pixels from center.
    '''
    center_stop = np.zeros_like(array)
    b = np.floor(np.max(array.shape)*b/2).astype(int)
    d = disk(b, dtype=array.dtype)
    center_stop[center_stop.shape[0]//2 - d.shape[0]//2:center_stop.shape[0]//2 + d.shape[0]//2+1,\
          center_stop.shape[1]//2 - d.shape[1]//2:center_stop.shape[1]//2 + d.shape[1]//2+1] = d
    tform = SimilarityTransform(translation=(-shift, 0))
    center_stop = warp(center_stop, tform)
    array[center_stop.astype(bool)] = 0
    return array

def create_aperture(aperture_radius, array_size, downscale_factor, b = 0, shift = 0):
    '''
    aperture_radius: final aperture radius in pixels (after downscaling).
    array_size: size of final array after embedding downscaled aperture array.
    downscale_factor: integer downscaling of aperture_radius before embedding into final array.
    b: fractional radius of central stop.
    shift: shift in pixels from center.
    '''
    # Creat edge smoothed aperture via rescaling high res aperture.
    aperture = disk(int(aperture_radius*downscale_factor), dtype='double')
    if b != 0:
        # add central stop
        aperture = add_center_stop(aperture, b=b, shift = shift)
    aperture = np.pad(aperture, pad_width = int(aperture_radius*downscale_factor))
    aperture = rescale(aperture, scale = 1/downscale_factor)
    # Insert in to array of zeros.
    a = np.zeros((array_size, array_size), dtype = 'float32')
    a[array_size//2-aperture.shape[0]//2:array_size//2+aperture.shape[0]//2,
      array_size//2-aperture.shape[0]//2:array_size//2+aperture.shape[0]//2] = aperture
    return a

def psf(v, b = None):
    # Bessel function calculation of 1D psf.
    '''
    v: (i.e. nu) 1-D array for domain of the psf.
    b: fractional radius of central stop.
    '''
    if b is None or b is 0:
        # psf of circular aperture
        return (2*jv(1,v)/v)**2
    else:
        # psf of circular aperture with center stop
        return (1/(1-b**2))**2*\
            (2*jv(1,v)/v-b**2*2*jv(1,v*b)/(v*b))**2