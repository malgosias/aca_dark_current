import numpy as np
from mica.archive import aca_l0
from chandra_aca import transform
from copy import deepcopy
import numpy.ma as ma
from astropy.table import Table
# local imports:
#import sim_aca_l0 as siml0
import classes
from classes import *


def centroids(slot, slot_data, img_size, bgd_object, nframes=1000):
    # Calls:
    #     get_centroids
   
    rows = []
    
    #nframes = kwargs['nframes']

    print('Slot = {}'.format(slot))

    slot_row = {}

    rowcol_cntrds, yagzag_cntrds, bgd_imgs, deque_dicts = \
                                get_centroids(slot_data, img_size, bgd_object, nframes=nframes)

    slot_data = Table(slot_data)
        
    slot_row['slot'] = slot        
    slot_row['time'] = slot_data['TIME'][:nframes]
    slot_row['row0'] = slot_data['IMGROW0'][:nframes]
    slot_row['col0'] = slot_data['IMGCOL0'][:nframes]
    slot_row['imgraw'] = slot_data['IMGRAW'][:nframes] # 64
    slot_row['bgdimg'] = bgd_imgs # 8x8
    slot_row['deque_dict'] = deque_dicts
        
    raw_yagzag = np.array(yagzag_cntrds).T[0] # [0] - yag, [1] - zag
    
    slot_row['yan'] = raw_yagzag[0]
    slot_row['zan'] = raw_yagzag[1]

    raw_rowcol = np.array(rowcol_cntrds).T # [0] - row, [1] - col, 1:7
    
    slot_row['row'] = raw_rowcol[0]
    slot_row['col'] = raw_rowcol[1]
            
    rows.append(slot_row)
    
    return rows


def get_centroids(slot_data, img_size, bgd_object, nframes=None):
    # For each frame:
    # 1. Compute and subtract background image
    #     a. get_background method of a bgd_object
    #     b. Compute bgd_img, algorithm depends on the type of bgd_object
    #         - Standard_Bgd: avg background (current algorithm)
    #         - DarkCurrent_Median_Bgd: median for sampled pixels, avg bgd for not sampled pixels
    #         - DarkCurrent_SigmaClip_Bgd: sigma clipping for sampled pixels, avg bgd for not sampled pixels
    # 2. Compute centroids in image coordinates 0:8 (variable name: row/col)
    # 3. Transform to get yagzag coordinates
    #
    # Calls:
    #     get_current_centroids
    #     bgd_object.get_background
        
    if img_size not in [6, 8]:
        raise ValueError('get_centroids:: expected img_size = 6 or 8')
        
    if img_size == 8:
        img_mask = get_mask_8x8_centered()
    else:
        # img_mask = None for science observations  # check for HRC?
        img_mask = None
    
    if nframes is None:
        nframes = len(slot_data)
   
    yagzag_centroids = [] # change for {}
    rowcol_centroids = []
    bgd_imgs = []
    deque_dicts = []

    for index in range(0, nframes):
        
        frame_data = slot_data[index:index + 1]

        bgd_object.bgdavg = frame_data['BGDAVG'][0]

        if isinstance(bgd_object, (DynamBgd_Median, DynamBgd_SigmaClip)):
            bgd_object.img = frame_data['IMGRAW'][0]
            bgd_object.row0 = frame_data['IMGROW0'][0]
            bgd_object.col0 = frame_data['IMGCOL0'][0]
            
        bgd_img = bgd_object.get_background() # 8x8
        
        bgd_imgs.append(bgd_img)

        if isinstance(bgd_object, (DynamBgd_Median, DynamBgd_SigmaClip)):
            deque_dict = bgd_object.deque_dict
            deque_dicts.append(deepcopy(deque_dict))

        raw_img = frame_data['IMGRAW'][0].reshape(8, 8)
        
        img = raw_img - bgd_img
        
        # For dark_current, don't oversubtract? px with bgd > raw val will be set to zero
        if isinstance(bgd_object, (DynamBgd_Median, DynamBgd_SigmaClip)):
            bgd_mask = bgd_img > raw_img
            img = raw_img - ma.array(bgd_img, mask=bgd_mask)
            img = img.data * ~bgd_mask

        if img_mask is not None:
            img = ma.array(img, mask=img_mask) # mask before computing centroids
        
        # Calculate centroids for current bgd subtracted img, use first moments
        rowcol = get_current_centroids(img, img_size)
        rowcol_centroids.append(rowcol)

        # Translate (row, column) centroid to (yag, zag)
        y_pixel = rowcol[0] + frame_data['IMGROW0']
        z_pixel = rowcol[1] + frame_data['IMGCOL0']            
        yagzag = transform.pixels_to_yagzag(y_pixel, z_pixel)

        yagzag_centroids.append(yagzag)
        
    return rowcol_centroids, yagzag_centroids, bgd_imgs, deque_dicts


def get_current_centroids(img, img_size):

    num = np.arange(0.5, 6.5)

    if (img_size == 8):
        # ER observations
        img = zero_6x6_corners(img, centered=True)
    else:
        # Science observations
        img = zero_6x6_corners(img, centered=False)
    
    # Use first moments to find centroids
    centroids = []
    for ax in [1, 0]: # [row, col]
        # Def of flat is where img_mask becomes relevant for ER data
        flat = np.sum(img, axis=ax)
        if (img_size == 6):            
            centroid = np.sum(flat[:-2] * num) / np.sum(flat[:-2]) # 0:6
        else:
            # 1:7, is +1 relevant? yes, if row0/col0 always the lower left pixel in 8x8
            centroid = np.sum(flat[1:-1] * num) / np.sum(flat[1:-1]) + 1 # 1:7
        centroids.append(centroid)
        
    return centroids


def zero_6x6_corners(img, centered=True): # img is a 8x8 array
    if not img.shape == (8, 8):
        raise ValueError("Img should be a 8x8 array")
    if centered:
        r4 = [1, 1, 6, 6]
        c4 = [1, 6, 1, 6]
    else:
        r4 = [0, 0, 5, 5]
        c4 = [0, 5, 0, 5]
    for rr, cc in zip(r4, c4):
        img[rr][cc] = 0.0
    return img


# In 8x8 img, mask the edge pixels, leave r/c 1:7 unmasked.
# For science observations the raw image is masked by default (r/c 0:6 are left unmasked).
def get_mask_8x8_centered():
    m = """\
        1 1 1 1 1 1 1 1
        1 0 0 0 0 0 0 1
        1 0 0 0 0 0 0 1
        1 0 0 0 0 0 0 1
        1 0 0 0 0 0 0 1
        1 0 0 0 0 0 0 1
        1 0 0 0 0 0 0 1
        1 1 1 1 1 1 1 1"""
    mask = np.array([line.split() for line in m.splitlines()], dtype=float)
    return mask