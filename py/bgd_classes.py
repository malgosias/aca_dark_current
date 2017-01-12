# Background image always 8x8 ?
import numpy as np
import collections


# Standard background, current algorithm
class StandardBgd(object):
    
    def __init__(self, bgdavg=0):
        self.img_size = 8
        self.bgdavg = bgdavg
        
    
    def __repr__(self):
        return ('<{} img_size={} bgdavg={}>'
                .format(self.__class__.__name__, self.img_size, self.bgdavg))
    
    def get_background(self):
        return np.ones((self.img_size, self.img_size)) * self.bgdavg # 8x8

    
#   Non-Standard Background:
#
#   For each image, pick out the 'X's
#
#   (currently: option *8x8*, *6x6*, redefine r, c in get_centroids to compute **8x8**)
#
#    *8x8*              *6x6*
#
#      (1)--->
#  (3) XXXXXXXX (4)       .XXXX.
#   |  X......X  |        X....X
#   |/ X......X  |/       X....X
#      X......X           X....X
#      X......X           X....X
#      X......X           .XXXX.
#      X......X
#      XXXXXXXX
#      (2)--->
#
# OR
#
#    **8x8**
#
#      (1)--->
#  (3) XXXXXXXX (4)
#   |  XXXXXXXX  |
#   |/ XX....XX  |/
#      XX....XX
#      XX....XX
#      XX....XX
#      XXXXXXXX
#      XXXXXXXX
#      (2)--->


class DynamBgd_Median(object):

    def __init__(self, img_size, ndeque=1000, deque_dict=None, img=None, row0=0, col0=0, bgdavg=0):

        self.img_size = img_size
        self.ndeque = ndeque
        self.deque_dict = deque_dict
        if self.deque_dict is None:
            self.deque_dict = collections.OrderedDict()
        self.img = img
        self.row0 = row0
        self.col0 = col0
        self.bgdavg = bgdavg
        
        if self.img_size == 8:
            self.r = [0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 7,
                      1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]
            self.c = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7,
                      0, 0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 7]
        else:
            self.r = [0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 1, 2, 3, 4, 1, 2, 3, 4]
            self.c = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 0, 0, 0, 5, 5, 5, 5]

        self.edge_descritpion = '1px outmost edge'
            
            
    def __repr__(self):
        return ('<{} img_size={} ndeque={} row0={} col0={} bgdavg={} edge_description={}>'
                .format(self.__class__.__name__, self.img_size, self.ndeque,
                        self.row0, self.col0, self.bgdavg, self.edge_descritpion))


    def get_background(self):

        if self.img is None:
            raise ValueError("DarkCurrent_Median_Bgd:: Can't compute background, img is None")
        
        bgd_img = np.zeros((8, 8))
        
        self.deque_dict = self.update_deque_dict()
        
        for rr in range(self.img_size):
            for cc in range(self.img_size):
                key = (rr + self.row0, cc + self.col0)
                if key in self.deque_dict.keys():
                    bgd_img[rr, cc] = self._get_value(key)
                else:
                    bgd_img[rr, cc] = self.bgdavg

        return bgd_img  # 8x8
    
        
    # Store values of sampled background pixels in a dictionary of deques.
    # Keys are absolute pixel coordinates.
    # Store up to ndeque values.
    def update_deque_dict(self):
        # Update deque_dict:
        # 1. Compute current coordinates of the edge pixels, coord values are in -512:511
        # 2. Append or init deque with edge value if we are on the edge.
        # 3. If > ndeque elements in a deque, pop the first one (popleft)
    
        # current edge row/col coords in -512:511
        r_current_edge = self.r + self.row0 * np.ones(len(self.r))
        c_current_edge = self.c + self.col0 * np.ones(len(self.c))
    
        edge_vals = self.edge_pixel_vals()
        deque_dict = self.deque_dict

        for (rr, cc) in zip(r_current_edge, c_current_edge):
            val = edge_vals[(rr, cc)]
            if (rr, cc) in deque_dict.keys():
                deque_dict[(rr, cc)].append(val)
                # Keep the length at ndeque
                if len(deque_dict[(rr, cc)]) > self.ndeque:
                    deque_dict[(rr, cc)].popleft()
            else: # initialize
                deque_dict[(rr, cc)] = collections.deque([val])
        return deque_dict
    

    # Pick out the edge pixels
    def edge_pixel_vals(self):

        vals = collections.OrderedDict()
    
        self.img = self.img.reshape(8, 8)
    
        for rr, cc in zip(self.r, self.c): # (r, c) define location of edge pixels (X's)
            r_abs = self.row0 + rr
            c_abs = self.col0 + cc
            key = (r_abs, c_abs) # e.g. (210, 124), a tuple
            vals[key] = self.img[rr, cc]
            
        return vals

    
    def _get_value(self, key):
        return np.median(self.deque_dict[key])
    
    
# Dynamic background, sigma clipping of ndeque values
class DynamBgd_SigmaClip(DynamBgd_Median):

    def _get_value(self, key):
        deque = self.deque_dict[key]
        if len(deque) > 2:
            d_min = np.argmin(deque)
            d_max = np.argmax(deque)
            m = np.zeros(len(deque))
            m[d_min] = 1
            m[d_max] = 1
            deque = np.ma.array(deque, mask=m)
        return np.mean(deque)
