import numpy as np
from chandra_aca import transform
import collections
import numpy.ma as ma
from astropy.table import Column
# local imports:
import centroids as cntr


def simulate_aca_l0(img_size, hot_pixels=None, nframes=1000, delta_t=4.1, integ=1.696, bgdavg=None,\
                    fwhm=1.8, mag=10.2, delta=-0.25,  ampl=8., period=1000., phase=0.):
    
    times = np.arange(nframes) * delta_t

    row0, col0, yaw, pitch = get_row0col0_yawpitch(times, ampl=ampl, period=period,
                                                   phase=phase, delta=delta)

    roff = yaw - row0 # row offset = difference between yaw and its integerized value (row0)
    coff = pitch - col0

    data = []
    img_size2 = img_size * img_size
    
    for i, time in enumerate(times):
        imgraw = simulate_star(fwhm, mag, integ, bgd=bgdavg, roff=roff[i], coff=coff[i]) # 8x8
        
        # add hot pixels if defined and if they fit in the current 8x8 window
        if hot_pixels is not None:
            for key, val in hot_pixels.items():
                rr = key[0] - row0[i]
                cc = key[1] - col0[i]
                if rr in range(8) and cc in range(8):
                    imgraw[rr, cc] = imgraw[rr, cc] + val        
        
        imgraw = imgraw.reshape(1, img_size2)[0]
        mask = img_size2 * [0]
        fill_value = 1.e20
        imgraw = ma.array(data=imgraw, mask=mask, fill_value=fill_value)
        
        data_row = (time, imgraw, row0[i], col0[i], bgdavg, img_size)
        data.append(data_row)
        
    data = np.ma.array(data, dtype=[('TIME', '>f8'), ('IMGRAW', '>f4', (64,)),
                                    ('IMGROW0', '>i2'), ('IMGCOL0', '>i2'),
                                    ('BGDAVG', '>i2'), ('IMGSIZE', '>i4')])

    true_centroids = np.array([4. + roff, 4 + coff]) # check this, 6x6?
    
    return data, true_centroids


def get_row0col0_yawpitch(times, ampl=8., period=1000., phase=0., delta=-0.25):

    # What's the meaning of delta?
    
    pxsize = 5. # arcsec per pixel

    ampl_yaw, ampl_pitch = do_assignment(ampl) # check if ampl is a number or an iterable
    period_yaw, period_pitch = do_assignment(period)
    phase_yaw, phase_pitch = do_assignment(phase)

    yaw = ampl_yaw / pxsize * np.sin(2 * np.pi * times / period_yaw + 2 * np.pi * phase_yaw) # ~row0    
    pitch = ampl_pitch / pxsize * np.sin(2 * np.pi * times / period_pitch + 2 * np.pi * phase_pitch) # ~col0
    
    row0 = np.array(np.round(yaw + delta), dtype=np.int)
    col0 = np.array(np.round(pitch + delta), dtype=np.int)
    return (row0, col0, yaw, pitch)


def do_assignment(val):
    if isinstance(val, collections.Iterable):
        if len(val) == 2 and all([isinstance(i, (int, float)) for i in val]):
            val1, val2 = val
        else:
            raise TypeError('do_assignment:: Expecting an iterable with 2 numeric elements')
    elif isinstance(val, (int, float)):
        val1 = val2 = val
    else:
        raise TypeError('do_assignemnt:: Expecting value to be a number or an iterable')
    return (val1, val2)


# 2-d Gaussian star with magnitude mag and FWHM, noise: gaussian
def simulate_star(fwhm, mag, integ, bgd=None, roff=0, coff=0):
    
    img_size = 8
    img_size2 = img_size * img_size
        
    if np.shape(bgd) not in [(), (img_size, img_size), (img_size2,)]:
        raise ValueError('bgd expected to be int, float or (8, 8) or (64,) array')
    
    star = np.zeros((img_size, img_size))

    # Mag to counts conversion
    gain = 5. # e-/ADU
    counts = integ * transform.mag_to_count_rate(mag) / gain

    # Gaussian model
    halfsize = np.int(img_size / 2)
    row, col = np.mgrid[-halfsize:halfsize, -halfsize:halfsize] + 0.5
    sigma = fwhm / (2. * np.sqrt(2. * np.log(2.)))
    g = np.exp(-((row - roff)**2  / sigma**2 + (col - coff)**2 / sigma**2) / 2.)
    
    # Zero 6x6 corners
    g = cntr.zero_6x6_corners(g, centered=True)
    
    # Normalize to counts
    i1 = np.int(halfsize + 0.5 - 3)
    i2 = np.int(halfsize + 0.5 + 3)
    g = counts * g / g[i1:i2][i1:i2].sum()    

    # Simulate star
    star = np.random.normal(g)
        
    # Add background
    if np.shape(bgd) == ():
        bgd = np.ones((img_size, img_size)) * bgd

    
    #if np.shape(bgd) == img_size2:
    #    bgd = bgd.reshape(img_size, img_size)
    #    
    ## bgd < 0 -> set equal to 0
    #bgd = bgd * (bgd > 0)
    #bgd = cntr.zero_6x6_corners(bgd, centered=True)
    
    star = star + bgd
    
    return np.rint(star)


# Simulate hot pixels - I am not using this yet
def simulate_hot_pixels(times, hp_number, hp_val):

    row0, col0 = get_row0_col0(t_m['time'][0]) # this won't work now
    img_size = 8
    row_min = row0.min()
    row_max = img_size + row0.max()
    col_min = col0.min()
    col_max = img_size + col0.max()

    #print row_min, row_max, col_min, col_max

    np.random.seed(42)
    hp_rows = np.random.randint(row_min, row_max, size=hp_number)
    np.random.seed(24)
    hp_cols = np.random.randint(col_min, col_max, size=hp_number)

    hot_pixels = {}

    for rr, cc in zip(hp_rows, hp_cols):
        hot_pixels[(rr, cc)] = hp_val

    return hot_pixels


def dither_acis():
    ampl = 8. # arcsec
    yaw_period = 1000.0 # sec
    pitch_period = 707.1 # sec
    period = [yaw_period, pitch_period]
    return (ampl, period)


def dither_hrc():
    ampl = 20. # arcsec
    yaw_period = 1087.0 # sec
    pitch_period = 768.6 # sec
    period = [yaw_period, pitch_period]
    return (ampl, period)
