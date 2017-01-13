# Plotting functions
import matplotlib.pylab as plt
import numpy as np
from Ska.Matplotlib import plot_cxctime
import Ska.Numpy

methods = ['StandardBgd', 'DynamBgd_Median', 'DynamBgd_SigmaClip']

def plot_d_ang(slot, slot_ref, key, dt, table):
    # plot delta yan(or zan)
    #ylim = [(76, 82), (2272, 2278)]

    ok1 = table['slot'] == slot
    ok2 = table['slot'] == slot_ref
    
    fig = plt.figure(figsize=(8, 2.5))
    
    for i, method in enumerate(methods):
        plt.subplot(1, 3, i + 1)
        ok = table['bgd_type'] == method
        ang_interp = Ska.Numpy.interpolate(table[ok * ok1][key][0], table[ok * ok1]['time'][0] + dt,
                                           table[ok * ok2]['time'][0],
                                           method="nearest")
        d_ang = table[ok * ok2][key][0] - ang_interp
        plt.plot(table[ok * ok2]['time'][0], d_ang, color='Darkorange',
                     label='std = {:.5f}'.format(np.std(d_ang - np.median(d_ang))))
        plt.ylim((np.median(d_ang) - 1., np.median(d_ang) + 1.))
        plt.xlabel('Time (sec)')
        plt.ylabel('delta {} (arcsec)'.format(key))
        plt.title('Slot {} - Slot {}, {}'.format(slot_ref, slot, method))
        plt.legend()
        plt.margins(0.05)

    plt.subplots_adjust(left=0.05, bottom=0.2, right=0.99, top=0.9, hspace=0.3, wspace=0.3)
    return


def plot_bgd_px_series(bgd_imgs, key):
    
    if keys is None:
        keys =  t[slot]['bgd_px_series'].keys()

    fig = plt.figure(figsize=(10, 3))

    for i, key in enumerate(keys):
        if i % 2 == 0:
            key_time = (-key[0], -key[1])
            plt.plot(t[slot]['bgd_px_series'][key_time], t[slot]['bgd_px_series'][key], 'b.')
        plt.title("Time series of " + text)
        plt.xlabel('Time (sec)')
        plt.ylabel('Bgd pixel value')

    return


def plot_coords_excess(slot, table, coord):
    fig = plt.figure(figsize=(10, 6))
    color = ['green', 'red', 'blue']
    
    for i, method in enumerate(methods):
        #print '{:.2f}'.format(np.median(t[coord][slot]))
        ok = table['bgd_type'] == method
        excess = table[ok][coord][slot] - table[ok]["true_" + coord][slot]
        std = np.std(excess - np.median(excess))
        plt.plot(table['time'][slot], excess, color=color[i], label="std = {:.3f}, ".format(std) + method)
        plt.margins(0.05)
    
    plt.ylabel(coord + " - true " + coord)
    plt.xlabel("Time (sec)")
    plt.title("Difference between derived " + coord + " and true " + coord + " coordinates.");
    plt.grid()
    plt.legend()
    return


def plot_coords(slot, table, coord):
    fig = plt.figure(figsize=(10, 6))
    color = ['green', 'red', 'blue']
    
    for i, method in enumerate(methods):
        #print '{:.2f}'.format(np.median(t[coord][slot]))
        ok = table['bgd_type'] == method
        ok1 = table['slot'] == slot
        plt.plot(table[ok * ok1]['time'][0], table[ok * ok1][coord][0], color=color[i], label=method)
        plt.margins(0.05)

    text = ""
    if "true_" + coord in table.colnames:
        plt.plot(table['time'][slot], table[ok]["true_" + coord][slot], '--', color='k', lw='2', label="True")
        text = " and true " + coord + " coordinates."
    plt.ylabel(coord)
    plt.xlabel("Time (sec)")
    plt.title("Derived " + coord + text);
    plt.grid()
    plt.legend()
    return


def plot_star_image(data):
    c32 = np.array([1, 2, 2, 6, 6, 7, 7, 6, 6, 2, 2, 1, 1]) - 0.5
    r32 = np.array([2, 2, 1, 1, 2, 2, 6, 6, 7, 7, 6, 6, 2]) - 0.5

    c6x6 = [0.5, 6.5, 6.5, 0.5, 0.5]
    r6x6 = [0.5, 0.5, 6.5, 6.5, 0.5]
    
    plt.imshow(data, cmap=plt.get_cmap('jet'), interpolation='none', origin='lower')
    #plt.colorbar()
    plt.plot(c32, r32, '--', lw=2, color='w')
    plt.plot(c6x6, r6x6 , '-', lw=2, color='w')
    plt.xlim(-0.5, 7.5)
    plt.ylim(-0.5, 7.5)
    return


def plot_bgd_image(img, img_number, row0, col0, img_size):
    
    r_min = row0.min()
    r_max = row0.max() + img_size
    c_min = col0.min()
    c_max = col0.max() + img_size
    
    data = -100 * np.ones((c_max - c_min, r_max - r_min))

    dr = row0[img_number] - r_min
    dc = col0[img_number] - c_min
        
    data[dr:dr + img_size, dc:dc + img_size] = img[:img_size, :img_size]

    plt.imshow(data, cmap=plt.get_cmap('jet'), interpolation='none', origin='lower')    
    
    return


def plot_bgd_images(table, slot, mag, n_start, n_stop, img_size, method):

    ok1 = table['bgd_type'] == method
    ok2 = table['slot'] == slot
    ok3 = table['mag'] == mag
    ok = ok1 * ok2 * ok3

    fig = plt.figure(figsize=(8.5, 25))

    for i, aa in enumerate(table[ok]['bgdimg'][0][n_start:n_stop]):
        plt.subplot(12, 10, i + 1)
        row0 = table[ok]['row0'][0]
        col0 = table[ok]['col0'][0]
        index = i + n_start
        plot_bgd_image(aa, index, row0, col0, img_size)
        plt.title('t{}:\n{}, {}'.format(index, row0[index], col0[index]));
        plt.axis('off')

    plt.subplots_adjust(left=0.05, bottom=0.2, right=0.99, top=0.9, hspace=0.3, wspace=0.1)

    print("Format of the titles is 'time: imgrow0, imgcol0'")
    print("Plot bgd images from {} to {}".format(n_start, n_stop))
    print("Method: {}, ndeque = {}".format(method, table[ok]['ndeque'][0]))

    return


def plot_bgd_patch(deque_dict, img_number, row0, col0, img_size):

    r_min = row0.min()
    r_max = row0.max() + img_size
    c_min = col0.min()
    c_max = col0.max() + img_size
    
    data = -100 * np.ones((c_max - c_min, r_max - r_min))

    dr = row0[img_number] - r_min
    dc = col0[img_number] - c_min
           
    keys = []
    
    for rr in range(r_min, r_max + 1):
        for cc in range(c_min, c_max + 1):
            keys.append((rr, cc))
        
    for key in deque_dict.keys():
        if key in keys:
            data[np.int(key[0] - r_min), np.int(key[1] - c_min)] = np.median(deque_dict[key])
        
    plt.imshow(data, cmap=plt.get_cmap('jet'), interpolation='none', origin='lower')    
    
    return


def plot_bgd_patches(table, slot, mag, n_start, n_stop, img_size, method):
    ok1 = table['bgd_type'] == method
    ok2 = table['slot'] == slot
    ok3 = table['mag'] == mag
    ok = ok1 * ok2 * ok3
    
    fig = plt.figure(figsize=(8.5, 25))

    for i, aa in enumerate(table[ok]['deque_dict'][0][n_start:n_stop]):
        plt.subplot(12, 10, i + 1)
        row0 = table[ok]['row0'][0]
        col0 = table[ok]['col0'][0]
        index = n_start + i        
        plot_bgd_patch(aa, index, row0, col0, img_size)
        plt.title('t{}:\n{}, {}'.format(index, row0[index], col0[index]));
        plt.axis('off')

    plt.subplots_adjust(left=0.05, bottom=0.2, right=0.99, top=0.9, hspace=0.3, wspace=0.1)
    
    print("Format of the titles is 'time: imgrow0, imgcol0'")
    print("Plot frames from {} to {}".format(n_start, n_stop))
    print("Method: {}, ndeque = {}".format(method, table[ok]['ndeque'][0]))
    
    return



