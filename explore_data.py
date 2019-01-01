'''Script for data exploration
'''
import glob
import os
import threading
import pickle

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

SET = "train" # 'train' or 'valid'
DATAPATH = "assets/flower_data/" + SET
FILE_EXT = ".jpg"

NUM_THREADS = 4
image_hist = {
    'filenames': [],
    'row_sizes': [],
    'col_sizes': [],
    'depths':    [],
}

LOAD_PKL = True
PKL_FN = os.path.join('assets', SET + "_data_hst.pkl")

class ReadImageThread(threading.Thread):
    '''Thread class for processing the images
    '''
    def __init__(self, name, fn):
        threading.Thread.__init__(self)
        self.fn = fn
        self.name = name

    def run(self):
        global image_hist

        for file in self.fn:
            im = Image.open(file)
            nd_img = np.asarray(im)
            im.close()

            image_hist['filenames'].append(file)
            image_hist['row_sizes'].append(nd_img.shape[0])
            image_hist['col_sizes'].append(nd_img.shape[1])
            image_hist['depths'].append(nd_img.shape[2])

def show_dim_data(image_hist):
    '''Show histogram of dimensions
    '''
    heights = np.array(image_hist['row_sizes'])
    widths = np.array(image_hist['col_sizes'])
    depths = np.array(image_hist['depths'])
    asps = np.array(image_hist['row_sizes']) / np.array(image_hist['col_sizes'])

    _, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    ax[0, 0].hist(heights, bins=10)
    ax[0, 0].set_title("height")
    ax[0, 1].hist(widths, bins=10)
    ax[0, 1].set_title("width")
    ax[1, 0].hist(depths, bins=10)
    ax[1, 0].set_title("depth")
    ax[1, 1].hist(asps, bins=10)
    ax[1, 1].set_title("aspect ratio")

    plt.show()

def display_images_by_aspect_ratio(image_hist, n_bins):
    '''Display images grouped according to aspect ratio
    '''
    # aspect ratio
    asps = np.array(image_hist['row_sizes']) / np.array(image_hist['col_sizes'])

    # get range of values
    min_asp = min(asps)
    max_asp = max(asps)

    step = (max_asp - min_asp) / n_bins
    imgs = np.array(image_hist['filenames'])
    for b in range(n_bins):
        # get bin boundaries
        lb = min_asp + (b * step)
        ub = lb + step
        bn = imgs[np.logical_and((asps >= lb), (asps < ub))]

        # sample for displaying
        if len(bn) > 4:
            disp_img = np.random.choice(bn, size=4)
        else:
            disp_img = bn

        _, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
        for i, sb in enumerate(ax.flatten()):
            if i >= len(bn):
                break

            im = Image.open(disp_img[i])
            sb.imshow(im)
            sb.set_title(disp_img[i])
            im.close()

        plt.show()

def main():
    '''Main
    '''
    filenames = glob.glob(os.path.join(DATAPATH, '**/*' + FILE_EXT), recursive=True)

    if LOAD_PKL:
        with open(PKL_FN, 'rb') as handle:
            image_hist = pickle.load(handle)

    else:
        readImageThreads = []

        print("Generating image size histogram using %d threads..." % (NUM_THREADS))
        for thread_idx in range(NUM_THREADS):
            # assign each thread a subset to process
            readImageThreads.append(ReadImageThread("ReadImageThread-{}".format(thread_idx),
                                                    filenames[thread_idx:][::NUM_THREADS]))
            readImageThreads[thread_idx].start()

        for thread_idx in range(NUM_THREADS):
            readImageThreads[thread_idx].join()

        with open(PKL_FN, 'wb') as handle:
            pickle.dump(image_hist, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Saved dataset properties to %s!" % PKL_FN)

    # show_dim_data(image_hist)
    # display_images_by_aspect_ratio(image_hist, 7)

    # label histogram
    labels = [int(os.path.basename(os.path.dirname(fn))) for fn in filenames]
    print("Range: %d - %d" % (min(labels), max(labels)))
    plt.hist(np.array(labels), bins=len(set(labels)))
    plt.show()

if __name__ == "__main__":
    main()
