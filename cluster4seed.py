#%%
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
maskfile = "demodata/mask.npy"

#%%
def flatten_image(v, maskfile=maskfile):
    mask = np.load(maskfile)
    value = []
    index = []
    for j in range(len(v[0])):
        for i in range(len(v)):
            t = v[i][j]
            if mask[i][j]:
                value.append(t)
                index.append((i, j))
    return index, value

def cluster_run(traning_set, findex, n):
    af = MiniBatchKMeans(n)
    af.fit(traning_set)
    labels = af.labels_
    closest, _ = pairwise_distances_argmin_min(af.cluster_centers_, traning_set)
    label_image = np.full((128,128), np.nan)
    for (i,j),l in zip(findex, labels):
      label_image[i,j] = l
    return label_image, [findex[i] for i in closest]

def cluster_guess(v):
    findex, fvalue = flatten_image(v)
    traning_set = [np.nan_to_num(b.flatten()) for b in fvalue]
    clusters = [(i, cluster_run(traning_set, findex, i)) for i in range(1, 25)]
    return clusters

def animation_corrlation(indices, cluster, images, out):
    image_number = len(images)
    fig, axs = plt.subplots(figsize=(4 * image_number + 1, 4), ncols=1 + image_number)
    fig.set_dpi(300.0)
    i, j = indices[0]
    im0 = axs[0].imshow(cluster)
    ims = [axs[t+1].imshow(im[i][j], cmap=plt.cm.Spectral_r, vmin = -1.0, vmax = 1.0)
           for t,im in zip(range(image_number), images)]
    patch0 = plt.Circle((j, i), 0.75, color="gray")
    patches = [plt.Circle((j, i), 0.75, color="gray") for t in ims]
    axs[0].add_patch(patch0)
    for t, patch in zip(range(image_number), patches):
        axs[t+1].add_patch(patch)
    fig.colorbar(im0, ax=axs[0])
    for t, im in zip(range(image_number), ims):
        fig.colorbar(im, ax=axs[t+1])

    def update(i):
        ii, jj = indices[i]
        for im, image in zip(ims, images):
            im.set_array(image[ii][jj])
        patch0.center = (jj, ii)
        for patch in patches:
            patch.center = (jj, ii)

    anim = FuncAnimation(fig, update, np.arange(len(indices)), interval=50)

    anim.save(out)
    plt.close(anim._fig)

def animation_cluster(cluster_image, ims, filename, resolution):
    id_indices = []
    for j in range(0, cluster_image.shape[1], resolution):
        for i in range(0, cluster_image.shape[0], resolution):
            t = cluster_image[i, j]
            if not np.isnan(t):
                id_indices.append((t,(i,j)))
    id_indices.sort(key=lambda x: x[0])
    indices = [x[1] for x in id_indices]
    animation_corrlation(indices, cluster_image, ims, filename)

def process(filename):
    data = pickle.load(filename)
    clusters = cluster_guess(data)
    return clusters
