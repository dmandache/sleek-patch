from skimage import *
from skimage.segmentation import slic, mark_boundaries, clear_border
from skimage.util import img_as_float
from skimage.transform import rescale, resize
from skimage.io import imsave, imread
from skimage.color import rgb2gray
from skimage import filters
from skimage.draw import disk, polygon_perimeter, polygon

from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage.morphology import binary_fill_holes, binary_closing
from scipy.stats import entropy

from skimage.morphology import remove_small_holes, remove_small_objects, convex_hull_object

import numpy as np
import random
import os

Q = 0.1         # q-th for quantile background removal threshold

BACKGROUND_REMOVAL_FUNCTION = {
    'isodata':  filters.threshold_isodata,
    'otsu':     filters.threshold_otsu,
    'li':       filters.threshold_li,
    'yen':      filters.threshold_yen,
    'triangle': filters.threshold_triangle,
}


def grid_patchify(image, patch_size=256, overlap=0, remove_background=False, background_removal_strategy='isodata', background_is='dark'):
    """
    Regular Grid sampling

    :param image:
    :param patch_size:
    :param overlap:
    :param remove_background:
    :param background_is:
    :return:
    """
    img_h, img_w, _ = image.shape


    def start_points(size, patch_size, overlap=0):
        points = [0]
        stride = patch_size - overlap
        counter = 1
        while True:
            pt = stride * counter
            if pt + patch_size >= size:
                #if pt + patch_size - size < patch_size//2:
                #    points.append(size - patch_size)
                break
            else:
                points.append(pt)
            counter += 1
        return points


    X_points = start_points(img_w, patch_size, overlap)
    Y_points = start_points(img_h, patch_size, overlap)

    count = 0
    patches = []
    centers = []

    for i in Y_points:
        for j in X_points:
            p = image[i:i+patch_size, j:j+patch_size]
            patches.append(p)
            centers.append([i+patch_size//2, j+patch_size//2])
            count += 1

    if remove_background:
        mean_patches = np.array([np.mean(p) for p in patches])
        background_thresh = BACKGROUND_REMOVAL_FUNCTION[background_removal_strategy](mean_patches)
        if background_is == 'dark':
            fg_ids = np.argwhere(mean_patches <= background_thresh).flatten()
        else:
            fg_ids = np.argwhere(mean_patches < background_thresh).flatten()
        return [patches[index] for index in fg_ids], [centers[index] for index in fg_ids]
    else:
        return patches, centers


def sleek_patchify(image, patch_size=256, overlap=0, scale=1, multichannel=False,
                  sigma=3, compactness=0.5, min_size_factor=0.1, max_size_factor=3, enforce_connectivity=True, slic_zero=False,
                  remove_background=False, background_removal_strategy='isodata', background_is='dark',
                  mask=None, logdir="./", debug=False):
    """
        Split image into patches (given patch size and overlap) using SLIC superpixel aglorithms,
        this method results in an adaptive grid.
    :param image: input image
    :param patch_size: int, patch size
    :param overlap: int,  number of overlapping pixels (approximate)
    :param scale: int, apply algo on downscaled image
    :param remove_background: boolean - if there is an annotation mask there's no need to remove background
    :param background_is: light(er) or dark(er)
    :param annotation_mask:  annotation mask ( 0 = background)

    :return: list of patches / list of dictionaries [{class, patches},{class, patches},...]
    :return: list of centers of the patches
    """
    if image.ndim > 2 and multichannel is False:
        image_gray = rgb2gray(image)
        image_gray = (image_gray*255).astype('uint8') # float to int
    else:
        image_gray = np.copy(image)
    
    patch_size_slic = patch_size - overlap
    segments = _get_slic_segments(image_gray, 
                                  patch_size_slic=patch_size_slic, 
                                  scale=scale,
                                  remove_background=remove_background,
                                  background_removal_strategy=background_removal_strategy,
                                  background_is=background_is,
                                  mask=mask,
                                  multichannel=multichannel,
                                  sigma=sigma, 
                                  compactness=compactness, 
                                  min_size_factor=min_size_factor, 
                                  max_size_factor=max_size_factor,
                                  enforce_connectivity=enforce_connectivity,
                                  slic_zero=slic_zero,
                                  buffer_size=patch_size // scale,
                                  debug=debug)
    
    centers = _segments_to_centers(segments)
    patches, centers = _centers_to_patches(image, centers, patch_size=patch_size)

    if debug:
        draw_markers(centers, patch_size, image, filename=os.path.join(logdir, 'markers.jpg'), on_image=True, linewidth=20)

    return patches, centers


def _get_slic_segments(image, patch_size_slic=256, scale=8, 
                            remove_background=False, background_removal_strategy='isodata', background_is='dark', mask=None, buffer_size=0, debug=False,
                            multichannel=False, sigma=3, compactness=0.5, min_size_factor=0.2, max_size_factor=3,
                            enforce_connectivity=True, slic_zero=False):
    # Compute number of segments according the desired "patch size"
    if mask is not None:
        # calcualte n of segmetns on the convex hull of mask
        # ch = convex_hull_object(rescale(mask, 1/scale))
        # ch = resize(ch, output_shape=(mask.shape[0], mask.shape[1]), order=0, preserve_range=True).astype('bool')
        # n_segments = (np.sum(ch) + np.sum(mask)) // 2 // (patch_size_slic ** 2)
        alpha = 1 #1.2
        n_segments = int(alpha * np.sum(mask) // (patch_size_slic ** 2))
        #n_segments += int(n_segments*0.15)
    else:
        n_segments = image.shape[0] * image.shape[1] // (patch_size_slic ** 2)
        
    # Downscale image
    image_small = rescale(image, scale=1 / scale, multichannel=multichannel)
    
    # Downscale mask
    if mask is not None:
        mask = rescale(mask, scale=1 / scale) 
        
    print(
        "Running slic (patch size = {} => number of segments = {}) on rescaled image : scale = {}, image size {}x{}".format(
            patch_size_slic,
            n_segments,
            scale,
            image_small.shape[0],
            image_small.shape[1]), flush=True)
    
    if debug:
        imsave('image_small.png', image_small)

    # Run SLIC
    segments_small = slic(img_as_float(image_small), mask=mask, 
                          min_size_factor=min_size_factor, max_size_factor=max_size_factor, max_iter=10,
                          n_segments=n_segments, multichannel=multichannel, slic_zero=slic_zero,
                          sigma=sigma, compactness=compactness, enforce_connectivity=True) 
    # segments_small = clear_border(segments_small, buffer_size=buffer_size)

    if debug:
        imsave('segments_small.png', segments_small)
    print(segments_small.shape)

    if remove_background and mask is None:
        segments_small = _remove_background_segments(image_small, segments_small,
                                                     background_removal_strategy=background_removal_strategy, background_is=background_is,
                                                     debug=debug)

    # Upscale segments image
    segments = resize(segments_small, output_shape=(image.shape[0], image.shape[1]), order=0, preserve_range=True)
    
    return segments


def _remove_background_segments(image, segments, background_removal_strategy='isodata', background_is='dark', debug=False):
    segments += 1  # keep label 0 for background

    # Compute mean intensity of superpixels
    mean_segments = np.zeros_like(image)
    for (i, label) in enumerate(np.unique(segments)):
        superpix = image[segments == label]
        # mask = remove_background[segments == label]
        mean_intensity = np.mean(superpix)  # np.quantile(superpix, 0.9)

        # mean_intensity = entropy(superpix) * np.mean(superpix)
        mean_segments[segments == label] = mean_intensity
    if debug:
        imsave('segments_mean_intensity.jpg', mean_segments)

    mean_segments = np.nan_to_num(mean_segments, nan=0)
    print(mean_segments.dtype, mean_segments.min(), mean_segments.max())

    # TODO keep entire superpixel when masking (don't cut labels)
    # Background removal
    if background_removal_strategy == 'quantile':
        if background_is == 'dark':
            background_thresh = np.quantile(mean_segments, Q)
        else:
            background_thresh = np.quantile(mean_segments, 1-Q)
    else:
        background_thresh = BACKGROUND_REMOVAL_FUNCTION[background_removal_strategy](mean_segments)
    print("background thresh = ", background_thresh)

    background_mask = np.zeros_like(segments)
    if background_is == 'dark':
        background_mask[mean_segments > background_thresh] = 1
    else:
        background_mask[mean_segments <= background_thresh] = 1
    background_mask = binary_fill_holes(background_mask)
    background_mask = remove_small_objects(background_mask, 200)
    # background_mask = ndimage.morphology.binary_opening(background_mask, structure=np.ones((10, 10)))
    segments[background_mask == 0] = 0

    if debug:
        imsave('mask.jpg', img_as_uint(background_mask))
        mean_segments[background_mask == 0] = 0
        imsave('segments_mean_intensity_masked.jpg', mean_segments)

    return segments


def _segments_to_centers(segments):
    # Patch centers
    labels = np.unique(segments)
    markers_center = np.array(center_of_mass(segments, segments, labels))
    markers_center = markers_center[np.isfinite(markers_center).any(axis=1)]  # remove NaN and Inf
    markers_center = markers_center.astype('int64')
    return markers_center


def _centers_to_patches(image, centers, patch_size=256):
    topleft = np.array(centers, dtype=np.int64) - patch_size // 2
    patches = []
    new_centers = []

    if image.ndim > 2:
        patch_shape = (patch_size, patch_size, image.shape[2])
    else:
        patch_shape = (patch_size, patch_size)

    for i, corner in enumerate(topleft):
        r, c = corner
        if r >= 0 and c >= 0 and r + patch_size <= image.shape[0] and c + patch_size <= image.shape[1]:
            p = image[r:r + patch_size, c:c + patch_size, ...]
        else:
            p = np.zeros(shape=patch_shape, dtype=image.dtype)
            dr = dc = 0
            if r < 0:
                dr = 0 - r
            if c < 0:
                dc = 0 - c
            # p[dr:, dc:, ...] = image[r+dr:r + patch_size, c+dc:c + patch_size, ...]
            padded_p = image[r + dr:np.minimum(r + patch_size, image.shape[0]),
                       c + dc:np.minimum(c + patch_size, image.shape[1]), ...]
            p[dr:dr + padded_p.shape[0], dc:dc + padded_p.shape[1], ...] = padded_p
        patches.append(p)
        new_centers.append(centers[i])
    return patches, new_centers


def draw_markers(markers_centers, patch_size, image, filename='markers.jpg', on_image=False, linewidth=12, color=None):
    """

    :param markers_centers:
    :param patch_size:
    :param image:
    :param filename:
    :param on_image:
    :param linewidth:
    :param color:
    :return:
    """
    def get_rand_color():
        return [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

    print(image.dtype, image.min(), image.max())

    draw_rgb = _get_canvas(np.copy(image), on_image)

    for point in markers_centers:
        r, c = point
        rcirc, ccirc = disk((r, c), radius=10)
        rsqr, csqr = polygon_perimeter(
            [r - patch_size // 2, r + patch_size // 2, r + patch_size // 2, r - patch_size // 2],
            [c - patch_size // 2, c - patch_size // 2, c + patch_size // 2, c + patch_size // 2],
            shape=draw_rgb.shape)
        if color is None:
            rand_color = get_rand_color()
        else:
            rand_color = color
        draw_rgb[rcirc, ccirc] = rand_color
        draw_rgb[rsqr, csqr] = rand_color
        if linewidth != 0:
            for i in range(linewidth // 2 + 1):
                try:
                    draw_rgb[rsqr - i, csqr - i] = rand_color
                    draw_rgb[rsqr + i, csqr + i] = rand_color
                except:
                    continue
    if True:
        imsave(filename, draw_rgb, quality=10)
    return draw_rgb


def reconstruct_patches(centers, patches, image, filename='reconstructed_image.jpg', on_image=False):

    draw_rgb = _get_canvas(image, on_image)

    assert len(centers) == len(patches)

    patch_size = patches[0].shape[0]

    for i, center in enumerate(centers):
        print(i, center)
        r, c = center
        r, c = int(r), int(c)
        offset = patch_size // 2

        x0, x1 = max(0, r - offset), min(image.shape[0], r + offset)
        y0, y1 = max(0, c - offset), min(image.shape[1], c + offset)

        draw_rgb[x0:x1, y0:y1] = patches[i]
    if True:
        imsave(filename, draw_rgb)
    return draw_rgb


def _get_canvas(image, on_image=False):
    if on_image:
        if image.ndim == 2 : # grayscale image
            draw_rgb = np.stack((image,) * 3, -1)
        else:
            draw_rgb = image
    else:
        draw_rgb = np.zeros(shape=(image.shape[0], image.shape[1], 3), dtype=int)
    return draw_rgb