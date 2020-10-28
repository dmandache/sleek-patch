from skimage import *
from skimage.segmentation import slic, mark_boundaries, clear_border
from skimage.util import img_as_float
from skimage.transform import rescale, resize
from skimage.io import imsave, imread
from skimage import filters, color
from skimage.draw import circle, polygon_perimeter, polygon

from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage.morphology import binary_fill_holes

import numpy as np
import random


def slic_patchify(image, patch_size=256, overlap=0, remove_background=False, annotation_mask=None):
    """
        Split image into patches (given patch size and overlap) using SLIC superpixel aglorithms,
        this method results in an adaptive grid.
    :param image: input image
    :param patch_size: patch size
    :param overlap: number of overlapping pixels
    :param remove_background: boolean - if there is an annotation mask there's no need to remove background
    :param annotation_mask:  annotation mask ( 0 = background)
    :return: list of patches / list of dictionaries [{class, patches},{class, patches},...]
    """
    scale = image.shape[0] // 1000

    if annotation_mask is not None:
        image = np.where(annotation_mask != 0, image, 0)

    if image.ndim > 2:
        image_gray = color.rgb2gray(image)
    else:
        image_gray = image_gray

    patch_size_slic = patch_size - overlap
    segments = _get_slic_segments(image_gray, patch_size_slic=patch_size_slic, scale=scale,
                                 remove_background=remove_background, buffer_size=patch_size // scale)
    centers = _segments_to_centers(segments)
    patches, centers = _centers_to_patches(image, centers, patch_size=patch_size)

    draw_markers(centers, patch_size, image_gray, filename='markers.png', on_image=True, linewidth=10)

    return patches, centers


def _get_slic_segments(image, patch_size_slic=256, scale=8, remove_background=False, buffer_size=0):
    # Compute number of segments according the desired "patch size"
    n_segments = image.shape[0] * image.shape[1] // (patch_size_slic ** 2)

    # Downscale image
    image_small = rescale(image, scale=1 / scale)
    print(
        "Running slic (patch size = {} => number of segments = {}) on rescaled image : scale = {}, image size {}x{}".format(
            patch_size_slic,
            n_segments,
            scale,
            image_small.shape[0],
            image_small.shape[1]))
    imsave('image_small.png', image_small)

    # Run SLIC
    segments_small = slic(img_as_float(image_small), n_segments=n_segments, sigma=3, compactness=0.3)
    # segments_small = clear_border(segments_small, buffer_size=buffer_size)

    imsave('segments_small.png', segments_small)
    print(segments_small.shape)

    if remove_background:

        # Compute mean intensity of superpixels
        mean_segments_small = np.zeros_like(image_small)
        for (i, label) in enumerate(np.unique(segments_small)):
            superpix = image_small[segments_small == label]
            mean_intensity = np.mean(superpix)
            mean_segments_small[segments_small == label] = mean_intensity
        imsave('segments_mean_intensity.png', mean_segments_small)

        # TODO keep entire superpixel when masking (don't cut labels)
        # Background removal
        background_thresh = filters.threshold_li(mean_segments_small)
        background_mask = np.zeros_like(segments_small)
        background_mask[mean_segments_small > background_thresh] = 1
        background_mask = binary_fill_holes(background_mask)
        # background_mask = ndimage.morphology.binary_opening(background_mask, structure=np.ones((10, 10)))
        segments_small[background_mask == 0] = 0
        imsave('mask.png', img_as_uint(background_mask))

    # Upscale segments image
    segments = resize(segments_small, output_shape=(image.shape[0], image.shape[1]), order=0, preserve_range=True)

    return segments


def _segments_to_centers(segments):
    # Patch centers
    markers_center = np.array(center_of_mass(segments, segments, np.unique(segments)))
    markers_center = markers_center[np.isfinite(markers_center).any(axis=1)]  # remove NaN and Inf
    return markers_center


def _centers_to_patches(image, centers, patch_size=256):
    topleft = np.array(centers, dtype=np.int64) - patch_size // 2
    patches = []
    new_centers = []
    for i, corner in enumerate(topleft):
        r, c = corner
        if r + patch_size <= image.shape[0] and c + patch_size <= image.shape[1]:
            patches.append(image[r:r + patch_size, c:c + patch_size, ...])
            new_centers.append(centers[i])
    return patches, new_centers


def draw_markers(markers_centers, patch_size, image, filename='markers.png', on_image=False, linewidth=10):
    def get_rand_color():
        return [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

    if on_image:
        draw_rgb = np.stack((image,) * 3, -1)
    else:
        draw_rgb = np.zeros(shape=(image.shape[0], image.shape[1], 3), dtype=int)

    for point in markers_centers:
        r, c = point
        rcirc, ccirc = circle(r, c, radius=10)
        rsqr, csqr = polygon_perimeter(
            [r - patch_size // 2, r + patch_size // 2, r + patch_size // 2, r - patch_size // 2],
            [c - patch_size // 2, c - patch_size // 2, c + patch_size // 2, c + patch_size // 2],
            shape=draw_rgb.shape)
        rand_color = get_rand_color()
        draw_rgb[rcirc, ccirc] = rand_color
        draw_rgb[rsqr, csqr] = rand_color
        if linewidth != 0:
            for i in range(linewidth // 2 + 1):
                try:
                    draw_rgb[rsqr - i, csqr - i] = rand_color
                    draw_rgb[rsqr + i, csqr + i] = rand_color
                except:
                    continue

    imsave(filename, draw_rgb)

#TODO
def reconstruct_patches(centers, patches, image, filename='reconstructed_image.png', on_image=False, linewidth=10):
    def get_rand_color():
        return [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

    if on_image:
        pass
        # draw_rgb = np.stack((image,)*3, -1)
    else:
        draw_rgb = np.zeros(shape=(image.shape[0], image.shape[1], 3), dtype=int)

    assert len(centers) == len(patches)

    for i, point in enumerate(centers):
        r, c = point
        rcirc, ccirc = circle(r, c, radius=10)
        rsqr, csqr = polygon_perimeter(
            [r - patch_size // 2, r + patch_size // 2, r + patch_size // 2, r - patch_size // 2],
            [c - patch_size // 2, c - patch_size // 2, c + patch_size // 2, c + patch_size // 2],
            shape=draw_rgb.shape)
        rand_color = get_rand_color()
        draw_rgb[rcirc, ccirc] = rand_color
        draw_rgb[rsqr, csqr] = rand_color
        if linewidth != 0:
            for i in range(linewidth // 2 + 1):
                try:
                    draw_rgb[rsqr - i, csqr - i] = rand_color
                    draw_rgb[rsqr + i, csqr + i] = rand_color
                except:
                    continue

    imsave(filename, draw_rgb)