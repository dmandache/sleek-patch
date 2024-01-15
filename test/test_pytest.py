from skimage.data import shepp_logan_phantom
import numpy as np

import matplotlib.pyplot as plt
from sleek.patchify import *

def test_grid_patchify():
    patch_size = 124
    overlap = 32
    # load test image
    image = shepp_logan_phantom() 
    image = np.array(image * 255).astype(np.uint8)
    #plt.imsave("./test/test_image.png",image)
    #print(image.shape, image.dtype, image.mean(), image.min(), image.max())
    # regular grid patchify
    patches_list, centers_list = grid_patchify(image, patch_size=patch_size, overlap=overlap, remove_background=True, background_is='dark')
    #print(len(patches_list))
    # display markers
    img_with_grid = draw_markers(centers_list, patch_size, image, filename='test_regular_grid_markers.jpg', on_image=True, linewidth=2) #, color=[255,255,255])
    #plt.imsave("./test/test_image_regular_grid.png",img_with_grid)
    # sleek patchify
    patches_list, centers_list = sleek_patchify(image, patch_size=patch_size, overlap=overlap, scale=1, remove_background=True, background_is='dark')
    #print(len(patches_list))
    # display markers
    img_with_grid = draw_markers(centers_list, patch_size, image, filename='test_regular_grid_markers.jpg', on_image=True, linewidth=2) #, color=[255,255,255])
    #plt.imsave("./test/test_image_sleek_grid.png",img_with_grid)
    return True


if __name__ == "__main__":
    test_grid_patchify()
    