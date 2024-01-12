from skimage.data import shepp_logan_phantom

import sleek.patchify

def test_grid_patchify():
    image = shepp_logan_phantom()
    print(image.shape)
    return True