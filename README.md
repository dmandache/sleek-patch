# sleek-patch
## Optimally patchify your images using the SLIC superpixel algorithm - simply sleek !

## What

The method consists of an image processing pipeline leading to the sampling of a bigger image into tiles, by taking into account the textures of said tiles, with the purpose of obtaining an optimal, conent-aware split, without "cutting up" homogenous structures.

## Why 

The method has been developed in the big data / deep learning context out of the need of sampling gigapixel medical images into minimally overlapping homogenous sub-parts for training a multiple instance learning model based on convultional neural networks, however, it has the potential of broader use.

## How

The method strongly relies on the SLIC superpixel segmentation algorithm implemented in Scikit-Image [skimage.segmentation.slic](https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic)

image -> convert to grayscale -> downscale -> Gaussian blur -> segment into superpixels -> filter out background superpixels -> extract superpixels centers of mass -> define corresponding patches



## Functions & Parameters

### Core Sampling Function: `sleek_patchify`
#### Parameters
- `image` : 
- `patch_size`
- `overlap` : note that while this value is exact for the regular grid sampling given as baseline, for the Sleek method the overlap value is approximative
- `scale` : integer, downscaling factor for speeding up the execution
SLIC (see [skimage.segmentation.slic](https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic) for more details)
- `multichannel` : boolean flag, if *False* converting the image to grayscale
- `sigma` : wdth of Gaussian smoothing kernel for pre-processing 
- `compactness`: balances color proximity and space proximity (higher values give more weight to space proximity, making superpixel shapes more square)
- `min_size_factor` : proportion of the minimum superpixel size to be removed with respect to the supposed initial square size
- `max_size_factor` : proportion of the maximum connected superpixel size
- `enforce_connectivity`=True
- `slic_zero`, boolean flag, if *True* runs the zero-parameter mode of SLIC
- `mask`: boolean 2D array given as mask for area of interest to patchify
Background removal
- `remove_background`: boolean flag, should be *False* if a mask is already provided
- `background_removal_strategy`: thresholding strategy applied on the mean intensity of the obtained pixel clusters, accepted values: *isodata, otsu, li, yen, triangle, quantile*
- `background_is`: specify if the background is lighter or darker than the foregroud, accepted values: *light, dark*
Saving intermediary steps
- `debug` : boolean flag, if *True* saves images of the intermediary steps, like the result of the SLIC algrithm, background mask, etc.
- `logdir` : path to the directory where to save the debugging files
#### Returns
- list of extracted patches
- list of coordinates for the centers of the patches inside the image

### Baseline Sampling Function
`grid_patchify` regular grid sampling with the same background removal stretegy as above

### Reconstruction Function
`reconstruct_patches` reconstruct the image from the sampled patches and their position

### Visualization Function
`draw_markers` draws sampled patches over the image

## Example

Example image comes from The Early **B**reast **C**ancer Core-**N**eedle **B**iopsy WSI (BCNB) Dataset, freely available at <https://bupt-ai-cz.github.io/BCNB/> and the foreground mask is produced by the author using [Icy Platform](https://icy.bioimageanalysis.org/).

WSI of size 14208 x 18080 pixels, sampled with patches of size 2048 x 2048 with an overlap of 256 pixels, the Sleek method is applied on the greyscale transformed image down-scaled with a factor of 10  

<table>
  <tr>
        <td>Image</td>
        <td>Mask</td>
  </tr>
  <tr>
        <td><img src="demo/280.jpg" width="256" /></td>
        <td><img src="demo/280_mask.jpg" width="256" /></td> 
  </tr>
 </table>

<table>
  <tr>
        <td>Regular Grid Sampling</td>
        <td>Sleek Patchification</td>
        <td>Masked Sleek Patchification</td>
  </tr>
  <tr>
        <td><img src="demo/280_regular_grid_markers.jpg" width="320" /></td>
        <td><img src="demo/280_sleek_markers.jpg" width="320" /></td>
        <td><img src="demo/280_masked_sleek_markers.jpg" width="320" /></td>
  </tr>
 </table>
