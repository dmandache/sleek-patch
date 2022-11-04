# **S**ample **O**ptimally with **SLIC** - So Sleek !

## 1 - What

The method consists of an image processing pipeline leading to the sampling of a bigger image into tiles, by taking into account the textures of said tiles, with the purpose of obtaining an optimal, conent-aware split, without "cutting up" homogenous structures.

## 2 - Why 

The method has been developed in the big data / deep learning context out of the need of sampling gigapixel medical images into minimally overlapping homogenous sub-parts for training a multiple instance learning model based on convultional neural networks, however, it has the potential of broader use.

Use case here: D. Mandache, E. B. à La Guillaume, Y. Badachi, J. . -C. Olivo-Marin and V. Meas-Yedid, "The Lifecycle of a Neural Network in the Wild: A Multiple Instance Learning Study on Cancer Detection from Breast Biopsies Imaged with Novel Technique," 2022 IEEE International Conference on Image Processing (ICIP), 2022, pp. 3601-3605, doi: 10.1109/ICIP46576.2022.9897596.

## 3 - How

The method strongly relies on the SLIC superpixel segmentation algorithm implemented in Scikit-Image [skimage.segmentation.slic](https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic)

Pipeline: image -> convert to grayscale -> downscale -> Gaussian blur -> estimate number of superpixels -> segment into superpixels -> filter out background superpixels -> extract centers of mass from superpixels -> upscale -> define corresponding patches 


## 4 - Functions & Parameters

### a) Core Sampling Function `sleek_patchify`
#### Parameters
- `image` 
- `patch_size`
- `overlap` : note that while this value is exact for the regular grid sampling given as baseline, for the Sleek method the overlap value is approximative
- `scale` : integer, downscaling factor for speeding up the execution
##### from SLIC (see [skimage.segmentation.slic](https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic) for more details)
  - `sigma` : wdth of Gaussian smoothing kernel for pre-processing 
  - `compactness`: balances color proximity and space proximity (higher values give more weight to space proximity, making superpixel shapes more square)
  - `min_size_factor` : proportion of the minimum superpixel size to be removed with respect to the supposed initial square size
  - `max_size_factor` : proportion of the maximum connected superpixel size
  - `slic_zero`, boolean flag, if *True* runs the zero-parameter mode of SLIC
  - `mask`: boolean 2D array given as mask for area of interest to patchify
##### for background removal
  - `remove_background`: boolean flag, should be *False* if a mask is already provided
  - `background_removal_strategy`: thresholding strategy applied on the mean intensity of the obtained pixel clusters, accepted values: *isodata, otsu, li, yen, triangle, quantile*
  - `background_is`: specify if the background is lighter or darker than the foregroud, accepted values: *light, dark*
##### for saving intermediary steps
  - `debug` : boolean flag, if *True* saves images of the intermediary steps, like the result of the SLIC algrithm, background mask, etc.
  - `logdir` : path to the directory where to save the debugging files
#### Returns
- list of extracted patches
- list of coordinates for the centers of the patches inside the image

### b) Baseline Sampling Function `grid_patchify`
regular grid sampling with the same background removal stretegy as above

### c) Reconstruction Function `reconstruct_patches`
reconstruct the image from the sampled patches and their position

### d) Visualization Function `draw_markers` 
draws sampled patches over the image

## 5 - Example

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

 ## 6 - Installation
 
 - download repository
 - `pip install -e /path/to/repository`
 
 ## 7 - Usage
 
 - `import sleek`
 - load `image` 
 - `patches, centers = sleek.sleek_patchify(image, ...)`
 
 For more details see [`demo`](demo/demo.ipynb).
