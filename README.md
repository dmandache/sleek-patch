# sleek-patch
## Optimally patchify your images using the SLIC superpixel algorithm - simply sleek !

## Motivation
- big images context
- content aware sampling: du not split up homogenous textures in the image
- reduce number of patches and improve their quality

## Method

slic clustering 

## Functions & Parameters
`sleek_patchify`
- `patch_size`
- `overlap` : note that while this value is exact for the regular grid sampling given as baseline, for the Sleek method the overlap value is approximative
- `background_removal_strategy`: thresholding strategy applied on the mean intensity of the obtained pixel clusters, accepted values: *isodata, otsu, li, yen, triangle, quantile*

`grid_patchify` regular grid sampling as baseline

`draw_markers` draws sampled patches

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
        <td><img src="demo/280_regular_grid_markers.jpg" width="256" /></td>
        <td><img src="demo/280_sleek_markers.jpg" width="256" /></td>
        <td><img src="demo/280_masked_sleek_markers.jpg" width="256" /></td>
  </tr>
 </table>
