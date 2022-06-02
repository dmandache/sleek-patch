# sleek-patch
## Optimally patchify your images using the SLIC superpixel algorithm - simply sleek !

## Motivation
big images
content aware sampling
reduce number of patches and improve their quality

## Method

slic clustering 

## Parameters

## Example

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