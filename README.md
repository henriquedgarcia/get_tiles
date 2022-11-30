# get_tiles
Script to get the tiles seen in viewport on a tiled spherical video. Additionally, 
it informs the fraction of the tile that appears in the viewport.

The tiling enumeration is done from left to right and from up to down. Below we show an equirectangular projection with 4x4 tiling.:

![tiling4x4.png!](img/tiling4x4.png)


The cubemap projection is packed like [MPEG 360lib](https://mpeg.chiariglione.org/standards/exploration/future-video-coding/n17197-algorithm-descriptions-projection-format-conversion) defaults:

![cubemap_packing.png!](img/cubemap_packing.png)


# Depends

- matplotlib
- numpy

# Usage
    get_tiles.py [-h] [-proj PROJECTION] [-fov FOV] [-tiling TILING] [-coord YAW PITCH ROLL] [-out OUTPUT_FILE]

Optional arguments:

    -h, --help              Show this help message and exit
    -proj PROJECTION        The projection [erp|cmp]. (Default: "erp")
    -fov FOV                The Field of View in degree (Default: "100x90")
    -tiling TILING          The tiling of projection. (Default: "3x2")
    -coord YAW PITCH ROLL   The center of viewport in degree. (Default: "0 0 0")
    -out OUTPUT_FILE        Save the projection marks to OUTPUT_FILE file.
	
# Example

## ERP

    python get_tiles.py -proj erp -fov 100x90 -tiling 12x8 -coord 30 -10 -10 -out erp_12x8.png

Output:  
```
{
  "29": "6.08%",
  "30": "44.90%",
  "31": "58.70%",
  "32": "19.32%",
  "41": "66.21%",
  "42": "100.00%",
  "43": "100.00%",
  "44": "54.54%",
  "53": "69.94%",
  "54": "100.00%",
  "55": "100.00%",
  "56": "73.86%",
  "65": "72.07%",
  "66": "100.00%",
  "67": "100.00%",
  "68": "72.46%",
  "77": "19.01%",
  "78": "45.61%",
  "79": "35.49%",
  "80": "2.63%"
}
```

![erp_12x8.png!](img/erp_12x8.png)

## CMP

    python get_tiles.py -proj cmp -fov 100x90 -tiling 12x8 -coord 30 -10 -10 -out cmp_12x8.png

Output:  
```
{
  "5": "0.64%",
  "6": "32.45%",
  "7": "80.87%",
  "8": "78.25%",
  "9": "13.72%",
  "17": "66.50%",
  "18": "100.00%",
  "19": "100.00%",
  "20": "100.00%",
  "21": "49.35%",
  "29": "76.86%",
  "30": "100.00%",
  "31": "100.00%",
  "32": "100.00%",
  "33": "74.82%",
  "41": "79.33%",
  "42": "100.00%",
  "43": "100.00%",
  "44": "100.00%",
  "45": "67.20%",
  "48": "99.36%",
  "49": "13.21%",
  "59": "0.58%",
  "60": "73.27%",
  "72": "29.39%"
}
```
![cmp_12x8.png!](img/cmp_12x8.png)


# Limitations

The projection resolution affects script accuracy. So the resolution 
is fixed at "1296x648" for erp and "972x648" for cmp. This allows 
projection tiling by 3x2, 6x4, 9x6 and 12x8 without pixel rounding, i.e. all 
tiles with the same numbers of pixels.
