## `img_proc` - small library and CLI program with a variety of image processing functions

### How to use

`python3 img_proc.py img.png --dither` will create a file `dither_result.png`.
There are some images to play with in `samples` directory.

### Available functoins

* Uniform quantization
* Median filter for quantization
* kNN filter for quantization
* Binary threshold
* White noise generation
* Dithering using Floyd-Steinberg algorithm
* Bayes filter
* Gauss filter
* Sobel, Prewitt, and Laplace operators

### Requirements

* `numpy`
* `scipy`
* `opencv_python` (OpenCV)
* `matplotlib`
