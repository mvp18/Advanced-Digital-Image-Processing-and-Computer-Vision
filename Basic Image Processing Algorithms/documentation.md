# Assignment- 1 Documentation
- In this assignment we have implemented some basic image processing operations as specified in this document.
- The auxiliary helper functions are written in the file `utils.py` and the main modules are implemented in the `modules.py` file. The order of implemented functions is same as the order given in the question.
- We call the functions in the `main.py` file. The details of the functions follow.

### In `utils.py` file
| Name | Input | Output |
| --- | --- | --- |
| `def isvalid(i, j, r, c)` | Determines whether a pixel `(i,j)` is outside, or illegal for the image of dimensions `r x c`. | 0 or 1. |
| `def euc_dist(x1, y1, x2, y2)` | Takes two points in their `(x,y)` coordinate form. | Euclidean distance. |
| `def gaussian1D(x, mean = 0, sigma = 1)` | Adjusts the mean and sigma of 1D gaussian. | A single value. |
| `def gaussian(m, n, sigma = 1)` | Returns a 2D gaussian filter of dimensions `mxn` (**m and n both should be odd**) with adjustable sigma and mean centered at `(0,0)` i.e. the central pixel. | A numpy 2D array. |
| a | b | c |

### In `modules.py` file
| Name | Input | Output |
| --- | --- | --- |
| `def filter2D(image, kernel)` | Takes a grayscale image and a kernel as numpy 2D arrays. | Convolves the kernel on the image with stride = 1 in both dimensions and returns the output as numpy array. |
| `def scaling(image, sigmag = 3, k = 5)` | Takes a grayscale image, sigma for gaussian and kernel size (`k`). | Performs simple gaussian filtering and returns the output image. |
| `def scaled_bilateral_filtering(image, sigmas = 4, sigmar = 12, sigmag = 3, k = 5)` | `sigmas`, `sigmar` and `sigmar` are *spatial*, *range* and *scaling* parameters respectively as mentioned in the paper. `k` is the neighborhood square size. | Performs scaled bilateral filtering and returns smoothed image as numpy array. |
| `def sharpen(image)` | Sharpens the *grayscale* image using Laplacian filter. | It first filters the image using the *Laplacian*. The filtered image is then added back to the original image using proper scaling. |
| `def edge_sobel(image)` | Takes an image and detects edges using **sobel operators** for X and Y directions. | Returns the gradient magnitude image. |
| a | b | c |
| a | b | c |

### Results and Comments
- Converting to Grayscale using the formula `Gray(i, j) = 0.2989 * R(i, j) + 0.5870 * G(i, j) + 0.1140 * B(i, j)`

| Input | Output |
| --- | --- |
| ![](cavepainting1.JPG) | ![](grayscale.png) |
- Filtering
- to be completed

### Running Instructions
- to be written

> By Soumava Paul (16EE10056) and Swagatam Haldar (16EE10063).
