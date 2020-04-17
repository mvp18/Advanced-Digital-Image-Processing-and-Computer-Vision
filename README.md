# Advanced-Digital-Image-Processing-and-Computer-Vision
This repository is the storehouse of all the codes that [we](#contributors) have written for the course assignments of *Advanced Digital Image Processing and Computer Vision (CS60052)* at IIT Kharagpur, for the session of Spring 2020.

Here a brief outline of the **main** algorithms implemented is given.

## 1. Basic Image Processing Algorithms
- Scaled Bilateral Filtering. [*[Reference Paper]*](https://github.com/mvp18/Advanced-Digital-Image-Processing-and-Computer-Vision/blob/master/Basic%20Image%20Processing%20Algorithms/ScaledBilateralFilter.pdf) 
- Harris Corner Detection.
- Morphological Operations.

## 2. Projective Geometry (Single view Geometry)
- Vanishing line of an image (or a view) in the *2D* homogenous space P<sup>2</sup> by finding intersections of pairs of parallel lines.
- Affine Rectification of an image by finding the *homography* that sends the vanishing line to the line at infinity.

## 3. Stereogeometry (Two view Geometry)
- Matching similar feature points (using SIFT) in two images of the *same scene* captured from *separate views*.
- Finding the fundamental matrix (F), camera parameters and epipoles & epipolar lines.
- Estimating (relative) 3D depth of the matched feature points in the scene.

## 4. Color Image Processing
- Finding the dominant color in a RGB color image using K-Means Clustering in the x-y 2D chromaticity plane.
- Transferring the dominant color to another image via color space transformations. [*[Reference Paper]*](https://github.com/mvp18/Advanced-Digital-Image-Processing-and-Computer-Vision/blob/master/Color%20Image%20Processing/relevant%20papers/ColorTransfer_2001_global(slide_algo).pdf)

## 5. Range Image Processing
- Point 1.
- Point 2.
- Point3.

# Running Instructions and Requirements
A detailed description of the functions and methods implemented along with **running instructions** and **requirements info** can be found in the respective documentation files (*pdf*) inside each folder. As an example, this is the [documentation](https://github.com/mvp18/Advanced-Digital-Image-Processing-and-Computer-Vision/blob/master/Stereogeometry/submission%20folder/Assignment-%203%20Documentation.pdf) of the third assignment (Stereogeometry).

# Contributors
- [Soumava Paul](https://mvp18.github.io/)
- [Swagatam Haldar](https://github.com/swag2198)

[Equal contributions for assignments 1 to 3. The last two had individual submissions.]
