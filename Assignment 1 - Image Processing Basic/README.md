# ENGG5104 Assignment 1 - Image Processing Basic
#### Dependencies:
1. Python 3.8.8
2. numpy 1.20.1
3. opencv-python 4.7.0.68
4. matplotlib 3.3.4

#### Code:
1. code/runHPF.py
    - Python code for Task 1: High-pass Filter with Fourier Transform

2. code/runTransform.py
    - Python code for Task 2: Image Affine Transformation

3. code/runHistEqualization.py
    - Python code for Task 3: Local Histogram Equalization

4. code/runLineDet.py
    - Python code for Task 4: Line Detection with Hough Transform

5. code/runBilateral.py
    - Python code for Task 5: Joint Bilateral Filter

Type the following commands to run the code:
```bash
cd code
python \<filename\>.py
```

#### Outputs:
1. code/results/hpf_fourier.png
    - Resulting image for Task 1
    - Compared to the original image, the edges are sharpened, and the remaining parts are removed

2. code/results/affine_result.png
    - Resulting image for Task 2

3. code/results/HistoEqualization.jpg
    - Resulting image after histogram equalization for Task 3

4. code/results/LocalHistoEqualization.jpg
    - Resulting image after local histogram equalization for Task 3

5. code/results/cdf_original.png
    - Cumulative distribution function of pixel intensity of the original image for Task 3

6. code/results/cdf_hist_equalization.png
    - Cumulative distribution function of pixel intensity of HistoEqualization.jpg for Task 3
    - Compared to the original image, the pixel intensity is more evenly distributed
    - However, for the middle region, the contrast is still insufficiently enhanced

7. code/results/cdf_local_hist_equalization.png
    - Cumulative distribution function of pixel intensity of LocalHistoEqualization.jpg for Task 3
    - Compared to histogram equalization, the contrast in the middle region is further enhanced

8. code/results/line_det.png
    - Resulting image for Task 4

9. code/results/Billateral.jpg
    - Resulting image after applying bilateral filter for Task 5

10. code/results/JointBillateral.jpg
    - Resulting image after applying joint bilateral filter for Task 5

11. code/results/Bilateral_diff.jpg
    - Difference between the above two images, converted to gray color for better visualization
    - From this image, we can see that the edges are better preserved with the joint bilateral filter

#### Other Folder:
1. code/misc/
    - Input images for the tasks

#### Remark:
1. The code for Task 3 and 5 may take a few minutes to run