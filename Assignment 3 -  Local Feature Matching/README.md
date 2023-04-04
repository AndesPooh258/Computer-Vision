# ENGG5104 Assignment 3 - Local Feature Matching
#### Dependencies:
1. Python 3.8.8
2. numpy 1.20.1
3. opencv-python 4.7.0.72

#### Code:
1. code/proj2.py
    - Main Python code for this assignment

2. code/match_functions.py
    - Python code for the local feature-matching algorithm

3. code/utils.py
    - Python code for visualization and evaluation

#### Command:
1. Run the local feature-matching algorithm:
```bash
cd code
python proj2.py
```

#### Outputs:
1. code/eval_result.png
    - Evaluation result of matches

2. code/eval_truth.png
    - Evaluation result of interest point from cheat_interest_points()

3. code/vis_arrows_result.png
    - Arrows visualization result of feature matching

4. code/vis_arrows_truth.png
    - Arrows visualization result of interest point from cheat_interest_points()

5. code/vis_dots_result.png
    - Dots visualization result of feature matching

6. code/vis_dots_truth.png
    - Dots visualization result of interest point from cheat_interest_points()

- Output 1 is obtained only if images with ground truth correspondences are used
- Output 2, 4, and 6 are obtained only if cheat_interest_points() is used

#### Other Files:
1. data/
    - Directory storing the test images
    - The Folders 'Episcopal Gaudi', 'Mount Rushmore', 'Notre Dame' are images with ground truth correspondences
    - The Folders 'Sushi', 'PTCG' are images without ground truth correspondences

#### Remark:
1. Modify some lines in proj2.py are needed to test different setting 
	- Modify the test_case variable in Line 8 to change test images
	- Test get_features() and match_features() with cheat_interest_points():
        => Keep Line 33-34 commented
    - Test all functions for image with ground truth correspondences:
        => Keep Line 38, 45-46, 50 commented
    - Test all functions for image without ground truth correspondences:
        => Keep Line 38, 45-46, 50, 91-92 commented
