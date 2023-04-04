# ENGG5104 Assignment 2 - Image Recognition
#### Dependencies:
1. Python 3.8.10
2. numpy 1.24.2
3. Pillow 9.4.0
4. torch 1.13.1+cu116
5. torchvision 0.14.1+cu116

#### Code:
1. code/train.py
    - Main Python code for this assignment

2. code/model.py
    - Python code for Task 1: Implement VGGNet

3. code/loss.py
    - Python code for Task 2: CrossEntropy Loss

4. code/transfroms.py
    - Python code for Task 3 - 4: Conventional Augmentations

5. code/flops.py
    - Python code printing model parameter FLOPs

#### Commands:
1. Run the neural network training:
```bash
cd code
python train.py --seed 42
```

2. Measure the FLOPs of model:
```bash
cd code
python train.py --calculate-flops True
```

#### Outputs:
1. code/Logs/log.txt
    - Output of neural network training

#### Other folders:
1. code/data/
    - Directory storing the generated dataset
	
2. code/dataset/
    - Python code for generating the dataset

#### Remarks:
1. Comment and uncomment somes lines in train.py are needed to test different tasks
	- Task 1 - 2: Keep Line 58-60, 62-63, 87, 159 commented
	- Task 3: Keep Line 62-63, 87, 159 commented
	- Task 4: Keep Line 127 commented
	
2. The tricks used in Task 4 are as follows:
	- Model Architecture: VGGNet-A with dropout (p = 0.7) added in the classifier
	- Weight Initialization: He initialization with uniform distribution
	- Data Augmentation: Added ColorJitter on top of the conventional augmentations
	- Scheduler: Cosine annealing decay
