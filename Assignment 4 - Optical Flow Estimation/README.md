# ENGG5104 Assignment 4 - Optical Flow Estimation
#### Dependencies:
1. Python 3.8.10
2. numpy 1.24.2
3. torch 1.13.1+cu116
4. torchvision 0.14.1+cu116
5. scipy 1.10.1
6. tensorboardX 2.6
7. colorama 0.4.6
8. tqdm 4.65.0
9. setproctitle 1.3.2

#### Code:
1. code/networks/FlowNetE.py
    - Python code for Task 1: Implement FlowNet Encoder

2. code/losses.py
    - Python code for Task 2, 4: Loss Function

3. code/networks/FlowNetER.py
    - Python code for Task 3: Refinement Module

4. code/networks/FlowNetERM.py
    - Python code for Task 4: Multi-scale Optimization

5. code/open_challenge/networks/FlowNetOurs.py
    - Python code for the model of Task 5: Open Challenge

6. code/open_challenge/losses.py
    - Python code for the loss function of Task 5: Open Challenge

#### Commands:
1a. Train the FlowNet:
```bash
cd code
source run_E.sh
```

1b. Test the trained FlowNet:
```bash
cd code
source test_E.sh
```

2a. Train the FlowNet with refinement module:
```bash
cd code
source run_ER.sh
```

2b. Test the trained FlowNet with refinement module:
```bash
cd code
source test_ER.sh
```

3a. Train the FlowNet with multiscale optimization:
```bash
cd code
source run_ERM.sh
```

3b. Test the trained FlowNet with multiscale optimization:
```bash
cd code
source test_ERM.sh
```

3a. Train the FlowNet for open challenge:
```bash
cd code/open_challenge
source run_ours.sh
```

3b. Test the trained FlowNet for open challenge:
```bash
cd code/open_challenge
source test_ours.sh
```

#### Outputs:
1. code/work/\<model_name\>_model_best.pth.tar, code/open_challenge/work/FlowNetOurs_model_best.pth.tar
    - Trained best models for evaluation

#### Other folders:
1. code/data/, code/open_challenge/data/
    - Directory storing the dataset

2. code/utils/, code/open_challenge/utils/
    - Python code for utility functions

#### Remarks:
1. Completed Tasks: Task 1 - 5
2. The tricks used in Task 5 are as follows:
	- Used deeper convolutional layer with additional feature refinement
    - Used dilated convolution in convolutional layer with 3x3 filters to increase coverage