Preparation
1. download data from given link and put them under the 'data/' and 'open_challenge/data' folders (you can use 'ln -s' to create a soft link the data)
2. python requirements
   - numpy
   - PyTorch (> 1.0)
   - scipy
   - scikit-image
   - tensorboardX
   - colorama, tqdm, setproctitle

Task1: Implement FlowNet Encoder
1. Implement 'networks/FlowNetE.py'

Task2: Loss Function
1. Implement 'EPELoss' in 'losses.py'
2. run 'run_E.sh'

Task3: Refinement Module
1. Implement 'networks/FlowNetER.py'
2. run 'run_ER.sh'

Task4: Multi-scale Optimization
1. Implement 'networks/FlowNetERM.py'
2. Implement 'MultiscaleLoss' in 'losses.py'
3. run 'run_ERM.sh'

Task5: Open Challenge
1. go into the 'open_challenge' directory, implement 'networks/FlowNetOurs.py' and 'Oursloss' in 'losses.py'
2. you can also modify input transformation code in 'dataset.py'
3. run 'run_ours.sh'

Notation
1. the best validation EPE is printed in the log.
2. FLOPs and params are printed in the log.
3. 'test_*.sh' is used to evalute the trained model

Submission
1. you need to sumbit the codes and trained checkpoints (save in 'work/' automatically)
2. you should only keep '*_model_best.pth.tar' and clean other irrelevant files in 'work/'
3. you should move 'data/' out before 'zip'
4. add a readme file to tell the TAs which tasks you complete.

