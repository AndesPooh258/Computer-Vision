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
   - Implement 'networks/FlowNetE.py'

Task2: Loss Function
   - Implement 'EPELoss' in 'losses.py'
   - run 'run_E.sh'

Task3: Refinement Module
   - Implement 'networks/FlowNetER.py'
   - run 'run_ER.sh'

Task4: Multi-scale Optimization
   - Implement 'networks/FlowNetERM.py'
   - Implement 'MultiscaleLoss' in 'losses.py'
   - run 'run_ERM.sh'

Task5: Open Challenge
   - go into the 'open_challenge' directory, implement 'networks/FlowNetOurs.py' and 'Oursloss' in 'losses.py'
   - you can also modify input transformation code in 'dataset.py'
   - run 'run_ours.sh'

Notation
   - the best validation EPE is printed in the log.
   - FLOPs and params are printed in the log.
   - 'test_*.sh' is used to evalute the trained model

Submission
   - you need to sumbit the codes and trained checkpoints (save in 'work/' automatically)
   - you should only keep '*_model_best.pth.tar' and clean other irrelevant files in 'work/'
   - you should move 'data/' out before 'zip'
   - add a readme file to tell the TAs which tasks you complete.

