
## Installation

Please refer the following commands to prepare the environment:

  conda create --name pyt
  conda activate pyt
  conda install pytorch=1.10.0 torchvision torchaudio cudatoolkit=10.2 -c pytorch
  pip install fiftyone tensorboard
  pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html
  pip install ftfy seaborn pycocotools terminaltables regex isort lvis opencv-python PyYAML
  conda install tensorflow-estimator==2.1.0

If your node could not connect to the internet(Some HPC node is not connected to the internet), you should use run the tools/mmcv_installation.sh to install the mmcv


## Getting Started
There are some projects in this codebase:
1. code for the paper Efficient Feature Distillation for Zero-shot Detection(EZSD)
2. the exploration for using the ViT feature in Detector backbone 
3. The Proposal selector
4. The New RPN
5. The few-shot training pipeline

to fully reproduce the EZSD, you have to run the following steps:
1. finetuning the CLIP
2. generating the CLIP proposal
3. extract the CLIP feature using the CLIP proposal
4. Train an detector with the CLIP feature