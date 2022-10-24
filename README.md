# SynapseReconstruction
 A synapse reconstruction algorithm combining modified Mask R-CNN and adaptive 3D connection algorithm


## Installation

``
conda create -n your_env python=3.6
``

``
conda install --yes --file requirements.txt
``

## Train

### Change different backbone

Change `BACKBONE = "resnet50"`, `BACKBONE = "resnet101"`, 
`BACKBONE = "se_resnet50"`, `BACKBONE = "se_resnet101"` or 
`BACKBONE = "sresnext101"`  in SynapseConfig()

### Change different loss

Change Focal loss and CIoU loss

Use code (model.py) from `line:1375 to 1581` rather than from `line:1322 to 1373`

Use code (model.py) from `line:2612 to 2616` rather than from `line:2608 to 2611`

## Inference and Fusion

## Connection

## Evaluation

