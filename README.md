# MRCGNN
Codes, datasets and appendix for AAAI-2023 paper "Multi-relational Contrastive Learning Graph Neural Network for Drug-drug Interaction Event Prediction"


![AppVeyor](https://img.shields.io/badge/python-3.7.10-blue)
![AppVeyor](https://img.shields.io/badge/numpy-1.18.5-red)
![AppVeyor](https://img.shields.io/badge/pytorch-1.7.1-brightgreen)
![AppVeyor](https://img.shields.io/badge/torch--geometric-2.0.0-orange)

## Run code
For how to use MRCGNN, we present an example based on the Deng's dataset.

1. To learn drug structural features from drug molecular graphs, you need to change the path in 'drugfeature_fromMG.py' first. If you want to use MRCGNN on your own dataset, please ensure the data in the 'trimnet' fold and the data in the 'codes for MRCGNN' fold are the same.)

```
python drugfeature_fromMG.py
```

2.Training/validating/testing for 5 times and get the average scores of multiple metrics.
```
python 5timesrun.py
```

3. You can see the final results of 5 runs in 'test.txt'

