# Trajectory-Based Neural Darwinism in Convolutional Neural Networks: Variation, Competition, and Selective Retention
This is the official implementation for Trajectory-Based Neural Darwinism in Convolutional Neural Networks: Variation, Competition, and Selective Retention

# Overview
Understanding how neural networks develop and stabilize internal representations remains a central challenge. Inspired by Edelman’s Neural Darwinism, we introduce the Neuron Darwinian Dynamics System (NDDS), a trajectory-based framework that treats neurons as evolving agents under both local and global selective pressures. We define the Global Darwinian Pressure (GDP) as the population-average neuron fitness, capturing system-wide selection dynamics. Layer-wise analyses show that selective pressure intensifies over training, particularly in deeper layers, reflecting progressive consolidation of high-fitness neurons. Ablation experiments further reveal that removing survived neurons leads to substantial accuracy loss, whereas eliminating low-fitness neurons causes minimal degradation, demonstrating NDDS’s ability to identify functionally critical units. Dynamic trajectory analyses show that survived neurons maintain coherent activity, stronger weights, and higher global Darwinian pressures, while eliminated neurons stagnate. Overall, our results support a Darwinian view of representation learning: networks achieve early-stage redundancy and later-stage specialization, enabling robust and stable task-relevant representations.
# Dependencies
```shell
conda create -n myenv python=3.7
conda activate myenv
conda install -c pytorch pytorch==1.9.0 torchvision==0.10.0
pip install scipy
```

# Datasets
Please download the Tiny-Imagenet Dataset. 

# ResNet18
All used ResNet18 models can be downloaded from here. Please put them in ResNet18().
