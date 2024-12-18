# FloodScan
Georgia Tech CS 7643 (Deep Learning) Final Project: Flood Mapping Semantic Segmentation using a U-Net Model with Feature Representations of Sentinel-1 and Sentinel-2 Data

Recent flood events around the world have brought the importance of understanding flood behavior to the forefront. Synthetic aperture radar (SAR) or multispectral (MS) imagery is commonly used to perform flood extent segmentation. The Sen1Floods11 dataset provides Sentinel-1 (S1) and Sentinel-2 (S2) imagery which are hand-labeled for water bodies. Recent papers have approached this problem differently, applying machine learning (ML) or fully convolutional neural networks (FCNNs) to this problem. However, they are not directly comparable because of differences in evaluation metrics.
We selected four feature sets based on state-of-the-art performances from these papers. We then train U-Net models from scratch using grid-search hyperparameter optimization and compare the performance of the four feature sets with each other. We were able to achieve Intersection over Union (IoU) results without tuning on the test data set that exceeded the baseline presented in the original Sen1Floods11 paper, which relies solely on raw S1 and S2 bands.
Our findings showed that while FCNNs are unable to outperform traditional ML models with an engineered feature space, engineered features do improve the performance of FCNNs. The results highlight the importance of leveraging domain knowledge of spectroscopy in satellite flood segmentation problems.

Here is our <a href="FloodScan.pdf" target="_blank">paper</a>.

FloodScan: Flood Mapping Semantic Segmentation Using a U-Net Model with Feature Representations of Sentinel-1 and Sentinel-2 Data

Yu Liu, Max Romanelli, Neelima Srivastava, Re ́mi Toutin

Georgia Institute of Technology

{yliu3577, mromanelli6, nsrivastava44, rtoutin3}@gatech.edu

written December 12, 2024
