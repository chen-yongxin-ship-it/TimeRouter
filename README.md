# TimeRouter
A Unified Framework for Handling Missing Data in Multivariate Time Series Forecasting

## Introduction
This paper focuses on handling missing data in multivariate time series forecasting. TimeRouter dynamically models compensatory relationships between exogenous and endogenous variables, reducing performance degradation caused by missing data. Specifically, we introduce hierarchical dynamic fusion space (HDFS), a novel architecture with embedded routing mechanisms that enable adaptive weight allocation and optimal information fusion. 
<p align="center">
<img src=".\figures\Introduction.png" width = "800" height = "" alt="" align=center />
</p>

## Overall Architecture
The architecture of TimeRouter is as follows: An L-layer single-stream encoder encodes variables into tokens and dynamically fuses features in the Hierarchical Dynamic Fusion Space (HDFS, shown by the orange dashed box). In HDFS, expert processing units (u0, u1, u2) are fully connected across layers and embed a dynamic router mechanism to generate a weight matrix.
<p align="center">
<img src=".\figures\TimeRouter.png" width = "800" height = "" alt="" align=center />
</p>

## Usage 

1. Short-term Electricity Price Forecasting Dataset have alreadly included in "./dataset/EPF". Multivariate datasets can be obtained from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) or [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy).

2. Install Pytorch and other necessary dependencies.
```
pip install -r requirements.txt
```
3. Train and evaluate model. We provide the experiment scripts under the folder ./scripts/. You can reproduce the experiment results as the following examples:

```
bash ./scripts/forecast_missing/EPF/TimeRouter.sh
```

## Main Results

### Short-Term Electricity Price Forecasting

<p align="center">

<img src=".\figures\Result_EPF.png" width = "800" height = "" alt="" align=center />

</p>

### Long-Term Multivariate Time Series Forecasting

<p align="center">

<img src=".\figures\Result_Multivariate.png" width = "800" height = "" alt="" align=center />

</p>

### Handling Missing Values
To further explore TimeRouter's adaptability to complex missing-data scenarios, we conducted experiments in scenarios where historical time series data is missing. Specifically, for endogenous and exogenous variables, we evaluated TimeRouter's generalizability in missing-data scenarios using two strategies:
1) Zero: Fills missing sequences with zero values of the same length.
2) Gaussian: Fills missing sequences with Gaussian-distributed values (sampled from 
N(0,1)) of the same length.

<p align="center">

<img src=".\figures\MissingValue.png" width = "800" alt="" align=center />

</p>

### Visualization of Weight Matrix
To explore the adaptive capacity of the dynamic routing mechanism under different input conditions, we conducted a visualization analysis of the routing weight matrix under three typical missing scenarios.

<p align="center">

<img src=".\figures\WeightMatrix.png" width = "800" alt="" align=center />

</p>

### Increase Variables Missing Rate
To investigate the impact of variable missingness on prediction performance, we conducted experiments on the EPF dataset by systematically controlling the missing rates (20\%-100\%) of both endogenous and exogenous variables.

<p align="center">

<img src=".\figures\MissingRate.png" width = "800" alt="" align=center />

</p>