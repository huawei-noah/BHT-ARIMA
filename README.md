# BHT-ARIMA

A tensor decomposition-based time series forecasting algorithm, which tactically incorporates the unique advantages of Hankelization, low-rank Tucker decomposition and ARIMA into a unified framework.  
More details (including parameter settings) refer to [the original paper](https://arxiv.org/abs/2002.12135).

### Paper
- **"[Block Hankel Tensor ARIMA for Multiple Short Time Series Forecasting](https://arxiv.org/abs/2002.12135)", AAAI-20**

### Datasets
  
Traffic dataset. The traffic data is originally collected from California department of transportation 1 and describes the road occupy rate of Los Angeles County highway network.We here use the same subset used in (Yu, Yin, and Zhu 2017) which selects **228 sensors** randomly. And We take **the first 40 time points** of them as data of our demo

### Getting Started

#### Prerequisites  

- python 3.5+
- python libraries
  - tensorly
  - scipy
  - numpy
  - pandas 

#### Run

```python
python main.py
```



### License
© Contributors, 2019. Licensed under an [MIT](LICENSE) license.
