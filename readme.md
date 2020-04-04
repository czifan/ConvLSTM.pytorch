# ConvLSTM.pytorch

This repository is an unofficial pytorch implementation of 
[Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](https://arxiv.org/abs/1506.04214v1).
We reimplement the experiments in the paper based on the MovingMNIST dataset, which is followed by [Github](https://github.com/jhhuang96/ConvLSTM-PyTorch).
Detailed understanding is available on my [Blog](https://www.cnblogs.com/CZiFan/).

## Requirements
- Pytorch>=0.4.0
- CPU or GPU
- Other packages can be installed with the following instruction:
```
pip install requirements.txt
```
  
## Quick start
Running the code with the following command, and the '--config' parameter represents different network architectures.
```
python main.py --config 5x5_5x5_128_5x5_64_5x5_64
```

## Results
| Model | Parameters(M) | Flops(G) | MSELoss |
|---|---|---|---|
| ConvLSTM(5x5-64)-5x5-256 | | | | 
| ConvLSTM(5x5-64)-5x5-128-5x5-128| 5.74 | 940.55 | | 
| ConvLSTM(5x5-64)-5x5-128-5x5-64-5x5-64| 4.51 | 739.23 | |


## Citation

```
@inproceedings{xingjian2015convolutional,
  title={Convolutional LSTM network: A machine learning approach for precipitation nowcasting},
  author={Xingjian, SHI and Chen, Zhourong and Wang, Hao and Yeung, Dit-Yan and Wong, Wai-Kin and Woo, Wang-chun},
  booktitle={Advances in neural information processing systems},
  pages={802--810},
  year={2015}
}
```