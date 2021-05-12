# Meta Transfer Learning for Zero Shot Super Resolution

## Environment
- OS : Window 10 Pro
- python : 3.8.5
- CUDA : 11.3
- cudnn : 8.3
- GPU : RTX3070 (-> 120x120 inference time : 5s)

## Implementation
```
pip install -r requirements.txt
python main.py --path={TARGET_IMAGE_PATH}
```

## Demo

100 epochs

<img src="images/Aatrox.png" width="120" height="120" />   120x120

<img src="images/result.png" width="240" height="240" />   240x240

#### This model is scale-invariant.

### reference

[Meta-Transfer Learning for Zero-Shot Super-Resolution](https://arxiv.org/pdf/2002.12213.pdf, "MZSR")
