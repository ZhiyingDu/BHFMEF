## Little Strokes Fell Great Oaks: Boosting the Hierarchical Features for Multi-exposure Image Fusion

This repository is the official implementation of  [BHFMEF](https://arxiv.org/abs/2404.06033)[ACM MM, 2023]  
**[AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning](https://arxiv.org/abs/2404.06033)** 
</br>
[Pan Mu](https://panmu123.github.io/),
[Zhiying Du](https://zhiyingdu.github.io/),
[Jinyuan Liu*](https://github.com/JinyuanLiu-CV),
[Cong Bai](https://homepage.zjut.edu.cn/congbai/)
(*Corresponding Author)

<!-- [Arxiv Report](https://arxiv.org/abs/2404.06033) | [Project Page](https://github.com/ZhiyingDu/BHFMEF) -->
[![arXiv](https://img.shields.io/badge/arXiv-2307.04725-b31b1b.svg)](https://arxiv.org/abs/2404.06033)

![image](https://github.com/ZhiyingDu/BHFMEF/assets/111031904/5dacbf20-3bcf-428e-96e2-6cd7489c843a)

## Dependency
Our code is based on the PyTorch library
* torch==1.11.0
* torchaudio==0.11.0
* torchvision==0.12.0

## Training
### Dataset File structure
To train/test the dataset should be defined as follows:

```
[Your Own data path]─┐           ├─► HR_over ──► *.png
                     ├─► train ──│
                     │           ├─► HR_under──► *.png
                     │
                     │           ├─► testA ─► *.png
                     ├─► test  ──│                  
                                 │─► testB ─► *.png
```
You can change the path of data in option.py:
```shell
parser.add_argument('--dir_train', type=str, default='/data/***/Dataset/train/',
                    help='training dataset directory')
parser.add_argument('--dir_test', type=str, default='/data/***/Dataset/test/',
                    help='test dataset directory')
```
## About BHFMEF Quantitative and Qualitative Experimental Results：

The prevalent quantitative evaluation metric for multi-exposure image fusion in computer vision involves comparing the fusion results with under-exposed and over-exposed images.

Therefore, If you want to get the best quantitative results, it would be better to set the weights of the calculated fusion image and each source image in MEFloss.py to **0.5, 0, 0, 0.5**. If you want to get the best qualitative results, the above weights are set to **0.25, 0.25, 0.25, 0.25** might be better.

If you have any question，please feel free to issue.

## Citation

if you find our repo useful for your research, please cite us:

```
@inproceedings{mu2023little,
  title={Little Strokes Fell Great Oaks: Boosting the Hierarchical Features for Multi-Exposure Image Fusion},
  author={Mu, Pan and Du, Zhiying and Liu, Jinyuan and Bai, Cong},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={2985--2993},
  year={2023}
}
```

<div align="center"> 
  Visitor count<br>
  <img src="https://profile-counter.glitch.me/ZhiyingDu/count.svg" /> 
</div>
