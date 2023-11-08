## Little Strokes Fell Great Oaks: Boosting the Hierarchical Features for Multi-exposure Image Fusion

This repository is the official implementation of  [BHFMEF](https://dl.acm.org/doi/10.1145/3581783.3612561)

![image](https://github.com/ZhiyingDu/BHFMEF/assets/111031904/5dacbf20-3bcf-428e-96e2-6cd7489c843a)

## About BHFMEF Quantitative and Qualitative Experimental Results：

The prevalent quantitative evaluation metric for multi-exposure image fusion in computer vision involves comparing the fusion results with under-exposed and over-exposed images.

Therefore, If you want to get the best quantitative results, it would be better to set the weights of the calculated fusion image and each source image in MEFloss.py to **0.5, 0, 0, 0.5**. If you want to get the best qualitative results, the above weights are set to **0.25, 0.25, 0.25, 0.25** might be better.

If you have any question，please contact：[a_chao2001@163.com](mailto:a_chao2001@163.com).

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
