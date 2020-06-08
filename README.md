# Open Compound Domain Adaptation

[[Project]](https://liuziwei7.github.io/projects/CompoundDomain.html) [[Paper]](https://arxiv.org/abs/1909.03403)

## Overview
`Open Compound Domain Adaptation (OCDA)` is the author's re-implementation of the compound domain adaptator described in:  
"[Open Compound Domain Adaptation](https://arxiv.org/abs/1909.03403)"   
[Ziwei Liu](https://liuziwei7.github.io/)<sup>\*</sup>,&nbsp; [Zhongqi Miao](https://github.com/zhmiao)<sup>\*</sup>,&nbsp; [Xingang Pan](https://xingangpan.github.io/),&nbsp; [Xiaohang Zhan](https://xiaohangzhan.github.io/),&nbsp; [Dahua Lin](http://dahua.me/),&nbsp; [Stella X. Yu](https://www1.icsi.berkeley.edu/~stellayu/),&nbsp; [Boqing Gong](http://boqinggong.info/)&nbsp; (CUHK & Berkeley & Google)&nbsp; 
in IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2020, **Oral Presentation**

<img src='./assets/intro.png' width=900>

Further information please contact [Zhongqi Miao](mailto:zhongqi.miao@berkeley.edu) and [Ziwei Liu](https://liuziwei7.github.io/).

## Requirements
* [PyTorch](https://pytorch.org/) (version >= 0.4.1)
* [scikit-learn](https://scikit-learn.org/stable/)

## Data Preparation

<img src='./assets/dataset.png' width=500>

* We will be publishing the data really soon!

## Getting Started (Training & Testing)

<img src='./assets/pipeline.png' width=900>

### C-Digits

To run experiments on the C-Digits datasets (SVHN -> Multi):
```bash
python main.py --config ./config svhn_bal_to_multi.yaml
```

### C-Faces

* We will be releasing code for C-Faces experiements very soon.

## Reproduced Benchmarks and Model Zoo (We will be releasing reimplemented model very soon.)

### C-Digits (Results may currently have variations, and bugs may appear in current released version.)

|  Source  |    MNIST (C)   |  MNIST-M (C)  |   USPS (C)  |  SymNum (O)  |   Avg. Acc   |      Download      |
| :------: | :------------: | :-----------: | :---------: | :----------: | :----------: | :----------------: |
|   SVHN   |      89.62     |     64.53     |    81.17    |    87.86     |    80.80     |      [model]()     |

### C-Faces (Will update soon.)

|  Source  |    C08 (C)   |    C09 (C)  |    C13 (C)  |    C14 (C)  |    C19 (O)  |   Avg. Acc   |      Download      |
| :------: | :----------: | :---------: | :---------: | :---------: | :---------: | :----------: | :----------------: |
|   C05    |              |             |             |             |             |              |      [model]()     |

## License and Citation
The use of this software is released under [BSD-3](https://github.com/zhmiao/OpenCompoundDomainAdaptation-OCDA/blob/master/LICENSE).
```
@inproceedings{compounddomainadaptation,
  title={Open Compound Domain Adaptation},
  author={Liu, Ziwei and Miao, Zhongqi and Pan, Xingang and Zhan, Xiaohang and Lin, Dahua and Yu, Stella X. and Gong, Boqing},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```
