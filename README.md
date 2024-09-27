<div align="center">
<h1>SSAClassifier </h1>
<h3>Semantic and Spatial Adaptive Pixel-level Classifier for Semantic Segmentation</h3>

[Xiaowen Ma](https://scholar.google.com/citations?hl=zh-CN&user=UXj8Q6kAAAAJ)<sup>1,2</sup>, [Zhenliang Ni](https://scholar.google.com/citations?user=2urTmpkAAAAJ&hl=zh-CN&oi=sra)<sup>1</sup>, [Xinghao Chen](https://scholar.google.com/citations?user=tuGWUVIAAAAJ&hl=zh-CN&oi=ao)<sup>1</sup>

<sup>1</sup> Huawei Noah’s Ark Lab, <sup>2</sup> Zhejiang University

Paper: ([this](https://arxiv.org/abs/2405.06525))

</div>

## 🔥 News

- **`2024/09/26`**: **SSAClassifier is accepted by NeurIPS2024!**

  

## 📷 Introduction

![](net.png)

SSAClassifier is an effecient and powerful pixel-level classifier, which significantly improves the segmentation performance of various baselines with a negligible increase in computational cost. It has three key parts: semantic prototype adaptation (SEPA), spatial prototype adaptation (SPPA), and online multi-domain distillation. 



## 🏆Performance

#### ADE20K

**Iters:** 160000	**Input size:** 512x512	**Batch size:** 16

- General models

  | +SSAClassifier |                           Backbone                           | Latency (ms) | Params(M) | Flops (G) | mIoU (ss) |
  | :------------: | :----------------------------------------------------------: | :----------: | --------- | :-------: | :-------: |
  |     OCRNet     | [HRNet-W48](https://download.openmmlab.com/pretrain/third_party/hrnetv2_w48-d2186c55.pth) |     69.3     | 8.7       |   165.0   |   47.67   |
  |    UperNet     | [Swin-T](https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth) |     54.3     | 61.1      |   236.3   |   47.56   |
  |   SegFormer    | [MiT-B5](https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth) |     70.1     | 82.3      |   52.6    |   50.74   |
  |    UperNet     | [Swin-L](https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window7_224_22k_20220412-aeecf2aa.pth) |    107.3     | 234.9     |   405.2   |   52.69   |
  |  ViT-Adapter   |    [ViT-Adapter-L](https://github.com/czczup/ViT-Adapter)    |    284.9     | 364.9     |   616.3   |   55.39   |

- Light weight models

  | +SSAClassifier |                           Backbone                           | Iters  | Latency (ms) | Params (M) | Flops (G) | mIoU (ss) |
  | :------------: | :----------------------------------------------------------: | :----: | :----------: | ---------- | :-------: | :-------: |
  |   AFFormer-B   | [AFFormer-B](https://github.com/dongbo811/AFFormer?tab=readme-ov-file) | 160000 |     26.0     | 3.3        |    4.4    |   42.74   |
  |  SeaFormer-B   | [SeaFormer-B](https://github.com/fudan-zvg/SeaFormer/tree/main/seaformer-cls) | 160000 |     27.3     | 8.8        |    1.8    |   42.46   |
  |   SegNext-T    | [MSCAN-T](https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_t_20230227-119e8c9f.pth) | 160000 |     23.3     | 4.6        |    6.3    |   43.90   |
  |  SeaFormer-L   | [SeaFormer-L](https://github.com/fudan-zvg/SeaFormer/tree/main/seaformer-cls) | 160000 |     29.9     | 14.2       |    6.4    |   45.36   |
  |    CGRSeg-B    | [EfficientFormerV2-S2](https://github.com/snap-research/EfficientFormer) | 160000 |     36.0     | 19.3       |    7.6    |   47.10   |
  |    CGRSeg-L    | [EfficientFormerV2-L](https://github.com/snap-research/EfficientFormer) | 160000 |     42.6     | 35.8       |   14.8    |   49.00   |

#### COCO-Stuff-10K

**Iters:** 80000	**Input size:** 512x512	**Batch size:** 16

- General models

  | +SSAClassifier |                           Backbone                           | Latency (ms) | Params(M) | Flops (G) | mIoU (ss) |
  | :------------: | :----------------------------------------------------------: | :----------: | --------- | :-------: | :-------: |
  |     OCRNet     | [HRNet-W48](https://download.openmmlab.com/pretrain/third_party/hrnetv2_w48-d2186c55.pth) |     69.3     | 8.7       |   165.0   |   37.94   |
  |    UperNet     | [Swin-T](https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth) |     54.3     | 61.1      |   236.3   |   42.30   |
  |   SegFormer    | [MiT-B5](https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth) |     70.1     | 82.3      |   52.6    |   45.55   |
  |    UperNet     | [Swin-L](https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window7_224_22k_20220412-aeecf2aa.pth) |    107.3     | 234.9     |   405.2   |   48.94   |
  |  ViT-Adapter   |    [ViT-Adapter-L](https://github.com/czczup/ViT-Adapter)    |    284.9     | 364.9     |   616.3   |   51.2    |

- Light weight models

  | +SSAClassifier |                           Backbone                           | Iters | Latency (ms) | Flops (G) | mIoU (ss) |
  | :------------: | :----------------------------------------------------------: | :---: | :----------: | :-------: | :-------: |
  |   AFFormer-B   | [AFFormer-B](https://github.com/dongbo811/AFFormer?tab=readme-ov-file) | 80000 |     26.0     |    4.4    |   36.40   |
  |  SeaFormer-B   | [SeaFormer-B](https://github.com/fudan-zvg/SeaFormer/tree/main/seaformer-cls) | 80000 |     27.3     |    1.8    |   35.92   |
  |   SegNext-T    | [MSCAN-T](https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_t_20230227-119e8c9f.pth) | 80000 |     23.3     |    6.3    |   38.91   |
  |  SeaFormer-L   | [SeaFormer-L](https://github.com/fudan-zvg/SeaFormer/tree/main/seaformer-cls) | 80000 |     29.9     |    6.4    |   38.48   |

#### PASCAL-Context

**Iters:** 80000	**Input size:** 480x480	**Batch size:** 16

- General models

  | +SSAClassifier |                           Backbone                           | Latency (ms) | Params (M) | Flops (G) | mIoU (ss) |
  | :------------: | :----------------------------------------------------------: | :----------: | ---------- | :-------: | :-------: |
  |     OCRNet     | [HRNet-W48](https://download.openmmlab.com/pretrain/third_party/hrnetv2_w48-d2186c55.pth) |     69.3     | 8.7        |   143.3   |   50.21   |
  |    UperNet     | [Swin-T](https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth) |     54.3     | 61.1       |   207.7   |   55.11   |
  |   SegFormer    | [MiT-B5](https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth) |     70.1     | 82.3       |   45.8    |   59.14   |
  |    UperNet     | [Swin-L](https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window7_224_22k_20220412-aeecf2aa.pth) |    107.3     | 234.9      |   363.2   |   61.83   |
  |  ViT-Adapter   |    [ViT-Adapter-L](https://github.com/czczup/ViT-Adapter)    |    284.9     | 364.9      |   616.3   |   66.05   |

- Light weight models

  | +SSAClassifier |                           Backbone                           | Latency (ms) | Flops (G) | mIoU (ss) |
  | :------------: | :----------------------------------------------------------: | :----------: | :-------: | :-------: |
  |   AFFormer-B   | [AFFormer-B](https://github.com/dongbo811/AFFormer?tab=readme-ov-file) |     26.0     |    4.4    |   49.72   |
  |  SeaFormer-B   | [SeaFormer-B](https://github.com/fudan-zvg/SeaFormer/tree/main/seaformer-cls) |     27.3     |    1.8    |   47.00   |
  |   SegNext-T    | [MSCAN-T](https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_t_20230227-119e8c9f.pth) |     23.3     |    6.3    |   52.58   |
  |  SeaFormer-L   | [SeaFormer-L](https://github.com/fudan-zvg/SeaFormer/tree/main/seaformer-cls) |     29.9     |    6.4    |   49.66   |



## 📚 Use example

- Environment

  ```shell
  conda create --name ssa python=3.8 -y
  conda activate ssa
  pip install torch==1.8.2+cu102 torchvision==0.9.2+cu102 torchaudio==0.8.2
  pip install timm==0.6.13
  pip install mmcv-full==1.7.0
  pip install opencv-python==4.1.2.30
  pip install "mmsegmentation==0.30.0"
  ```

  SSAClassifier is built based on [mmsegmentation-0.30.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.30.0), which can be referenced for data preparation.

- Train

  ```shell
  # Single-gpu training
  python train.py configs/swin/upernet_swin_tiny_ade20k_ssa.py
  
  # Multi-gpu (4-gpu) training
  bash dist_train.sh configs/swin/upernet_swin_tiny_ade20k_ssa.py 4
  ```

- Test

  ```shell
  # Single-gpu testing
  python test.py configs/swin/upernet_swin_tiny_ade20k_ssa.py ${CHECKPOINT_FILE} --eval mIoU
  
  # Multi-gpu (4-gpu) testing
  bash dist_test.sh configs/swin/upernet_swin_tiny_ade20k_ssa.py ${CHECKPOINT_FILE} 4 --eval mIoU
  ```

- Benchmark

  ```shell
  python benchmark.py configs/swin/upernet_swin_tiny_ade20k_ssa.py ${CHECKPOINT_FILE} --repeat-times 5
  ```



## 🌟Citation

If you are interested in our work, please consider giving a 🌟 and citing our work below. 

```
@misc{ssaclassifier,
   title={Semantic and Spatial Adaptive Pixel-level Classifier for Semantic Segmentation}, 
   author={Xiaowen Ma and Zhenliang Ni and Xinghao Chen},
   year={2024},
   eprint={2405.06525},
   archivePrefix={arXiv},
   primaryClass={cs.CV}
}
```



## 💡Acknowledgment

Thanks to previous open-sourced repo:
[SeaFormer](https://github.com/fudan-zvg/SeaFormer/tree/main) [CAC](https://github.com/tianzhuotao/CAC) [AFFormer](https://github.com/dongbo811/AFFormer) [SegNeXt](https://github.com/Visual-Attention-Network/SegNeXt) [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/tree/v0.30.0) [CGRSeg](https://github.com/nizhenliang/CGRSeg) [ViT-Adapter](https://github.com/czczup/ViT-Adapter)

