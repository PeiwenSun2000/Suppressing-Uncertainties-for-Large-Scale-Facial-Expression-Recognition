# Suppressing-Uncertainties-for-Large-Scale-Facial-Expression-Recognition

An third party re-implementation based on  the original Kai Wang's code

This is Wang Kai's original code.[link](https://github.com/kaiwang960112/Self-Cure-Network)

Wang Kai's manuscript has been accepted by CVPR2020! [link](https://arxiv.org/pdf/2002.10392.pdf)

Wonderful work!!!!! Simple but better performance over RAN.

## Problems and improvement

### Pretrain model matching problem

The orginal code has a problem of matching pretrain model and current model, since they have different layer name. In this code this problem is solved.

### The backbone is extracted from torch.vision

Part of backbone is inserted to the original code, which would cause certain problem. In this code, backbone and training code are totally separated.

### Only resnet18

The original code only support resnet18. But according to paper, vgg16 and IR should also work. In this code, IResnet18 is added to backbones.

More about IResnet18. The pretrained model comes from the training for [insightface/arcface](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)

### Generate feature

We may use the feature directly to other project. Here is the generater.

## TODO

- [ ] VGG16

- [ ] Verify the performance of SCN+RAN

- [ ] Support of pretrain model on MSCeleb

## Train

- Pytorch

  Torch 1.2.0 or higher and torchvision 0.4.0 or higher are required.

- Data Preparation

  Download basic emotions dataset of [RAF-DB](http://www.whdeng.cn/RAF/model1.html#dataset), and make sure it have a structure like following:

```
- datasets/raf-basic/
         EmoLabel/
             list_patition_label.txt
         Image/aligned/
	     train_00001_aligned.jpg
             test_0001_aligned.jpg
             ...
```

- Start Training
Note: you should design your own args all by yourself！！
```
      python train.py --margin_1=0.07
```

--margin_1 denotes the margin in Rank Regularization which is set to 0.15 with batch size 1024 in the paper. Here --margin_1=0.07 with smaller batch size 64[default] in train.py can get similar results.
