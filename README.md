PR-FPN: Progressive Spatial and Channel Feature-Refined Pyramid Network for Object Detection
---------------------
This project is based on [mmdetection](https://github.com/open-mmlab/mmdetection), [mmyolo](https://github.com/open-mmlab/mmyolo), [detectron2](https://github.com/facebookresearch/detectron2).

Environment
----------------
```
mmengine==0.7.3
mmcv==2.0.0
mmdet==3.0.0
mmyolo==0.5.0
detectron2==0.6
#dcnv2==0.1.1
dcn_4==1.0.0
```

Install
-------------
Please refer to [mmdetection](https://mmdetection.readthedocs.io/en/latest/get_started.html), [mmyolo](https://mmyolo.readthedocs.io/en/latest/get_started/installation.html), [detectron2](https://detectron2.readthedocs.io/tutorials/install.html), [dcnv4](./DCNv4/README.md) for installation.

Dataset
----------
```
PR-FPN
├── DCNv2
├── DCNv4
├── detectron2
├── mmdetection
├── mmyolo
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
├── faster-rcnn_r50_afpn_1x_coco.py
├── yolov8_n-v61_syncbn_fast_8xb16-300e_coco.py
├── yolov5_n-v61_syncbn_fast_8xb16-300e_coco.py
├── train.py
├── test.py
```
Train
--------------
Single gpu for training:
```shell
CUDA_VISIBLE_DEVICES=0 python ./mmdetection/tools/train.py faster-rcnn_r50_prfpn_1x_coco.py --work-dir ./weight/

python3 ./detectron2/tools/train_net.py --config-file ./detectron2/configs/COCO-Detection/mask_rcnn_R_50_PRFPN_1x.yaml --num-gpus 1
```

Multiple gpus for training:
```shell
CUDA_VISIBLE_DEVICES=0,1 ./mmdetection/tools/dist_train.sh faster-rcnn_r50_prfpn_1x_coco.py 2 --work-dir ./weight/

python3 ./detectron2/tools/train_net.py --config-file ./detectron2/configs/COCO-Detection/mask_rcnn_R_50_PRFPN_1x.yaml --num-gpus 2
```
If you want to train more models, please refer to [train.py](train.py).

Test / Evaluate
-----------
```shell
CUDA_VISIBLE_DEVICES=0 _DEVICES=1 python ./mmdetection/tools/test.py faster-rcnn_r50_prfpn_1x_coco.py <CHECKPOINT_FILE>

python3 ./detectron2/tools/train_net.py --config-file <config.yaml> --num-gpus 1 --eval-only MODEL.WEIGHTS /path/to/model_checkpoint
```

For example,
```shell
CUDA_VISIBLE_DEVICES=0 python ./mmdetection/tools/test.py faster-rcnn_r50_prfpn_1x_coco.py ./weight/prfpn_weight.pth
```
If you want to test more models, please refer to [test.py](test.py).

Citations
------------
If you find PR-FPN useful in your research, please consider citing:
```

```
