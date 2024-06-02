import os

# Multiple gpus
os.system("CUDA_VISIBLE_DEVICES=0,1 ./mmdetection/tools/dist_train.sh faster-rcnn_r50_prfpn_1x_coco.py 2 --work-dir ./weight/")
# os.system("CUDA_VISIBLE_DEVICES=0,1 ./mmyolo/tools/dist_train.sh yolov5_n-v61_syncbn_fast_8xb16-300e_coco.py 2 --work-dir ./weight/")
# os.system("CUDA_VISIBLE_DEVICES=0,1 ./mmyolo/tools/dist_train.sh yolov8_n-v61_syncbn_fast_8xb16-300e_coco.py 2 --work-dir ./weight/")
# os.system('python3 ./detectron2/tools/train_net.py --config-file ./detectron2/configs/COCO-Detection/faster_rcnn_R_50_PRFPN_1x.yaml --num-gpus 2')
# os.system('python3 ./detectron2/tools/train_net.py --config-file ./detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_PRFPN_1x.yaml --num-gpus 2')
# os.system('python3 ./detectron2/tools/train_net.py --config-file ./detectron2/configs/COCO-PanopticSegmentation/panoptic_prfpn_R_50_1x.yaml --num-gpus 2')
# os.system('python3 ./detectron2/tools/train_net.py --config-file ./detectron2/configs/Cityscapes/mask_rcnn_R_50_PRFPN.yaml --num-gpus 2')


# Single gpu
# os.system("CUDA_VISIBLE_DEVICES=0 python ./mmdetection/tools/train.py faster-rcnn_r50_prfpn_1x_coco.py --work-dir ./weight/")
# os.system('python3 ./detectron2/tools/train_net.py --config-file ./detectron2/configs/COCO-Detection/faster_rcnn_R_50_PRFPN_1x.yaml --num-gpus 1')
# os.system('python3 ./detectron2/tools/train_net.py --config-file ./detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_PRFPN_1x.yaml --num-gpus 1')
# os.system('python3 ./detectron2/tools/train_net.py --config-file ./detectron2/configs/COCO-PanopticSegmentation/panoptic_prfpn_R_50_1x.yaml --num-gpus 1')
# os.system('python3 ./detectron2/tools/train_net.py --config-file ./detectron2/configs/Cityscapes/mask_rcnn_R_50_PRFPN.yaml --num-gpus 1')
# os.system("CUDA_VISIBLE_DEVICES=0 python ./mmyolo/tools/train.py yolov5_n-v61_syncbn_fast_8xb16-300e_coco.py --work-dir ./weight/")
# os.system("CUDA_VISIBLE_DEVICES=0 python ./mmyolo/tools/train.py yolov8_n-v61_syncbn_fast_8xb16-300e_coco.py --work-dir ./weight/")

