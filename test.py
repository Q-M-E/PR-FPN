import os

# Single gpu
os.system("CUDA_VISIBLE_DEVICES=0 python ./mmdetection/tools/test.py faster-rcnn_r50_prfpn_1x_coco.py ./weight/prfpn_weight.pth")
# os.system("CUDA_VISIBLE_DEVICES=0 python ./mmyolo/tools/test.py yolov5_n-v61_syncbn_fast_8xb16-300e_coco.py ./weight/yolov5_n_prfpn.pth")
# os.system("CUDA_VISIBLE_DEVICES=0 python ./mmyolo/tools/test.py yolov8_n-v61_syncbn_fast_8xb16-300e_coco.py ./weight/yolov8_n_prfpn.pth")
# os.system('python3 ./detectron2/tools/train_net.py --config-file <config.yaml> --num-gpus 1 --eval-only MODEL.WEIGHTS /path/to/model_checkpoint')