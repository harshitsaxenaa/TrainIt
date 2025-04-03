# How I Fine-Tuned YOLOv5

## 1. Weapon Detection Model Training  
```bash
python train.py --img 640 --batch 16 --epochs 50 --data custom_weapon.yaml --weights yolov5n.pt --name weapon_model


## 2. Human Detection Model Training
python train.py --img 640 --batch 16 --epochs 50 --data custom_human.yaml --weights yolov5n.pt --name human_detection_model
