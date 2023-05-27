# YOLOv8 yordamida Face Detection (Yuz aniqlash) dasturini yaratish
### YOLOv8 texnologiyasi yordamida yaratilgan sun'iy intellekt modeli yuzni aniqlashga yordam beradi.

Buni amalga oshirish uchun quyidagi manbalardan foydalanamiz:

1. [YOLOv8](https://github.com/ultralytics/ultralytics) github
2. [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) dataset

WIDER FACE datasetini train, test va validation datalariga ajratib olamiz va ular yordamida *.yaml fayl shakllantirib olamiz:
```python
# Fayl nomi: face.yaml

path: yolov8/dataset

train: train/images
val: validation/images

nc: 1
names: ['face']
```


```python
from ultralytics import YOLO

# Modelni yuklab olish
model = YOLO("yolov8m.pt")

# Modeldan foydalanish
model.train(data="face.yaml", epochs=10)
```

```bash
pip install ultralytics
```
![yolo](https://cdn-images-1.medium.com/v2/resize:fill:1600:480/gravity:fp:0.5:0.4/1*9gavyPR_Z0NHBm8mu6Z5dA.png)

![face](https://i.pinimg.com/originals/2b/db/ee/2bdbeec2feb61c059e86b4868a970879.jpg)
