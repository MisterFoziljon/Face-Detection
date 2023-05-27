# YOLOv8 yordamida Face Detection (Yuz aniqlash) dasturini yaratish
![face](https://i.pinimg.com/originals/2b/db/ee/2bdbeec2feb61c059e86b4868a970879.jpg)

### YOLOv8 texnologiyasi yordamida yaratilgan sun'iy intellekt modeli yuzni aniqlashga yordam beradi.
#### Buni amalga oshirish uchun quyidagi manbalardan foydalanamiz:

1. #### [YOLOv8](https://github.com/ultralytics/ultralytics) github
![yolo](https://cdn-images-1.medium.com/v2/resize:fill:1600:480/gravity:fp:0.5:0.4/1*9gavyPR_Z0NHBm8mu6Z5dA.png)

2. #### [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) dataset
![wider](https://machinelearningmastery.ru/img/0-507943-363418.jpeg)

WIDER FACE datasetini ```train```, ```test``` va ```validation``` datalariga ajratib olamiz va ular yordamida *.yaml fayl shakllantirib olamiz:

```python
# Fayl nomi: face.yaml

path: yolov8/dataset

train: train/images
val: validation/images
test: test/images

nc: 1
names: ['face']
```
Modelni qurish uchun mavjud YOLOv8 modellaridan foydalanamiz. Quyida ularning ro'yxati keltirilgan:

| Model                                                                                | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------------------------------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) | 640                   | 37.3                 | 80.4                           | 0.99                                | 3.2                | 8.7               |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt) | 640                   | 44.9                 | 128.4                          | 1.20                                | 11.2               | 28.6              |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) | 640                   | 50.2                 | 234.7                          | 1.83                                | 25.9               | 78.9              |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt) | 640                   | 52.9                 | 375.2                          | 2.39                                | 43.7               | 165.2             |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt) | 640                   | 53.9                 | 479.1                          | 3.53                                | 68.2               | 257.8             |
 
Hosil bo'lgan face.yaml faylni train qilish uchun quyidagi kodga beramiz:
```python
from ultralytics import YOLO

# Modelni yuklab olish
model = YOLO("yolov8m.pt")

# Modeldan foydalanib train qilish
model.train(data="face.yaml", epochs=10)
```

Yoki terminal yordamida quyidagi kodni ishlatishingiz mumkin:
```bash
yolo train model=yolov8m.pt data=face.yaml epochs=10 imgsz=640 batch=16
```
 
Train qilish natijasida hosil bo'lgan model(best.pt) tezligi biroz pastroq bo'lgani uchun uni quyidagi modellarga o'tkazamiz va ularda modelni ishlashini sinovdan o'tkazamiz:

1. onnx model ([ONNX](https://onnx.ai/) - Open Neural Network Exchange)
2. tflite model ([TensorFlow](https://www.tensorflow.org/lite/guide?hl=ru) Lite) 
3. engine model ([TensorRT](https://developer.nvidia.com/tensorrt))

