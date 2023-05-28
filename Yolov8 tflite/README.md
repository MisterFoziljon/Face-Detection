# YOLOv8-tflite

`YOLOv8` yordamida train qilingan modelni tflite formatga o'tkazish va undan foydalanish.


[![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fatrox%2Fsync-dotenv%2Fbadge&style=flat)](https://github.com/MisterFoziljon/Face-Detection)
[![Python Version](https://img.shields.io/badge/Python-3.8--3.11-FFD43B?logo=python)](https://github.com/MisterFoziljon/Face-Detection)


# Dastur ishlashi uchun kerakli muhitni yaratib olamiz (environments):

1. `CUDA` [`CUDA official website`](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#download-the-nvidia-cuda-toolkit).

   🚀 Tavsiya etiladigan versiya: `CUDA 11.7`

2. `cuDNN` [`cuDNN archive`](https://developer.nvidia.com/rdp/cudnn-archive)

   🚀 Tavsiya etiladigan versiya: `cuDNN v8.8.0`
   
3. `Tensorflow` [`Tensorflow official website`](https://www.tensorflow.org/?hl=ru).

   🚀 Tavsiya etiladigan versiya: `tensorflow`==2.11.1

4. python requirements:

   ``` shell
   pip install -r requirements.txt
   ```


5. YOLOv8 yordamida o'qitilgan modelni `best.pt`dan  `best.onnx` ko'rinishiga o'tkazish.
``` python
from ultralytics import YOLO

model = YOLO("best.pt")
model.export(format="onnx")
```

Deploy uchun script:

``` shell
python deploy.py --video_path video/video.mp4
```

Foydalanilgan manbalar: 
* [Tensorflow](https://www.tensorflow.org/?hl=ru)
* [YOLOv8](https://github.com/ultralytics/ultralytics)
