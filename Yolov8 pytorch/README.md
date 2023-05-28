# YOLOv8-TensorRT

`YOLOv8` yordamida train qilingan modeldan foydalanish!


[![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fatrox%2Fsync-dotenv%2Fbadge&style=flat)](https://github.com/MisterFoziljon/Face-Detection)
[![Python Version](https://img.shields.io/badge/Python-3.8--3.10-FFD43B?logo=python)](https://github.com/MisterFoziljon/Face-Detection)


# Dastur ishlashi uchun kerakli muhitni yaratib olamiz (environments):

1. `CUDA` [`CUDA official website`](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#download-the-nvidia-cuda-toolkit).

   🚀 Tavsiya etiladigan versiya: `CUDA` >= 11.4

2. `TensorRT` [`TensorRT official website`](https://developer.nvidia.com/nvidia-tensorrt-8x-download).

   🚀 Tavsiya etiladigan versiya: `TensorRT` >= 8.4

3. python requirements:

   ``` shell
   pip install -r requirements.txt
   ```

4. ONNX export yoki TensorRT API dan foydalanish uchun  [`ultralytics`](https://github.com/ultralytics/ultralytics).

   ``` shell
   pip install ultralytics
   ```

5. YOLOv8 yordamida o'qitilgan modelni `best.pt` ko'rinishida kerakli faylga saqlash.


Deploy uchun script:

``` shell
python deploy.py --video_path video/video.mp4
```

Foydalanilgan manbalar: 
* [TensorRT](https://github.com/triple-Mu/YOLOv8-TensorRT)
* [YOLOv8](https://github.com/ultralytics/ultralytics)
