# YOLOv8-pytorch

`YOLOv8` yordamida train qilingan modeldan foydalanish!


[![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fatrox%2Fsync-dotenv%2Fbadge&style=flat)](https://github.com/MisterFoziljon/Face-Detection)
[![Python Version](https://img.shields.io/badge/Python-3.7--3.11-FFD43B?logo=python)](https://github.com/MisterFoziljon/Face-Detection)


# Dastur ishlashi uchun kerakli muhitni yaratib olamiz (environments):

1. `CUDA` [`CUDA official website`](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#download-the-nvidia-cuda-toolkit).

   🚀 Tavsiya etiladigan versiya: `CUDA` == 11.7

2. `Pytorch` [`PyTorch official website`](https://pytorch.org).

   🚀 Tavsiya etiladigan versiya: 
   `torch`==2.0.0+cu117 
   `torchvision`==0.15.1+cu117 
   `torchaudio`==2.0.1 
   ([website dan yuklash](https://pytorch.org/get-started/previous-versions/))
   

3. python requirements:

   ``` shell
   pip install -r requirements.txt
   ```


4. YOLOv8 yordamida o'qitilgan modelni `best.pt` ko'rinishida kerakli faylga saqlash.


Deploy uchun script:

``` shell
python deploy.py --video_path video/video.mp4
```

Foydalanilgan manbalar: 
* [PyTorch](https://pytorch.org)
* [YOLOv8](https://github.com/ultralytics/ultralytics)
