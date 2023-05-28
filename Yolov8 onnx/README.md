# YOLOv8-onnx

`YOLOv8` yordamida train qilingan modelni onnx formatga o'tkazish va undan foydalanish.


[![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fatrox%2Fsync-dotenv%2Fbadge&style=flat)](https://github.com/MisterFoziljon/Face-Detection)
[![Python Version](https://img.shields.io/badge/Python-3.8--3.11-FFD43B?logo=python)](https://github.com/MisterFoziljon/Face-Detection)


# ONNX haqida:

* [ONNX official website](https://onnx.ai/)
* [ONNX PYPI](https://pypi.org/project/onnx)
* [ONNX in PyTorch](https://pytorch.org/docs/stable/onnx.html)
* [ONNX in GitHub](https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md)

python requirements:

   ``` shell
   pip install -r requirements.txt
   ```

YOLOv8 yordamida o'qitilgan modelni `best.pt`dan  `best.onnx` ko'rinishiga o'tkazish:
``` python
from ultralytics import YOLO

model = YOLO("best.pt")
model.export(format="onnx")
```

ONNX modelni yuklash:
``` python
import onnx

onnx_model = onnx.load("path/to/the/model.onnx")
```

Deploy uchun script:

``` shell
python deploy.py --video_path video/video.mp4
```
