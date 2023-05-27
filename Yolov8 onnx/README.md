```best.pt``` modelni ```best.onnx``` formatdagi modelga o'tkazamiz:

```python
from ultralytics import YOLO

model = YOLO("best.pt")
model.export(format="onnx")
```

Ushbu modelni yuklash va ishlatish uchun ```onnx.ipnb``` ga kiring.
