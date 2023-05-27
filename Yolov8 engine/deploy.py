import argparse
from RT.models.torch_utils import det_postprocess
from RT.models.cudart_api import TRTEngine
from RT.models.utils import blob
from ultralytics import YOLO
import numpy as np
import torch
import cv2

def ImageBox(image, new_shape=(640, 640), color=(0, 0, 0)):
    
    width, height, channel = image.shape
    
    ratio = min(new_shape[0] / width, new_shape[1] / height)
    
    new_unpad = int(round(height * ratio)), int(round(width * ratio))
    
    dw, dh = (new_shape[0] - new_unpad[0])/2, (new_shape[1] - new_unpad[1])/2

    if (height, width) != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return image, ratio, (dw, dh)


def main(args: argparse.Namespace) -> None:
    video_path = args.videos

    enggine = TRTEngine('best.engine')
    H, W = enggine.inp_info[0].shape[-2:]
    print(video_path)
    video = cv2.VideoCapture(video_path)

    while True:

        ret, frame = video.read()

        image, ratio, dwdh = ImageBox(frame)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        tensor = blob(image, return_seg=False)
        tensor = torch.asarray(tensor)
        
        results = enggine(tensor)
        
        dwdh = np.array(dwdh * 2, dtype=np.float32)
        
        bboxes, scores, labels = det_postprocess(results)
        bboxes = (bboxes-dwdh)/ratio

        
        for (bbox, score, label) in zip(bboxes, scores, labels):
            bbox = bbox.round().astype(np.int32).tolist()
            
            cv2.rectangle(frame, (bbox[0],bbox[1]) , (bbox[2],bbox[3]) , (255,255,255), 1)
            
            cv2.putText(frame,
                        f'{score:.3f}', (bbox[0], bbox[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, [225, 255, 255],
                        thickness=2)
            
        cv2.imshow("input", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos', type=str, help='video file')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)