import argparse
from ultralytics import YOLO
import numpy as np
import torch
import cv2

model = YOLO("best.pt")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main(args: argparse.Namespace) -> None:
    video_path = args.video_path

    video = cv2.VideoCapture(video_path)

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    while True:

        ret, frame = video.read()

        frame = cv2.resize(frame,(width,height),interpolation = cv2.INTER_AREA)

        results = model.predict(frame, conf=0.7, stream = True,device = device)

        for result in results:
            boxes = result.boxes.cpu().numpy()  
            for box in boxes:
                r = box.xyxy[0].astype(int)
                print(r)
                cv2.rectangle(frame, r[:2], r[2:], (255, 255, 255), 2)
                
        cv2.imshow("input", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows() 
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help='video file')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
