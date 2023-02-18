import os
import argparse
from logging import getLogger, StreamHandler, DEBUG, Formatter

import cv2
import torch
from PIL import Image

from ultralytics import YOLO


logger = getLogger("Log")
logger.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
handler_format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(handler_format)
logger.addHandler(handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="input video path")
    parser.add_argument("--output", type=str, required=True, help="output dir path")
    args = parser.parse_args()

    logger.debug('Input video: {}'.format(args.video))
    logger.debug('Output dir: {}'.format(args.output))
    if os.path.exists(args.output) is False:
        os.makedirs(args.output)

    model = YOLO("../../yolo/weights/yolov8x-seg.pt")

    frame = 0
    video = cv2.VideoCapture(args.video)
    while True:
        ret, img = video.read()
        if ret is False:
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(source=img, max_det=20, conf=0.3,
                        classes=0, retina_masks=True, save=False)

        res = results[0]
        for i in range(len(res)):
            bbox = res.boxes.xyxy[i].to("cpu").numpy().astype(int)
            mask = res.masks.data[i].to("cpu").numpy()
            masked = (img * mask[:, :, None])[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            img_masked = Image.fromarray(masked.astype('uint8'))
            name = f'{frame}_{i}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.jpg'
            save_path = os.path.join(args.output, name)
            img_masked.save(save_path)

        frame += 1

    return 0


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
