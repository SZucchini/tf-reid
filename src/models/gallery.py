# import os
import glob
import argparse
from logging import getLogger, StreamHandler, DEBUG, Formatter

import cv2
import torch
import numpy as np
from PIL import Image
from natsort import natsorted
from torchvision import transforms

from ultralytics import YOLO
from features.build_features import get_hist


logger = getLogger("Log")
logger.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
handler_format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(handler_format)
logger.addHandler(handler)


def queue(queue, val):
    queue = np.roll(queue, -1)
    queue[-1] = val
    return queue


def hist_sim(candidates_hist, query_hist):
    if candidates_hist.ndim == 1:
        candidates_hist = np.array([candidates_hist])
    top = np.dot(query_hist, candidates_hist.T)
    bot = np.linalg.norm(query_hist) * np.linalg.norm(candidates_hist, axis=1)
    cos_sim = top / bot
    return cos_sim


class Gallery:
    def __init__(self):
        self.img_hist = None
        self.shoe_hist = None
        self.xpos = None
        self.frame = None
        self.files = []
        self.candidates = None

    def resister(self, query):
        if self.img_hist is None:
            self.img_hist = query['img_hist']
            self.shoe_hist = query['shoe_hist']
            self.xpos = np.array([0, 0, 0, 0, query['xpos']])
            self.frame = np.array([0, 0, 0, 0, query['frame']])
        else:
            self.img_hist = np.vstack([self.img_hist, query['img_hist']])
            self.xpos = np.vstack([self.xpos, np.array([0, 0, 0, 0, query['xpos']])])
            self.frame = np.vstack([self.frame, np.array([0, 0, 0, 0, query['frame']])])
        self.files.append([query['file']])

    def update(self, idx, query):
        if np.all(self.shoe_hist[idx] < 0):
            self.shoe_hist[idx] = query['shoe_hist']
        self.xpos[idx] = queue(self.xpos[idx], query['xpos'])
        self.frame[idx] = queue(self.frame[idx], query['frame'])
        self.files[idx].append(query['file'])

    def get_candidates_by_frame(self, query):
        diff = query['frame'] - self.frame[:, 4]

        idx = np.where((diff > 0) & (diff < 5))[0]
        if len(idx) > 0:
            return idx, "continue"
        idx = np.where((diff > 3600) & (diff < 5400))[0]
        if len(idx) > 0:
            return idx, "new"

        return None, "None"

    def get_nearest_idx(self, query):
        diff = abs(query['xpos'] - self.xpos[self.candidates, 4])
        while True:
            # only for side video
            if diff[np.argmin(diff)] > 1500:
                idx = None
                break
            idx = self.candidates[np.argmin(diff)]
            if self.frame[idx, 4] == query['frame']:
                diff[np.argmin(diff)] = 10000
            else:
                break
            if np.all(diff == 10000):
                idx = None
        return idx

    def get_similar_idx(self, query):
        idx = None
        self.candidates = self.candidates[np.argsort(self.frame[self.candidates, 4])]
        img_hist_sim = hist_sim(self.img_hist[self.candidates], query['img_hist'])
        shoe_hist_sim = hist_sim(self.shoe_hist[self.candidates], query['shoe_hist'])
        for i in range(len(self.candidates)):
            if shoe_hist_sim[i] < 0:
                if img_hist_sim[i] > 0.6:
                    idx = self.candidates[i]
                    break
            else:
                sim = (img_hist_sim[i] + shoe_hist_sim[i]) / 2
                if sim > 0.6:
                    idx = self.candidates[i]
                    break
        return idx

    def build(self, query):
        logger.debug('query: {}'.format(query))
        if self.img_hist is None:
            logger.debug('Init Gallery')
            self.resister(query)
            return 0

        self.candidates, situation = self.get_candidates_by_frame(query)
        logger.debug('candidates: {0}, situation: {1}'.format(self.candidates, situation))
        if self.candidates is None:
            logger.debug('Candidates are None. Register new query.')
            self.resister(query)
            return 0

        if situation == "continue":
            update_idx = self.get_nearest_idx(query)
        elif situation == "new":
            update_idx = self.get_similar_idx(query)

        logger.debug('update_idx: {}'.format(update_idx))
        if update_idx is not None:
            logger.debug('Update Gallery')
            self.update(update_idx, query)
        else:
            logger.debug('Update idx is None. Register new query.')
            self.resister(query)
        return 0


def get_runners(files):
    imgs = [img_transforms(Image.open(file).convert("RGB")) for file in files]
    imgs = torch.stack(imgs).to(device)
    with torch.no_grad():
        output = vit(imgs).argmax(dim=1)
    idx = np.where(output == 1)[0]
    del output
    torch.cuda.empty_cache()
    files = np.array(files)
    return files[idx]


def get_shoe_bbox(file):
    results = yolo.predict(source=file, save=False)
    for res in results:
        conf = res.boxes.conf
        if conf.shape[0] == 0:
            box = None
        else:
            idx = torch.argmax(conf)
            box = res.boxes.xyxy[idx].to("cpu").numpy().astype(int)
    return box


def get_query(file):
    query = {}
    img = cv2.imread(file)
    img_hist = get_hist(img, bins=9, div=2)

    shoe_bbox = get_shoe_bbox(file)
    if shoe_bbox is not None:
        shoe_img = img[shoe_bbox[1]:shoe_bbox[3], shoe_bbox[0]:shoe_bbox[2]]
        shoe_hist = get_hist(shoe_img, bins=9, div=2)
    else:
        shoe_hist = np.full_like(img_hist, -255)

    query = {
        'img_hist': img_hist,
        'shoe_hist': shoe_hist,
        'xpos': int(file.split('/')[-1].split('_')[2]),
        'frame': int(file.split('/')[-1].split('_')[0]),
        'file': file
    }
    return query


def build_gallery(files):
    runners = get_runners(files)
    logger.debug('runners: {}'.format(runners))
    if len(runners) == 0:
        return 0
    elif len(runners) == 1:
        query = get_query(runners[0])
        logger.debug('Build Gallery')
        gallery.build(query)
    else:
        x = np.array([int(runner.split('/')[-1].split('_')[2]) for runner in runners])
        runners = runners[np.argsort(x)[::-1]]
        for runner in runners:
            query = get_query(runner)
            logger.debug('Build Gallery')
            gallery.build(query)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="input images dir path")
    # parser.add_argument("--k1", type=int, default=10, help="")
    # parser.add_argument("--k2", type=int, default=10, help="")
    # parser.add_argument("--vsim_thr", type=float, default=0.7, help="")
    # parser.add_argument("--csim_thr", type=float, default=0.2, help="")
    args = parser.parse_args()

    input = args.input_dir + '/*.jpg'
    logger.debug('input images dir: {}'.format(input))
    files = natsorted(glob.glob(input))

    cluster = []
    for i in range(len(files)):
        filename = files[i].split('/')[-1]
        frame = int(filename.split('_')[0])
        if len(cluster) > 0:
            if cluster[-1] == frame:
                cluster.append(frame)
            else:
                build_gallery(files)
                cluster = [frame]
        else:
            cluster.append(frame)

        if i == len(files) - 1:
            build_gallery(files)

    logger.debug('Gallery files')
    for i, file in enumerate(gallery.files):
        logger.debug('{}: {}'.format(i, file))
    logger.debug('Gallery frame')
    for i, frame in enumerate(gallery.frames):
        logger.debug('{}: {}'.format(i, frame))


if __name__ == '__main__':
    gallery = Gallery()
    yolo = YOLO("../../yolo/weights/best400.pt")

    vit = torch.load('../../models/vit_runner_cpu.pth')
    img_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit.eval()
    vit.to(device)

    main()
