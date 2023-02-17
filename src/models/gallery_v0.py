# import os
import glob
import pickle
import argparse
from logging import getLogger, StreamHandler, DEBUG, Formatter

import cv2
import torch
import numpy as np
from PIL import Image
from natsort import natsorted
from torchvision import transforms
from scipy.optimize import linear_sum_assignment

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
    corrs = []
    if candidates_hist.ndim == 1:
        candidates_hist = np.array([candidates_hist])
    for hist in candidates_hist:
        corr = cv2.compareHist(hist, query_hist, cv2.HISTCMP_CORREL)
        corrs.append(corr)
    return np.array(corrs)


def get_ave(ave_hist, num, hist):
    ave_hist = (ave_hist * num + hist) / (num + 1)
    num += 1
    return ave_hist, num


class Gallery:
    def __init__(self):
        self.img_hist = None
        self.num = None
        self.shoe_hist = None
        self.shoe_score = None
        self.xpos = None
        self.frame = None
        self.files = []
        self.cand_series = None
        self.cand_circle = None

    def resister(self, query_lst):
        for query in query_lst:
            if self.img_hist is None:
                self.img_hist = np.array([query['img_hist']])
                self.num = [1]
                self.shoe_hist = np.array([query['shoe_hist']])
                self.shoe_score = np.array([query['shoe_score']])
                self.xpos = np.array([[0, 0, 0, 0, query['xpos']]])
                self.frame = np.array([[0, 0, 0, 0, query['frame']]])
            else:
                self.img_hist = np.vstack([self.img_hist, query['img_hist']])
                self.num.append(1)
                self.shoe_hist = np.vstack([self.shoe_hist, query['shoe_hist']])
                self.shoe_score = np.vstack([self.shoe_score, query['shoe_score']])
                self.xpos = np.vstack([self.xpos, np.array([0, 0, 0, 0, query['xpos']])])
                self.frame = np.vstack([self.frame, np.array([0, 0, 0, 0, query['frame']])])
            self.files.append([query['file']])

    def update(self, idx, query):
        self.img_hist[idx], self.num[idx] = get_ave(self.img_hist[idx],
                                                    self.num[idx],
                                                    query['img_hist'])
        if self.shoe_score[idx] < query['shoe_score']:
            self.shoe_hist[idx] = query['shoe_hist']
            self.shoe_score[idx] = query['shoe_score']
        self.xpos[idx] = queue(self.xpos[idx], query['xpos'])
        self.frame[idx] = queue(self.frame[idx], query['frame'])
        self.files[idx].append(query['file'])

    def get_candidates_by_frame(self, frame_query):
        diff = frame_query[0]['frame'] - self.frame[:, 4]

        for i in range(5):  # 閾値
            idx_series = np.where(diff == i+1)[0]
            if len(idx_series) > 0:
                break
        if len(idx_series) == 0:
            idx_series = None
        idx_circle = np.where((diff > 3900) & (diff < 5100))[0]  # 閾値
        if len(idx_circle) == 0:
            idx_circle = None

        return idx_series, idx_circle

    def eval_entry(self, frame_query):
        idx = -1
        if self.cand_series is None:
            entry_query = frame_query
            return None, entry_query
        for i, query in enumerate(frame_query):
            for cand in self.cand_series:
                xdiff = query['xpos'] - self.xpos[cand, 4]
                if xdiff < -200:  # 閾値
                    idx = i
                    break
        if idx == -1:
            return frame_query, None
        else:
            entry_query = [frame_query.pop(idx)]
            return frame_query, entry_query

    def get_circle_idx(self, entry_query):
        idx = None
        # idx_ave = None
        # max_sim = 0.65  # 閾値
        # max_sim_ave = 0.4  # 閾値
        img_hist_sim = hist_sim(self.img_hist[self.cand_circle], entry_query[0]['img_hist'])
        shoe_hist_sim = hist_sim(self.shoe_hist[self.cand_circle], entry_query[0]['shoe_hist'])
        logger.debug('img_hist_sim: {}'.format(img_hist_sim))
        logger.debug('shoe_hist_sim: {}'.format(shoe_hist_sim))

        if np.all(img_hist_sim < 0.1):
            idx = None
            return

        high_img_hist = img_hist_sim[img_hist_sim > 0.7]
        if len(high_img_hist) > 0:
            idx = self.cand_circle[img_hist_sim == np.max(high_img_hist)]
            if len(idx) > 0:
                idx = idx[0]
            return idx
        if len(high_img_hist) == 0:
            idx = "pass"
            return idx

        # for i in range(len(self.cand_circle)):
        #     if shoe_hist_sim[i] == 1:
        #         sim = img_hist_sim[i]
        #         logger.debug('sim: {}'.format(sim))
        #         if sim > max_sim:
        #             idx = self.cand_circle[i]
        #             max_sim = sim
        #     else:
        #         sim = (img_hist_sim[i] + shoe_hist_sim[i]) / 2  # 閾値
        #         logger.debug('sim: {}'.format(sim))
        #         if sim > max_sim_ave:
        #             idx_ave = self.cand_circle[i]
        #             max_sim_ave = sim

        return idx

    def eval_out(self, frame_query):
        if np.max(self.xpos[self.cand_series, 4]) < 3400:  # 閾値
            return self.cand_series
        else:
            right = self.cand_series[np.argmax(self.xpos[self.cand_series, 4])]
            xdiff = np.array([query['xpos'] - self.xpos[right, 4] for query in frame_query])
            if np.all(xdiff < -200):  # 閾値
                return np.delete(self.cand_series, np.argmax(self.xpos[self.cand_series, 4]))
            else:
                return self.cand_series

    def get_series_idx(self, frame_query):
        sim_matrix = None
        for query in frame_query:
            sim = []
            # shoe_scoreで重みつき和にする？
            img_hist_sim = hist_sim(self.img_hist[self.cand_series], query['img_hist'])
            shoe_hist_sim = hist_sim(self.shoe_hist[self.cand_series], query['shoe_hist'])
            for i in range(len(self.cand_series)):
                if shoe_hist_sim[i] == 1:
                    sim.append(img_hist_sim[i])
                else:
                    sim.append((img_hist_sim[i] + shoe_hist_sim[i]) / 2)
            if sim_matrix is None:
                sim_matrix = np.array([sim])
            else:
                sim_matrix = np.vstack([sim_matrix, np.array(sim)])

        query_idx, cand_idx = linear_sum_assignment(sim_matrix, maximize=True)
        for q, c in zip(query_idx, cand_idx):
            if sim_matrix[q, c] < 0.3:  # 閾値
                query_idx = np.delete(query_idx, np.where(query_idx == q))
                cand_idx = np.delete(cand_idx, np.where(query_idx == c))
        return cand_idx, query_idx

    def build(self, frame_query):
        if self.img_hist is None:
            logger.debug('Init Gallery')
            self.resister(frame_query)
            return 0

        self.cand_series, self.cand_circle = self.get_candidates_by_frame(frame_query)
        if self.cand_series is None and self.cand_circle is None:
            logger.debug('No candidates. Register new frame query.')
            logger.debug('The number of frame query: {}'.format(len(frame_query)))
            self.resister(frame_query)
            return 0

        if self.cand_circle is not None:
            frame_query, entry_query = self.eval_entry(frame_query)
            if entry_query is not None:
                circle_idx = self.get_circle_idx(entry_query)
                if circle_idx == "pass":
                    logger.debug('Pass resist and update.')
                    pass
                elif circle_idx is not None:
                    self.update(circle_idx, entry_query[0])
                else:
                    logger.debug('No color candidates. Register new frame query.')
                    logger.debug('The number of entry query: {}'.format(len(entry_query)))
                    self.resister(entry_query)

        if frame_query is None:
            return 0

        if len(frame_query) != 0:
            self.cand_series = self.eval_out(frame_query)
            series_update, query_update = self.get_series_idx(frame_query)
            if len(series_update) == 0:
                return 0
            for idx, q in zip(series_update, query_update):
                self.update(self.cand_series[idx], frame_query[q])

        return 0


def get_runners(files):
    imgs = [img_transforms(Image.open(file).convert("RGB")) for file in files]
    imgs = torch.stack(imgs).to(device)
    with torch.no_grad():
        output = vit(imgs).argmax(dim=1)
    np_output = output.to("cpu").numpy()
    idx = np.where(np_output == 1)[0]
    del imgs
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
            score = None
        else:
            idx = torch.argmax(conf)
            box = res.boxes.xyxy[idx].to("cpu").numpy().astype(int)
            score = int(conf[idx].to("cpu").numpy())
    del results
    del conf
    torch.cuda.empty_cache()
    return box, score


def get_query(file):
    query = {}
    img = cv2.imread(file)
    img_hist = get_hist(img, bins=9, div=2)

    shoe_bbox, score = get_shoe_bbox(file)
    if shoe_bbox is not None:
        shoe_img = img[shoe_bbox[1]:shoe_bbox[3], shoe_bbox[0]:shoe_bbox[2]]
        shoe_hist = get_hist(shoe_img, bins=9, div=2)
        shoe_score = score
    else:
        shoe_hist = np.zeros_like(img_hist)
        shoe_score = 0

    xpos = (int(file.split('/')[-1].split('_')[2]) + int(file.split('/')[-1].split('_')[4])) / 2

    query = {
        'img_hist': img_hist,
        'shoe_hist': shoe_hist,
        'shoe_score': shoe_score,
        'xpos': xpos,
        'frame': int(file.split('/')[-1].split('_')[0]),
        'file': file
    }
    return query


def get_frame_query(files):
    frame_query = []
    if len(files) == 1:
        query = get_query(files[0])
        frame_query.append(query)
        return frame_query

    x = np.array(
        [(int(file.split('/')[-1].split('_')[2]) + int(file.split('/')[-1].split('_')[4])) / 2
            for file in files])
    files = files[np.argsort(x)[::-1]]
    for file in files:
        query = get_query(file)
        frame_query.append(query)
    return frame_query


def build_gallery(files):
    runners = get_runners(files)
    logger.debug('runners: {}'.format(len(runners)))
    if len(runners) == 0:
        return 0

    frame_query = get_frame_query(runners)
    logger.debug('Length of frame queries: {}'.format(len(frame_query)))
    gallery.build(frame_query)

    with open('../../models/gallery_v0.pickle', mode='wb') as f:
        pickle.dump(gallery, f)

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
    cluster_frames = []
    for i in range(len(files)):
        filename = files[i].split('/')[-1]
        frame = int(filename.split('_')[0])
        if len(cluster) > 0:
            if cluster_frames[-1] == frame:
                cluster.append(files[i])
                cluster_frames.append(frame)
            else:
                build_gallery(cluster)
                cluster = [files[i]]
                cluster_frames = [frame]
        else:
            cluster.append(files[i])
            cluster_frames.append(frame)

        if i == len(files) - 1:
            build_gallery(cluster)

    logger.debug('Gallery files')
    for i, file in enumerate(gallery.files):
        logger.debug('{}: {}'.format(i, file))
    logger.debug('Gallery frame')
    for i, frame in enumerate(gallery.frame):
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
