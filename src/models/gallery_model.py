import glob
import argparse

import cv2
import torch
import numpy as np

from ultralytics import YOLO
from features.build_features import img2vec, get_hist


def get_shoe_bbox(file):
    results = model.predict(source=file, save=False)
    for res in results:
        conf = res.boxes.conf
        if conf.shape[0] == 0:
            box = None
        else:
            idx = torch.argmax(conf)
            box = res.boxes.xyxy[idx].to("cpu").numpy().astype(int)
    return box


def hist_sim(query, features):
    if features.ndim == 1:
        features = np.array([features])
    top = np.dot(query, features.T)
    bot = np.linalg.norm(query) * np.linalg.norm(features, axis=1)
    cos_sim = top / bot
    return cos_sim


class Gallery:
    def __init__(self):
        self.vec_identifys = None
        self.color_identifys = None
        self.shoe_identifys = None
        self.frames = None
        self.files = None

    def get_vsim_topk(self, query, k=10):
        vsim = torch.cosine_similarity(query, self.vec_identifys)
        if vsim.shape[0] < k:
            k = vsim.shape[0]
        topk_idx = torch.argsort(vsim.cpu(), descending=True)[:k]
        return vsim, topk_idx

    def get_hsim_topk(self, feature, query, vsim_topk_idx, k=5):
        hsim = hist_sim(query, feature[vsim_topk_idx])
        if hsim.shape[0] < k:
            k = hsim.shape[0]
        topk_idx = np.argsort(hsim)[:k]
        return hsim, topk_idx

    def make_gallery(self, file_name, img_query, c_query, s_query):
        if self.vec_identifys is None:
            print('Initialize gallery...')
            self.vec_identifys = torch.stack([img_query], dim=0)
            self.color_identifys = np.array([c_query])
            if s_query is not None:
                self.shoe_identifys = np.array([s_query])
            else:
                self.shoe_identifys = np.array(np.ones(30*3*2))
            self.files = [[file_name]]
            self.frames = [[file_name.split('_')[-1].split('.')[0]]]
            print('vec identifys:', self.vec_identifys)
            print('color identifys:', self.color_identifys)
            print('shoe identifys:', self.shoe_identifys)
            print('files:', self.files)
            print('frames:', self.frames)
            return

        print()
        print('Culc image vector similarity...')
        vsim, vtopk_idx = self.get_vsim_topk(img_query)
        print('vsim_topk:', vsim[vtopk_idx])
        print('vsim_topk_idx:', vtopk_idx)
        print('vsim_top1:', vsim[vtopk_idx[0]])

        if vsim[vtopk_idx[0]] < 0.7:
            print()
            print('Image vector similarity is low.')
            print('new register by image vector')
            self.vec_identifys = torch.cat([self.vec_identifys, img_query], dim=0)
            self.color_identifys = np.concatenate([self.color_identifys, [c_query]], axis=0)
            if s_query is not None:
                self.shoe_identifys = np.concatenate([self.shoe_identifys, [s_query]], axis=0)
            else:
                self.shoe_identifys = np.concatenate([self.shoe_identifys, [np.ones(30*3*2)]],
                                                     axis=0)
            self.files.append([file_name])
            self.frames.append([file_name.split('_')[-1].split('.')[0]])
            return

        print()
        print('Culc color similarity...')
        img_hsim, img_htopk_idx = self.get_hsim_topk(self.color_identifys, c_query, vtopk_idx)
        print('img_hsim shape:', img_hsim.shape)
        print('img_hsim_topk_idx:', img_htopk_idx)
        print('img_hsim_top1:', img_hsim[img_htopk_idx[0]])

        if s_query is not None:
            print()
            print('Culc shoe similarity...')
            shoe_hsim, shoe_htopk_idx = self.get_hsim_topk(self.shoe_identifys, s_query, vtopk_idx)
            print('shoe_hsim shape:', shoe_hsim.shape)
            print('shoe_hsim_topk_idx:', shoe_htopk_idx)
            print('shoe_hsim_top1:', shoe_hsim[shoe_htopk_idx[0]])

            if img_hsim[img_htopk_idx[0]] < 0.55 and shoe_hsim[shoe_htopk_idx[0]] < 0.6:
                print('Color and shoe similarity is low.')
                print('new register by color and shoe')
                print(self.vec_identifys.shape, img_query.shape)
                self.vec_identifys = torch.cat([self.vec_identifys,
                                                torch.stack([img_query], dim=0)], dim=0)
                self.color_identifys = np.concatenate([self.color_identifys, [c_query]], axis=0)
                self.shoe_identifys = np.concatenate([self.shoe_identifys, [s_query]], axis=0)
                self.files.append([file_name])
                self.frames.append([file_name.split('_')[-1].split('.')[0]])
                return

            else:
                print('Color and shoe similarity is high.')
                if shoe_hsim[shoe_htopk_idx[0]] > 0.6:
                    print('Shoe similarity is high.')
                    self.files[vtopk_idx[shoe_htopk_idx[0]]].append(file_name)
                    self.frames[vtopk_idx[shoe_htopk_idx[0]]].append(file_name.split('_')[-1].split('.')[0])
                    if np.all(self.shoe_identifys[vtopk_idx[shoe_htopk_idx[0]]] == 1):
                        self.shoe_identifys[vtopk_idx[shoe_htopk_idx[0]]] = s_query
                    return
                else:
                    print('Shoe similarity is low. Using color similarity.')
                    self.files[vtopk_idx[img_htopk_idx[0]]].append(file_name)
                    self.frames[vtopk_idx[img_htopk_idx[0]]].append(file_name.split('_')[-1].split('.')[0])
                    if np.all(self.shoe_identifys[vtopk_idx[img_htopk_idx[0]]] == 1):
                        self.shoe_identifys[vtopk_idx[img_htopk_idx[0]]] = s_query

        else:
            print()
            print('shoe query is None')
            if img_hsim[img_htopk_idx[0]] < 0.55:
                print('Color hist similarity is low.')
                print('new register by color')
                print(self.vec_identifys.shape, img_query.shape)
                self.vec_identifys = torch.cat([self.vec_identifys,
                                                torch.stack([img_query], dim=0)], dim=0)
                self.color_identifys = np.concatenate([self.color_identifys, [c_query]], axis=0)
                self.shoe_identifys = np.concatenate([self.shoe_identifys, [np.ones(30*3*2)]],
                                                     axis=0)
                self.files.append([file_name])
                self.frames.append([file_name.split('_')[-1].split('.')[0]])
                return
            else:
                print('Color hist similarity is high.')
                print('add file and frame to DB')
                self.files[vtopk_idx[img_htopk_idx[0]]].append(file_name)
                self.frames[vtopk_idx[img_htopk_idx[0]]].append(file_name.split('_')[-1].split('.')[0])
                return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="input images dir path")
    parser.add_argument("--distance", type=str, default="lab", help="")
    parser.add_argument("--k1", type=int, default=10, help="")
    parser.add_argument("--k2", type=int, default=10, help="")
    parser.add_argument("--vsim_thr", type=float, default=0.7, help="")
    parser.add_argument("--csim_thr", type=float, default=0.2, help="")
    args = parser.parse_args()

    input = args.input_dir + '/*.jpg'
    print('input images:', input)
    files = glob.glob(input)
    files.sort()

    gallery = Gallery()
    img_features = img2vec(files)
    print()

    for i, (file, img_feature) in enumerate(zip(files, img_features)):
        print('query file:', file)
        img = cv2.imread(file)
        shoe_bbox = get_shoe_bbox(file)
        if shoe_bbox is not None:
            print('shoe bbox was detected!', shoe_bbox)
        else:
            print('shoe bbox was not detected!')

        img_hist = get_hist(img, bins=64, div=6)
        if shoe_bbox is not None:
            shoe_img = img[shoe_bbox[1]:shoe_bbox[3], shoe_bbox[0]:shoe_bbox[2]]
            shoe_hist = get_hist(shoe_img, bins=32, div=2)
        else:
            shoe_hist = None
        gallery.make_gallery(file, img_feature, img_hist, shoe_hist)
        print('--------------------------------------------------------------------')

    print()
    print('files')
    for i, file in enumerate(gallery.files):
        print(i, file)
    print('frames')
    for i, frame in enumerate(gallery.frames):
        print(i, frame)
    print('vec_identifys:', gallery.vec_identifys.shape)
    print('color_identify:', gallery.color_identifys.shape)


if __name__ == '__main__':
    model = YOLO("../../yolo/weights/best400.pt")
    main()
