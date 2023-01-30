import os
import sys
sys.path.append("..")
import glob
import argparse

import torch
import numpy as np
from PIL import Image
from skimage import color

from features.build_features import img2vec, get_all_palette


def ciede_sim(query, features):
    q = color.rgb2lab(np.array([query/255]))
    db = color.rgb2lab(features/255)
    # print('q shape:', q.shape)
    # print('db shape:', db.shape)
    diff = color.deltaE_ciede2000(q, db)
    csim = np.sum(diff, axis=(1, 2))
    return csim


def rgb_sim(query, features):
    q = np.array([query])
    diff = np.linalg.norm(features-q, axis=2)
    csim = np.sum(diff, axis=1)
    return csim


class Gallery:
    def __init__(self):
        self.vec_identifys = None
        self.color_identifys = None
        self.frames = None
        self.files = None

    def get_vsim_topk(self, query, k=10):
        vsim = torch.cosine_similarity(query, self.vec_identifys)
        if vsim.shape[0] < k:
            k = vsim.shape[0]
        topk_idx = torch.argsort(vsim.cpu(), descending=True)[:k]
        # return vsim[topk_idx], topk_idx
        return vsim, topk_idx

    def get_csim_topk(self, query, vsim_topk_idx, k=5, distance="lab"):
        if vsim_topk_idx.shape[0] == 1:
            features = self.color_identifys[vsim_topk_idx].reshape(1, 4, 5, 3)
        else:
            features = self.color_identifys[vsim_topk_idx]
        if distance == "lab":
            csim = ciede_sim(query, features)
        else:
            csim = rgb_sim(query, features)

        # csim = csim / csim.max()
        # csim = 1 - csim
        if csim.shape[0] < k:
            k = csim.shape[0]
        topk_idx = np.argsort(csim)[:k]
        # return csim[topk_idx], topk_idx
        return csim, topk_idx

    def make_gallery(self, file_name, img_query, c_query, files):
        if self.vec_identifys is None:
            self.vec_identifys = torch.stack([img_query], dim=0)
            self.color_identifys = np.array([c_query])
            self.files = [[file_name]]
            self.frames = [[file_name.split('_')[-1].split('.')[0]]]
            return

        vsim, vtopk_idx = self.get_vsim_topk(img_query)
        print('vsim_topk:', vsim[vtopk_idx])
        print('vsim_topk_idx:', vtopk_idx)
        # print('color_identifys shape:', self.color_identifys.shape)

        if vsim[vtopk_idx[0]] < 0.7:
            print('new register by img vec')
            self.vec_identifys = torch.cat([self.vec_identifys, img_query], dim=0)
            self.color_identifys = np.concatenate([self.color_identifys, [c_query]], axis=0)
            self.files.append([file_name])
            self.frames.append([file_name.split('_')[-1].split('.')[0]])
            return

        csim, ctopk_idx = self.get_csim_topk(c_query, vtopk_idx)
        print('csim shape:', csim.shape)
        print('csim_topk_idx:', ctopk_idx)
        print('csim_top1:', csim[ctopk_idx[0]])
        if csim[ctopk_idx[0]] > 300:
            print('new register by color')
            print(self.vec_identifys.shape, img_query.shape)
            self.vec_identifys = torch.cat([self.vec_identifys,
                                            torch.stack([img_query], dim=0)], dim=0)
            self.color_identifys = np.concatenate([self.color_identifys, [c_query]], axis=0)
            self.files.append([file_name])
            self.frames.append([file_name.split('_')[-1].split('.')[0]])
            return

        # 既存のデータの対応するインデックスにfile_nameとframeを追加
        self.files[vtopk_idx[ctopk_idx[0]]].append(file_name)
        self.frames[vtopk_idx[ctopk_idx[0]]].append(file_name.split('_')[-1].split('.')[0])


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

    img_features = img2vec(files)
    c_features = get_all_palette(files, c_num=5, p_num=4, sort=True, black='del')

    gallery = Gallery()
    # files, img_features, c_featuresをループ処理
    for i, (file, img_feature, c_feature) in enumerate(zip(files, img_features, c_features)):
        gallery.make_gallery(file, img_feature, c_feature, files)
        """
        img_feature = img_feature.to('cpu').detach().numpy().copy()
        c_feature = c_feature.reshape(-1, 3)
        # 類似度計算
        vsim, topk_idx = gallery.get_vsim_topk(img_feature, k=args.k1)
        csim, topk_idx = gallery.get_csim_topk(c_feature, k=args.k2, distance=args.distance)
        # 類似度閾値を超えたものを抽出
        vsim_idx = np.where(vsim > args.vsim_thr)[0]
        csim_idx = np.where(csim < args.csim_thr)[0]
        # 閾値を超えたものがあれば、類似画像として登録
        if len(vsim_idx) > 0 and len(csim_idx) > 0:
            idx = np.intersect1d(vsim_idx, csim_idx)
            if len(idx) > 0:
                print('similar images:', file)
                for j in idx:
                    print('  ', gallery.files[j][0])
                gallery.files[j].append(file)
                gallery.frames[j].append(file.split('_')[0].split('.')[0])
            else:
                gallery.make_gallery(file, img_feature, c_feature)
        else:
            gallery.make_gallery(file, img_feature, c_feature)
        """

    print('files')
    for i, file in enumerate(gallery.files):
        print(i, file)
    print('frames')
    for i, frame in enumerate(gallery.frames):
        print(i, frame)
    print('vec_identifys:', gallery.vec_identifys.shape)
    print('color_identify:', gallery.color_identifys.shape)


if __name__ == '__main__':
    main()
