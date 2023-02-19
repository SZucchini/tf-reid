import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor

import clip
from colorthief import ColorThief


def img2vec(files):
    model, preprocess = clip.load('ViT-B/32', jit=True)
    model = model.eval()
    preprocess = Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        ToTensor()
    ])

    imgs = []
    for i, file in enumerate(tqdm(files)):
        img = preprocess(Image.open(file).convert("RGB"))
        imgs.append(img)
    img_input = torch.tensor(np.stack(imgs)).cuda()

    with torch.no_grad():
        img_features = model.encode_image(img_input).float()

    print('images shape:', img_input.shape)
    print('features shape:', img_features.shape)
    # img_features = img_features.to('cpu').detach().numpy().copy()
    return img_features


def get_all_palette(files, c_num, p_num, sort=False, black='del'):
    c_features = []
    for f in files:
        color_thief = ColorThief(f)
        palette = color_thief.get_multi_palette(color_count=c_num,
                                                quality=1,
                                                palette_num=p_num,
                                                sort=sort,
                                                black=black)
        c_features.append(palette)
    return np.array(c_features)


def get_hist(cv_img, bins=9, div=2):
    bgr_hist = []
    window = cv_img.shape[0] // div
    for i in range(div):
        if i == div-1:
            data = cv_img[i*window:, :, :]
        else:
            data = cv_img[i*window:(i+1)*window, :, :]
        for j in range(3):
            hist = cv2.calcHist([data], [j], None, [bins], [0, 256])[1:]
            # 改善の余地あり？
            hist = cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
            bgr_hist.append(hist.reshape(bins-1,))

    bgr_hist = np.array(bgr_hist)
    bgr_hist = bgr_hist.reshape(-1)
    return bgr_hist


def get_hist_no_div(cv_img, bins=9):
    bgr_hist = []
    cv_img = cv2.resize(cv_img, (100, 200))
    for i in range(3):
        hist = cv2.calcHist([cv_img], [i], None, [bins], [0, 256])[1:]
        hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        bgr_hist.append(hist.reshape(bins-1,))

    bgr_hist = np.array(bgr_hist)
    return bgr_hist.reshape(-1)


def get_h_hist(cv_img, bins=49):
    cv_img = cv2.resize(cv_img, (100, 200))
    hsv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv_img], [0], None, [bins], [0, 180])[1:]
    h_hist = cv2.normalize(h_hist, h_hist, 0, 1, cv2.NORM_MINMAX)
    return h_hist.reshape(-1)
