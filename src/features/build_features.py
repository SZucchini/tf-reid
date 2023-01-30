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
