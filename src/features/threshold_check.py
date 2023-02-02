import os
import glob
import argparse

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor

import clip


def img2vec(input_dir):
    model, preprocess = clip.load('ViT-B/32', jit=True)
    model = model.eval()
    preprocess = Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        ToTensor()
    ])

    imgs = []
    input = input_dir + '/*.jpg'
    print('input path:', input)

    files = glob.glob(input)
    files.sort()
    for i, file in enumerate(tqdm(files)):
        img = preprocess(Image.open(file).convert("RGB"))
        imgs.append(img)
    img_input = torch.tensor(np.stack(imgs)).cuda()
    print('images shape:', img_input.shape)

    with torch.no_grad():
        img_features = model.encode_image(img_input).float()
    print('features shape:', img_features.shape)

    return files, img_features


def visualize_rank(files, img_features, output_dir):
    row, col = 5, 10
    for query in img_features:
        probs = torch.cosine_similarity(img_features, query)
        idx_sorted = np.argsort(-probs.cpu(), axis=0)
        fig, ax = plt.subplots(row, col, figsize=(20, 25))
        fig_name = output_dir + '/' + files[idx_sorted[0]].split('/')[-1].split('.')[0] + '.png'
        print('save fig name:', fig_name)

        loc = 0
        for idx in idx_sorted:
            img_path = files[idx]
            img = np.asarray(Image.open(img_path))
            cos_sim = round(probs[idx.item()].item(), 3)
            if cos_sim == 1:
                cos_sim = "Query image"

            ax[loc//col, loc-((loc//col)*col)].imshow(img)
            ax[loc//col, loc-((loc//col)*col)].tick_params(labelleft=False, labelbottom=False)
            ax[loc//col, loc-((loc//col)*col)].tick_params(top=False, bottom=False,
                                                           left=False, right=False)
            ax[loc//col, loc-((loc//col)*col)].set_xlabel(cos_sim, fontsize=22)
            loc += 1

        for i in range(loc, row*col):
            ax[i//col, i-((i//col)*col)].axis("off")

        fig.tight_layout()
        plt.savefig(fig_name)
        plt.close()


def main():
    # argments settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="input images dir path")
    parser.add_argument("--output_dir", type=str, required=True, help="output dir path")
    args = parser.parse_args()

    # check output dir exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    files, img_features = img2vec(args.input_dir)
    visualize_rank(files, img_features, args.output_dir)


if __name__ == '__main__':
    main()
