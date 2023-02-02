import os
import glob
import argparse

import numpy as np
from PIL import Image
from skimage import color
import seaborn as sns
import matplotlib.pyplot as plt

from colorthief import ColorThief


def print_args(args):
    for k, v in vars(args).items():
        print(k, v)
    print()


def get_param_lst(param_range):
    start, end = int(param_range.split(':')[0]), int(param_range.split(':')[1])
    lst = list(range(start, end+1))
    return lst


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


def plot_palette(files, c_num, p_num, c_features, output_dir):
    fig, ax = plt.subplots(len(files), p_num*c_num+1, figsize=(400, 5*c_num*p_num+10))
    fig_name = output_dir + f'/c{c_num}_p{p_num}.png'
    print('save fig name:', fig_name)

    for p, (f, feature) in enumerate(zip(files, c_features)):
        img = Image.open(f)
        ax[p, 0].imshow(img)
        ax[p, 0].axis('off')

        for i in range(p_num):
            for j in range(c_num):
                im = Image.new('RGB', (5, 5), tuple(feature[i, j, :]))
                ax[p, i*c_num+j+1].imshow(im)
                ax[p, i*c_num+j+1].axis('off')

    fig.tight_layout()
    plt.savefig(fig_name)
    plt.close()


def ciede_sim(c_feature1, c_feature2):
    c1 = color.rgb2lab(np.array(c_feature1)/255)
    c2 = color.rgb2lab(np.array(c_feature2)/255)
    diff = color.deltaE_ciede2000(c1, c2)
    return diff.sum()


def rgb_sim(c_feature1, c_feature2):
    c1 = np.array(c_feature1)
    c2 = np.array(c_feature2)
    diff = np.linalg.norm(c2-c1, axis=1)
    return diff.sum()


def culc_sim(c_features, distance="rgb"):
    print('color distance method:', distance)
    num_img = c_features.shape[0]
    sim = np.zeros((num_img, num_img))
    for i in range(num_img):
        for j in range(i, num_img):
            if distance == "rgb":
                sim[i, j] = rgb_sim(c_features[i], c_features[j])
            else:
                sim[i, j] = ciede_sim(c_features[i], c_features[j])
            if i != j:
                sim[j, i] = sim[i, j]

    sim = sim / sim.max()
    sim = 1 - sim
    return sim


def culc_ave_sim(sim, imgs_per_person):
    print('similarity matrix')
    print(sim)
    persons = sim.shape[0] // imgs_per_person
    ave_sim = np.zeros((persons, persons))
    sim_nan = np.where(sim == 1, np.nan, sim)
    print('similarity matrix 1 to nan')
    print(sim_nan)

    for i in range(persons):
        for j in range(persons):
            ave_sim[i, j] = np.nanmean(sim_nan[i*10:(i+1)*10-1, j*10:(j+1)*10-1])
    print('average similarity matrix')
    print(ave_sim)
    return ave_sim


def culc_sim_acc(sim):
    ans = np.array([0, 1, 2, 3, 4, 5, 6])
    max_idx = np.argmax(sim, axis=1)
    acc = np.count_nonzero(ans == max_idx) / max_idx.shape[0]
    print('Accuracy:', acc)
    return acc


def save_sim_matrix(sim, c_num, p_num, output_dir):
    acc = culc_sim_acc(sim)
    fig_name = output_dir + f'/c{c_num}_p{p_num}_acc{acc:.2f}.png'
    print('save figure name:', fig_name)
    plt.figure()
    sns.heatmap(sim, square=True, cbar=True, annot=True, cmap='Blues',
                xticklabels=1, yticklabels=1, vmin=0, vmax=1)
    plt.savefig(fig_name)
    plt.close()


def main():
    # argments settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="input images dir path")
    parser.add_argument("--color_param_range", type=str, default="1:10",
                        help="evaluate color num per palette")
    parser.add_argument("--palette_param_range", type=str, default="1:10",
                        help="evaluate multi palette num")
    parser.add_argument("--palette_sort", type=int, default=0,
                        help="sort palette or not")
    parser.add_argument("--black", type=str, default="del",
                        help="delete or turn to white black pixels")
    parser.add_argument("--distance", type=str, default="rgb",
                        help="color distance culc method (rgb or lab)")
    parser.add_argument("--imgs_per_person", type=int, default=10,
                        help="number of images per same person")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="output dir path")
    args = parser.parse_args()
    print_args(args)

    if args.palette_sort:
        palette_dir = args.output_dir + '/palette_sorted'
        matrix_dir = args.output_dir + f'/matrix_{args.distance}_sorted'
    else:
        palette_dir = args.output_dir + '/palette'
        matrix_dir = args.output_dir + f'/matrix_{args.distance}'
    print('palette directory:', palette_dir)
    print('matrix directory:', matrix_dir)
    # check output dir exist
    if not os.path.exists(palette_dir):
        os.makedirs(palette_dir)
    if not os.path.exists(matrix_dir):
        os.makedirs(matrix_dir)

    input = args.input_dir + '/*.jpg'
    print('input images:', input)
    files = glob.glob(input)
    files.sort()
    color_params = get_param_lst(args.color_param_range)
    palette_params = get_param_lst(args.palette_param_range)
    print('color parameters:', color_params)
    print('palette parameters:', palette_params)

    for c_num in color_params:
        for p_num in palette_params:
            print()
            print(f'Number of (colors, palettes) = ({c_num}, {p_num})')
            c_features = get_all_palette(files, c_num, p_num, args.palette_sort, args.black)
            print('c_features shape:', c_features.shape)
            # plot_palette(files, c_num, p_num, c_features, palette_dir)
            sim = culc_sim(c_features, args.distance)
            ave_sim = culc_ave_sim(sim, args.imgs_per_person)
            save_sim_matrix(ave_sim, c_num, p_num, matrix_dir)


if __name__ == '__main__':
    main()
