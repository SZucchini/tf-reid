import glob
import pickle
import argparse
from logging import getLogger, StreamHandler, DEBUG, Formatter

import cv2
import numpy as np
from PIL import Image
from natsort import natsorted
import torch
from torchvision import transforms
from scipy.optimize import linear_sum_assignment

from ultralytics import YOLO
from scene import Scene
from features.build_features import get_hist, get_hist_no_div, get_h_hist


logger = getLogger("Log")
logger.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
handler_format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(handler_format)
logger.addHandler(handler)


def get_runners(files):
    if len(files) == 0:
        return None
    imgs = [img_transforms(Image.open(file).convert("RGB")) for file in files]
    imgs = torch.stack(imgs).to(device)
    with torch.no_grad():
        output = vit(imgs).argmax(dim=1)

    np_output = output.to("cpu").numpy()
    idx = np.where(np_output == 1)[0]
    del imgs, output
    torch.cuda.empty_cache()

    if len(idx) == 0:
        return None

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
            score = conf[idx].to("cpu").numpy()
    del results, conf
    torch.cuda.empty_cache()
    return box, score


def get_query(file):
    img = cv2.imread(file)
    img_hist = get_h_hist(img, bins=49)

    shoe_bbox, score = get_shoe_bbox(file)
    if shoe_bbox is not None:
        shoe_img = img[shoe_bbox[1]:shoe_bbox[3], shoe_bbox[0]:shoe_bbox[2]]
        shoe_hist = get_h_hist(shoe_img, bins=49)
        shoe_score = score
    else:
        shoe_img = np.zeros_like(img)
        shoe_hist = np.zeros_like(img_hist)
        shoe_score = 0
    xpos = int(file.split('/')[-1].split('_')[4])

    q = {
        'img_hist': img_hist,
        'shoe_img': shoe_img,
        'shoe_hist': shoe_hist,
        'shoe_score': shoe_score,
        'xpos': xpos,
        'frame': int(file.split('/')[-1].split('_')[0]),
        'file': file
    }
    return q


def get_frame_query(files):
    query = []
    if len(files) == 1:
        q = get_query(files[0])
        query.append(q)
        return query

    for file in files:
        q = get_query(file)
        query.append(q)
    return query


def match(query, pred):
    """
    クエリの数 < 予測候補の数の場合、クエリの数に合わせられてしまうため、失敗の可能性がある
    """
    q_xpos = np.array([q['xpos'] for q in query])
    p_xpos = []
    pred_idx_dict = {}
    for i in range(len(pred)):
        if pred[i] != -1:
            p_xpos.append(pred[i])
            new_idx = len(p_xpos) - 1
            pred_idx_dict[new_idx] = i

    cost_matrix = abs(np.tile(q_xpos, (len(p_xpos), 1)).T - np.array(p_xpos))
    q_idx, p_idx = linear_sum_assignment(cost_matrix)

    p_idx = [pred_idx_dict[idx] for idx in p_idx]
    new_query = []
    for i in range(len(query)):
        if i not in q_idx:
            # queryの位置が左よりではない場合は除外すべき
            # この追加部分が悪さしているかも
            # if query[i]['xpos'] > 1000:
            #     pass
            # else:
            new_query.append(query[i])

    return q_idx, p_idx, new_query


def make_scene(files, scenes):
    runners = get_runners(files)
    if runners is None:
        logger.debug('No runners')
        return scenes
    else:
        logger.debug('Runners: {}'.format(len(runners)))
        query = get_frame_query(runners)
        if len(scenes) == 0:
            logger.debug('Init Scene')
            scenes.append(Scene())
            for q in query:
                scenes[-1].resister(q)
        else:
            if query[0]['frame'] - scenes[-1].last > 60:  # 閾値
                logger.debug('Add Scene')
                scenes.append(Scene())
                for q in query:
                    scenes[-1].resister(q)
            else:
                logger.debug('Update Scene')
                pred = scenes[-1].pred_xpos()
                q_idx, p_idx, new_query = match(query, pred)
                if len(q_idx) > 0:
                    logger.debug('Update query: {}'.format(len(q_idx)))
                    for q, p in zip(q_idx, p_idx):
                        scenes[-1].update(query[q], p)
                if len(new_query) > 0:
                    logger.debug('Add new query: {}'.format(len(new_query)))
                    for q in new_query:
                        scenes[-1].resister(q)
    return scenes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="input images dir path")
    args = parser.parse_args()

    input = args.input_dir + '/*.jpg'
    files = natsorted(glob.glob(input))
    logger.debug('Input images dir: {}'.format(input))
    logger.debug('Input images: {}'.format(len(files)))

    scenes = []
    last_frame = 0
    frame_files = []
    for i in range(len(files)):
        frame = int(files[i].split('/')[-1].split('_')[0])
        if frame == last_frame:
            frame_files.append(files[i])
            if i == len(files) - 1:
                scenes = make_scene(frame_files, scenes)
        else:
            scenes = make_scene(frame_files, scenes)
            last_frame = frame
            frame_files = [files[i]]

    with open('../../models/scenes_new_hhist.pickle', mode='wb') as f:
        pickle.dump(scenes, f)

    logger.debug('Scenes: {}'.format(len(scenes)))
    return 0


if __name__ == '__main__':
    vit = torch.load('../../models/vit_new_timm_epoch200.pth')
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

    yolo = YOLO("../../yolo/weights/best500.pt")
    main()
