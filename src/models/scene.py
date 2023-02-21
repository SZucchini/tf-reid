import numpy as np

from kalman import KalmanFilter


class Scene:
    def __init__(self):
        self.img_hsit = None
        self.shoe_img = []
        self.shoe_hsit = None
        self.shoe_score = []
        self.start = None
        self.last = None
        self.kf = []
        self.img_file = []
        self.flag = []

    def resister(self, q):
        if self.start is None:
            self.img_hist = np.array([q['img_hist']])
            self.shoe_img.append(q['shoe_img'])
            self.shoe_hist = np.array([q['shoe_hist']])
            self.start = q['frame']
        else:
            self.img_hist = np.vstack([self.img_hist, q['img_hist']])
            self.shoe_img.append(q['shoe_img'])
            self.shoe_hist = np.vstack([self.shoe_hist, q['shoe_hist']])
        self.last = q['frame']
        self.shoe_score.append(q['shoe_score'])
        self.kf.append(KalmanFilter(np.array([q['xpos'], 0]), 1))  # 初期速度で結構変わる
        self.img_file.append(q['file'])
        self.flag.append(1)

    def update(self, q, idx):
        if q['xpos'] > 2500 and q['xpos'] < 3500:  # この閾値で、保存する画像特徴量の位置を定める
            self.img_hist[idx] = q['img_hist']
            self.img_file[idx] = q['file']
        if self.shoe_score[idx] < q['shoe_score']:
            self.shoe_img[idx] = 0
            self.shoe_img[idx] = q['shoe_img']
            self.shoe_hist[idx] = q['shoe_hist']
            self.shoe_score[idx] = q['shoe_score']
        self.last = q['frame']
        self.kf[idx].update(q['xpos'])

    def pred_xpos(self):
        pred = []
        for i in range(len(self.kf)):
            if self.flag[i]:
                xpos = self.kf[i].predict()
                if xpos > 3900:
                    self.flag[i] = 0
                    pred.append(-1)
                else:
                    pred.append(xpos)
            else:
                pred.append(-1)
        return pred
