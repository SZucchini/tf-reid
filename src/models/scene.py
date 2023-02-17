import numpy as np

from kalman import KalmanFilter


"""
query = [q1, q2, q3, ...]
qx = {
    'img_hist': img_hist,
    'shoe_hist': shoe_hist,
    'shoe_score': shoe_score,
    'xpos': xpos,
    'frame': int(file.split('/')[-1].split('_')[0]),
    'file': file
}
"""


class Scene:
    # 初期化、登録、更新、予測座標取得
    def __init__(self):
        self.img_hsit = None
        self.shoe_hsit = None
        self.shoe_score = []
        self.start = None
        self.last = None
        self.kalman = []
        self.img_file = []
        self.flag = []

    def resister(self, q):
        if self.start is None:
            self.img_hist = np.array([q['img_hist']])
            self.shoe_hist = np.array([q['shoe_hist']])
            self.start = q['frame']
        else:
            self.img_hist = np.vstack([self.img_hist, q['img_hist']])
            self.shoe_hist = np.vstack([self.shoe_hist, q['shoe_hist']])
        self.last = q['frame']
        self.shoe_score.append(q['shoe_score'])
        self.kalman.append(KalmanFilter(q['xpos'], 1))
        self.img_file.append(q['file'])
        self.flag.append(1)

    def update(self, q, idx):
        if self.shoe_score[idx] < q['shoe_score']:
            self.shoe_hist[idx] = q['shoe_hist']
            self.shoe_score[idx] = q['shoe_score']
        self.last = q['frame']
        self.kalman[idx].update(q['xpos'])

    def pred_xpos(self):
        pred = []
        for i in range(len(self.kalman)):
            if self.flag[i]:
                xpos = self.kalman[i].predict()
                if xpos > 3900:
                    self.flag[i] = 0
                    pred.append(-1)
                else:
                    pred.append(xpos)
            else:
                pred.append(-1)
        return pred
