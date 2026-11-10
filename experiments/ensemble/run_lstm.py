# -*- coding: utf-8 -*-
"""
run_lstm.py — LSTM 성분 모델 (중간 피드백 「Chronos·LSTM 비교」)
==============================================================================
구조
  · 과거 168시간(7일) 시퀀스 [SMP, 수요예측, 휴일, 시각 sin·cos] → LSTM(64) 마지막 은닉 상태
  · + 대상일 정보 [D+1 수요예측 24, 전일 SMP 24, 휴일(D+1·D), 요일 7, LNG 변동비 월초 변화]
  · → 완전연결 128 → 24시간 동시 출력, 손실 L1(VST 척도)
학습
  · 분기 첫날(1·4·7·10월) 그 이전 자료로만 재학습, 마지막 60일로 조기 종료 → 다음 분기 예측
  · 씨앗 2개 평균
산출
  outputs/ensemble/pred_lstm.pkl
"""
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from common import OOS_START, OUT, load_hourly, load_monthly, to_matrix

torch.set_num_threads(4)
SEEDS = [0, 1]


class Net(nn.Module):
    def __init__(self, c_in, f_in, hid=64):
        super().__init__()
        self.lstm = nn.LSTM(c_in, hid, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hid + f_in, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 24))

    def forward(self, seq, fut):
        _, (h, _) = self.lstm(seq)
        return self.head(torch.cat([h[-1], fut], 1))


def make_arrays(P, Dm, off, dow, lng_chg, idx, st):
    pz = np.arcsinh((P - st["pm"]) / st["ps"])
    dz = (Dm - st["dm"]) / st["ds"]
    hs, hc = np.sin(2 * np.pi * np.arange(24) / 24), np.cos(2 * np.pi * np.arange(24) / 24)
    seqs, futs = [], []
    for i in idx:
        s = np.stack([pz[i - 7:i].ravel(), dz[i - 7:i].ravel(), np.repeat(off[i - 7:i], 24),
                      np.tile(hs, 7), np.tile(hc, 7)], axis=1)
        fu = np.r_[dz[i], pz[i - 1], off[i], off[i - 1], np.eye(7)[dow[i]], lng_chg[i]]
        seqs.append(s); futs.append(fu)
    seq = np.nan_to_num(np.array(seqs, dtype=np.float32))
    fut = np.nan_to_num(np.array(futs, dtype=np.float32))
    y = pz[idx].astype(np.float32)
    return seq, fut, y


def train(seq, fut, y, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    n = len(y)
    va = np.arange(n - 60, n)
    tr = np.arange(0, n - 60)
    tr = tr[~np.isnan(y[tr]).any(1)]
    va = va[~np.isnan(y[va]).any(1)]
    T = lambda a: torch.from_numpy(a)
    net = Net(seq.shape[2], fut.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    best, best_state, wait = np.inf, None, 0
    for ep in range(80):
        net.train()
        perm = np.random.permutation(tr)
        for b in range(0, len(perm), 64):
            k = perm[b:b + 64]
            loss = (net(T(seq[k]), T(fut[k])) - T(y[k])).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = (net(T(seq[va]), T(fut[va])) - T(y[va])).abs().mean().item()
        if v < best - 1e-4:
            best, best_state, wait = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 8:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net


def main():
    h = load_hourly()
    dates = h["date"].iloc[::24].to_numpy()
    P, Dm = to_matrix(h, "smp"), to_matrix(h, "dem")
    off = h["off"].to_numpy().reshape(-1, 24)[:, 0].astype(float)
    dow = pd.to_datetime(pd.Series(dates)).dt.dayofweek.to_numpy()
    lng = load_monthly()["pm_vc_lng"]
    yms = [d[:7] for d in dates]
    lv = lng.reindex(yms).to_numpy()
    lng_chg = np.nan_to_num(lv / np.r_[lv[0], lv[:-1]] - 1)
    last = int(np.where(~np.isnan(P).all(axis=1))[0].max())
    cuts = [i for i in range(len(dates)) if dates[i] >= OOS_START and dates[i][8:] == "01"
            and dates[i][5:7] in ("01", "04", "07", "10") and i <= last] + [last + 1]
    pred = np.full((len(dates), 24), np.nan)
    t0 = time.time()
    for a, b in zip(cuts[:-1], cuts[1:]):
        tr_idx = np.arange(8, a)
        st = {"pm": np.nanmedian(P[:a]), "dm": np.nanmean(Dm[:a]), "ds": np.nanstd(Dm[:a])}
        st["ps"] = np.nanmedian(np.abs(P[:a] - st["pm"])) * 1.4826
        seq, fut, y = make_arrays(P, Dm, off, dow, lng_chg, tr_idx, st)
        te_idx = np.arange(a, b)
        s2, f2, _ = make_arrays(P, Dm, off, dow, lng_chg, te_idx, st)
        outs = []
        for sd in SEEDS:
            net = train(seq, fut, y, sd)
            with torch.no_grad():
                outs.append(net(torch.from_numpy(s2), torch.from_numpy(f2)).numpy())
        pred[te_idx] = np.sinh(np.mean(outs, 0)) * st["ps"] + st["pm"]
        print(f"  {dates[a]} ~ {dates[b - 1]}  {time.time() - t0:5.0f}초", flush=True)
    base = pd.DataFrame({"date": np.repeat(dates, 24), "hour": np.tile(np.arange(1, 25), len(dates)),
                         "y": P.ravel(), "p": pred.ravel()})
    base[base["date"] >= OOS_START].to_pickle(OUT / "pred_lstm.pkl")
    print(f"완료 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
