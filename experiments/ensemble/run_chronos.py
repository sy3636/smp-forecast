# -*- coding: utf-8 -*-
"""
run_chronos.py — Chronos-2 제로샷 D+1 예측 (표본 외, 전 기간)
==============================================================================
  · 대상일마다 그 전날 24시까지의 SMP 만 context 로 사용 → 누출 없음 (학습 자체가 없는 제로샷)
  · 공변량 : 과거 구간 값 + 대상일 값(known future)
      dem  — 전력거래소 하루전 수요예측 (D+1 값은 D일 저녁 공표)
      off  — 휴일·주말 여부 (달력)
      lng  — 변동비순위표 LNG 평균 변동비 (적용 전월 사전 공표, 월 단위 계단)
  · 변형(variant)마다 분위수 [0.05 … 0.95] 를 저장 → ensemble.py 에서 개발 구간 성적으로 선택

실행
  python experiments/ensemble/run_chronos.py                 # 전체 변형
  python experiments/ensemble/run_chronos.py --only dem_off  # 일부
산출
  outputs/ensemble/chronos_<변형>.pkl  — {"dates", "q_levels", "q"(일수×24×분위수)}
"""
import argparse
import pickle
import time

import numpy as np

from common import OUT, load_hourly, load_monthly, to_matrix

Q_LEVELS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
START = "2021-04-01"
VARIANTS = {                     # 이름 : (context 길이, 공변량)
    "uni": (2048, []),
    "dem": (2048, ["dem"]),
    "dem_off": (2048, ["dem", "off"]),
    "dem_off_c1024": (1024, ["dem", "off"]),
    "dem_off_c4096": (4096, ["dem", "off"]),
    "dem_off_lng": (2048, ["dem", "off", "lng"]),
    "off": (2048, ["off"]),      # 15시 단계용 — D+1 수요예측 없이
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    import torch  # noqa: F401
    from chronos import Chronos2Pipeline
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=args.device)

    h = load_hourly()
    dates = h["date"].iloc[::24].to_numpy()
    P = to_matrix(h, "smp")
    cov = {"dem": to_matrix(h, "dem"), "off": to_matrix(h, "off")}
    mon = load_monthly()
    lng = mon["pm_vc_lng"].ffill().bfill()
    cov["lng"] = np.repeat(lng.reindex([d[:7] for d in dates]).to_numpy()[:, None], 24, axis=1)
    last = int(np.where(~np.isnan(P).all(axis=1))[0].max())
    targets = [i for i in range(len(dates)) if dates[i] >= START and i <= last]

    for name, (ctx, covs) in VARIANTS.items():
        if args.only and name not in args.only:
            continue
        t0 = time.time()
        out = np.full((len(dates), 24, len(Q_LEVELS)), np.nan, dtype=np.float32)
        for b in range(0, len(targets), 128):
            idx = targets[b:b + 128]
            inputs = []
            for i in idx:
                item = {"target": P[:i].ravel()[-ctx:]}
                if covs:
                    item["past_covariates"] = {c: cov[c][:i].ravel()[-ctx:] for c in covs}
                    item["future_covariates"] = {c: cov[c][i] for c in covs}
                inputs.append(item)
            q, _ = pipe.predict_quantiles(inputs, prediction_length=24, quantile_levels=Q_LEVELS, batch_size=256)
            for i, x in zip(idx, q):
                out[i] = x[0].float().cpu().numpy()
        with open(OUT / f"chronos_{name}.pkl", "wb") as f:
            pickle.dump({"dates": dates, "q_levels": Q_LEVELS, "q": out}, f)
        print(f"{name:<16} {len(targets)}일  {time.time() - t0:5.0f}초", flush=True)


if __name__ == "__main__":
    main()
