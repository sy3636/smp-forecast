# -*- coding: utf-8 -*-
"""
run_chronos_ft.py — Chronos-2 미세조정(LoRA) D+1 예측
==============================================================================
  · 매년 1월 1일, 그 이전 SMP·수요예측·휴일만으로 LoRA 미세조정 → 그 해 예측 (누출 없음)
  · 공변량 dem·off 는 「미래 값을 아는 변수」로 지정(known_covariates_names)
  · 예측 방식·저장 형식은 run_chronos.py 와 동일 → ensemble.py 에서 chr_ft_<태그> 로 사용

실행
  python experiments/ensemble/run_chronos_ft.py --steps 50 --cuts 2025-01-01 --tag speed   # 속도 확인
  python experiments/ensemble/run_chronos_ft.py --tag lora
"""
import argparse
import pickle
import shutil
import tempfile
import time

import numpy as np

from common import OUT, load_hourly, to_matrix
from run_chronos import Q_LEVELS

CUTS = ["2023-01-01", "2024-01-01", "2025-01-01", "2026-01-01"]
COVS = ["dem", "off"]
CTX = 2048


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="lora", choices=["lora", "full"])
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--cuts", nargs="*", default=CUTS)
    ap.add_argument("--tag", default="lora")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--until", default="9999-12-31", help="마지막 구간의 끝(미포함)")
    ap.add_argument("--quarterly", action="store_true", help="2023-01 부터 분기 첫날마다 미세조정 (--cuts 무시)")
    ap.add_argument("--merge", action="store_true", help="기존 chronos_ft_<태그>.pkl 에서 다시 계산하지 않는 날짜는 유지")
    args = ap.parse_args()

    from chronos import Chronos2Pipeline
    from chronos.chronos2.preprocess import from_list_of_dicts

    h = load_hourly()
    dates = h["date"].iloc[::24].to_numpy()
    P = to_matrix(h, "smp")
    cov = {"dem": to_matrix(h, "dem"), "off": to_matrix(h, "off")}
    # 마지막 예측 대상 = 수요예측이 있는 마지막 날 (SMP 는 전날까지만 있으면 됨)
    last = int(np.where(~np.isnan(to_matrix(h, "dem")).all(axis=1))[0].max())
    out = np.full((len(dates), 24, len(Q_LEVELS)), np.nan, dtype=np.float32)
    path = OUT / f"chronos_ft_{args.tag}.pkl"
    if args.merge and path.exists():
        with open(path, "rb") as f:
            old = pickle.load(f)
        pos = {d: i for i, d in enumerate(dates)}
        for j, d in enumerate(old["dates"]):
            if d in pos:
                out[pos[d]] = old["q"][j]
    if args.quarterly:
        args.cuts = [f"{y}-{m:02d}-01" for y in range(2023, 2027) for m in (1, 4, 7, 10)]
        args.cuts = [c for c in args.cuts if c <= dates[last]]
    cuts = list(args.cuts) + [args.until]
    t0 = time.time()
    for a, b in zip(cuts[:-1], cuts[1:]):
        ia = int(np.searchsorted(dates, a))
        base = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=args.device)
        train = [{"target": P[:ia].ravel(), "past_covariates": {c: cov[c][:ia].ravel() for c in COVS}}]
        train = from_list_of_dicts(train, prediction_length=24, known_covariates_names=COVS)
        tmp = tempfile.mkdtemp(prefix="chronos_ft_")
        pipe = base.fit(train, prediction_length=24, finetune_mode=args.mode, context_length=CTX,
                        learning_rate=args.lr, num_steps=args.steps, batch_size=args.batch, output_dir=tmp,
                        remove_printer_callback=True, report_to="none", logging_steps=10 ** 6,
                        save_strategy="no")
        shutil.rmtree(tmp, ignore_errors=True)
        pipe.model.eval()                              # 학습 모드(dropout) 해제 — MPS 어텐션은 dropout 미지원
        t_fit = time.time() - t0
        targets = [i for i in range(ia, len(dates)) if dates[i] < b and i <= last]
        for k in range(0, len(targets), 128):
            idx = targets[k:k + 128]
            inputs = [{"target": P[:i].ravel()[-CTX:],
                       "past_covariates": {c: cov[c][:i].ravel()[-CTX:] for c in COVS},
                       "future_covariates": {c: cov[c][i] for c in COVS}} for i in idx]
            q, _ = pipe.predict_quantiles(inputs, prediction_length=24, quantile_levels=Q_LEVELS, batch_size=256)
            for i, x in zip(idx, q):
                out[i] = x[0].float().cpu().numpy()
        print(f"  {a} 미세조정 {t_fit:5.0f}초 · 누적 {time.time() - t0:5.0f}초 ({len(targets)}일 예측)", flush=True)
        del pipe, base
    with open(OUT / f"chronos_ft_{args.tag}.pkl", "wb") as f:
        pickle.dump({"dates": dates, "q_levels": Q_LEVELS, "q": out}, f)
    print(f"완료 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
