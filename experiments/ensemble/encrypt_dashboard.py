# -*- coding: utf-8 -*-
"""
encrypt_dashboard.py — 대시보드를 비밀번호로 암호화한 배포용 HTML 한 장 생성 (Render 정적 사이트용)
==============================================================================
  · 입력 : outputs/ensemble/my_dashboard.html (build_my_dashboard.py 결과)
  · 방식 : PBKDF2-SHA256(60만 회)로 비밀번호에서 키를 만들고 AES-256-GCM 으로 대시보드 전체를 암호화
           → login_template.html(하늘·구름 첫 화면)에 암호문만 넣음. 비밀번호가 맞을 때만 브라우저 안에서 풀림
  · 비밀번호 : 알파벳 12자리. 화면에 보이지 않게 두 번 입력받고, 어디에도 저장·출력하지 않음
  · 산출물에는 데이터가 암호문으로만 들어 있음 — 비밀번호를 모르면 소스 보기로도 내용을 볼 수 없음

실행 (대시보드를 다시 만들 때마다 이어서 실행)
  python experiments/ensemble/build_my_dashboard.py
  python experiments/ensemble/encrypt_dashboard.py
산출
  outputs/ensemble/deploy/index.html
"""
import argparse
import base64
import getpass
import json
import os
import re
import sys
from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from common import OUT

HERE = Path(__file__).resolve().parent
ITER = 600_000                                           # PBKDF2 반복 수 (맞히기 시도를 느리게 함)
RULE = re.compile(r"[A-Za-z]{12}")


def ask_password(env_name):
    if env_name:                                         # 시험용 — 환경 변수에서 읽음
        pw = os.environ.get(env_name, "")
    else:
        pw = getpass.getpass("비밀번호 (알파벳 12자리, 화면에 보이지 않음): ")
        if getpass.getpass("한 번 더 입력: ") != pw:
            sys.exit("두 비밀번호가 다릅니다. 다시 실행해 주세요.")
    if not RULE.fullmatch(pw):
        sys.exit("비밀번호는 알파벳(A~Z, a~z) 12자리여야 합니다. 다시 실행해 주세요.")
    return pw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--password-env", default=None, help="시험용: 비밀번호를 이 환경 변수에서 읽음")
    ap.add_argument("--iter", type=int, default=ITER, help="시험용: PBKDF2 반복 수 (배포본은 기본값 그대로)")
    ap.add_argument("--out", default=str(OUT / "deploy" / "index.html"), help="시험용: 저장 위치")
    args = ap.parse_args()

    html = (OUT / "my_dashboard.html").read_text(encoding="utf-8")
    pw = ask_password(args.password_env)
    salt, iv = os.urandom(16), os.urandom(12)
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=args.iter).derive(pw.encode("utf-8"))
    ct = AESGCM(key).encrypt(iv, html.encode("utf-8"), None)          # 암호문 + 인증 태그 (WebCrypto 와 같은 형식)
    enc = lambda b: base64.b64encode(b).decode("ascii")
    payload = {"iter": args.iter, "salt": enc(salt), "iv": enc(iv), "ct": enc(ct)}

    page = (HERE / "login_template.html").read_text(encoding="utf-8")
    page = page.replace("/*__PAYLOAD__*/null", json.dumps(payload, separators=(",", ":")))
    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(page, encoding="utf-8")
    print(f"저장 : {dst}  ({len(page) / 1e6:.2f}MB) — 대시보드는 암호문으로만 들어 있음")
    print("이 파일 하나를 Render 정적 사이트에 올리면 됩니다. 비밀번호는 따로 안전하게 보관하세요.")


if __name__ == "__main__":
    main()
