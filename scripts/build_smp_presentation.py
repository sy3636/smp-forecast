# -*- coding: utf-8 -*-
"""Build the six-slide 4-2 team SMP forecast presentation."""
import csv
from collections import Counter
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "ensemble" / "SMP_예측_발표_4-2조.pptx"
DATA = ROOT / "data" / "processed" / "smp_dataset_hourly.csv"
ENSEMBLE = ROOT / "outputs" / "ensemble"
START, END = "2025-01-01", "2026-10-31"

W, H = 13.333, 7.5
NAVY = "101C34"
NAVY2 = "1B2A46"
INK = "18243A"
MUTED = "64748B"
BG = "F4F7FB"
WHITE = "FFFFFF"
LINE = "DCE4EF"
BLUE = "2E72D2"
CYAN = "29B6B0"
ORANGE = "F39A3C"
RED = "D85B5B"
GREEN = "2B9A78"
PURPLE = "8655B5"
FONT = "Apple SD Gothic Neo"

EVENTS = [
    ("spike", "육지 급등", "휴일", "2025-03-23", "demand_fcst_land", "warnings_spike.csv"),
    ("spike", "육지 급등", "평일", "2026-01-09", "demand_fcst_land", "warnings_spike.csv"),
    ("crash", "육지 급락", "휴일", "2025-03-09", "demand_fcst_land", "warnings_crash.csv"),
    ("crash", "육지 급락", "평일", "2025-11-26", "demand_fcst_land", "warnings_crash.csv"),
    ("jeju", "제주 급락", "휴일", "2025-03-22", "demand_fcst_jeju", "warnings_jeju_v2.csv"),
    ("jeju", "제주 급락", "평일", "2025-03-28", "demand_fcst_jeju", "warnings_jeju_v2.csv"),
]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fmt(value, digits=1):
    return f"{value:,.{digits}f}"


def rgb(hex_color):
    return RGBColor.from_string(hex_color)


def set_background(slide, color=BG):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = rgb(color)


def rect(slide, x, y, w, h, color, radius=True, line_color=None):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(color)
    if line_color:
        shape.line.color.rgb = rgb(line_color)
        shape.line.width = Pt(0.8)
    else:
        shape.line.fill.background()
    return shape


def text(slide, x, y, w, h, value, size=16, color=INK, bold=False,
         align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.MIDDLE, margin=0.02):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(margin)
    frame.margin_right = Inches(margin)
    frame.margin_top = Inches(margin)
    frame.margin_bottom = Inches(margin)
    frame.vertical_anchor = valign
    p = frame.paragraphs[0]
    p.alignment = align
    p.space_before = Pt(0)
    p.space_after = Pt(0)
    run = p.add_run()
    run.text = str(value)
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = rgb(color)
    return box


def title(slide, eyebrow, heading, subheading=None, page=None):
    text(slide, 0.62, 0.28, 10.8, 0.26, eyebrow.upper(), 10, BLUE, True)
    text(slide, 0.62, 0.58, 12.0, 0.52, heading, 26, NAVY, True)
    if subheading:
        text(slide, 0.64, 1.08, 12.0, 0.36, subheading, 11, MUTED)
    rect(slide, 0.62, 1.49, 12.05, 0.012, LINE, radius=False)
    if page is not None:
        text(slide, 12.2, 7.12, 0.45, 0.2, f"{page:02d}", 9, MUTED, True, PP_ALIGN.RIGHT)


def add_cell(slide, x, y, w, h, value, *, fill=WHITE, color=INK, size=12,
             bold=False, align=PP_ALIGN.CENTER):
    rect(slide, x, y, w, h, fill, radius=False, line_color=LINE)
    text(slide, x + 0.04, y + 0.02, w - 0.08, h - 0.04, value, size, color, bold, align)


def load_examples():
    segments = read_csv(ENSEMBLE / "events_segments.csv")
    daily = read_csv(DATA)
    explain = {r["date"]: r for r in read_csv(ENSEMBLE / "explain_daily.csv")}
    warnings = {name: {r["date"]: r for r in read_csv(ENSEMBLE / name)}
                for name in {e[5] for e in EVENTS}}
    data_by_day = {}
    for row in daily:
        date = row["date"]
        if date in {e[3] for e in EVENTS}:
            data_by_day.setdefault(date, []).append(row)

    examples = []
    for key, name, group, date, demand_col, warning_file in EVENTS:
        event_rows = [r for r in segments if r["key"] == key and r["date"] == date]
        if not event_rows:
            raise ValueError(f"No event segment for {key} on {date}")
        segment = max(event_rows, key=lambda r: int(r["시간수"]))
        h1, h2 = int(segment["시작시"]), int(segment["끝시"])
        relevant = [r for r in data_by_day[date] if h1 <= int(r["hour_n"]) <= h2]
        demand = [float(r[demand_col]) for r in relevant if r.get(demand_col)]
        if not demand:
            raise ValueError(f"No forecast demand for {date}")
        alert = warnings[warning_file].get(date)
        if alert is None:
            raise ValueError(f"No warning score for {date} in {warning_file}")

        actual = float(segment["실적평균"])
        threshold = float(segment["문턱"])
        if key == "spike":
            price = f"{fmt(actual)}원/kWh · {h1}~{h2}시 (기준 {fmt(threshold)}원)"
            event_hours = f"{h1}~{h2}시"
        elif key == "crash":
            price = f"{fmt(actual)}원/kWh · {h1}~{h2}시 (<30원)"
            event_hours = f"{h1}~{h2}시"
        else:
            price = f"{fmt(actual)}원/kWh · {h1}~{h2}시 (0원 이하)"
            event_hours = f"{h1}~{h2}시"

        prob = float(alert["확률"]) * 100
        level = alert["단계"]
        if key == "jeju":
            reason = f"전날 15시 제주 급락 확률 {prob:.1f}% · {level} (구간 적중)"
        else:
            e = explain.get(date)
            if e is None:
                reason = f"전날 예측 경보 {prob:.1f}% · {level}"
            else:
                contributions = [
                    ("최근 가격", float(e["최근 가격"])),
                    ("전력 수요", float(e["전력 수요"])),
                    ("태양광", float(e["태양광"])),
                    ("요일·휴일", float(e["요일·휴일"])),
                    ("연료비·원전", float(e["연료비·원전"])),
                    ("계절·시간", float(e["계절·시간"])),
                ]
                top = sorted(contributions, key=lambda pair: abs(pair[1]), reverse=True)[:2]
                why = " · ".join(f"{label} {value:+.1f}원" for label, value in top)
                reason = f"경보 {prob:.1f}% · 일평균 예측 기여: {why}"
        examples.append({
            "key": key, "name": name, "group": group, "date": date,
            "price": price, "demand": f"{fmt(sum(demand) / len(demand))} MW · 사건 시간대 {event_hours} 평균",
            "reason": reason, "hours": int(segment["시간수"]),
        })
    return examples


def calculate_tables():
    segments = [r for r in read_csv(ENSEMBLE / "events_segments.csv")
                if START <= r["date"] <= END]
    table1 = {}
    for key, name, group, *_ in EVENTS:
        rows = [r for r in segments if r["key"] == key and r["구분"] == group]
        table1[(key, group)] = {
            "n": len(rows),
            "success": sum(r["경보모델"] == "성공" for r in rows),
            "part": sum(r["경보모델"] == "부분" for r in rows),
            "fail": sum(r["경보모델"] == "실패" for r in rows),
        }

    false_alarms = [r for r in read_csv(ENSEMBLE / "events_false_alarms.csv")
                    if r["모델"] == "경보" and START <= r["date"] <= END]
    fa = {}
    for key, name, group, *_ in EVENTS:
        rows = [r for r in false_alarms if r["key"] == key and r["구분"] == group]
        fa[(key, group)] = {
            "watch": sum(r["단계"] == "주의" for r in rows),
            "alarm": sum(r["단계"] == "경보" for r in rows),
        }

    daily = {r["date"]: r for r in read_csv(ENSEMBLE / "events_daily.csv")}
    cm = {}
    for key, name, group, _, _, warning_file in EVENTS:
        rows = read_csv(ENSEMBLE / warning_file)
        counts = Counter()
        for r in rows:
            date = r["date"]
            if not (START <= date <= END) or date not in daily:
                continue
            actual_group = r.get("구분") or daily[date]["구분"]
            if actual_group != group:
                continue
            actual = float(r["실제"]) == 1
            predicted = r["단계"] in ("주의", "경보")
            counts[("TP" if predicted else "FN") if actual else ("FP" if predicted else "TN")] += 1
        cm[(key, group)] = counts
    return table1, fa, cm


def slide_cover(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_background(slide, NAVY)
    rect(slide, 0, 0, 0.16, H, CYAN, radius=False)
    rect(slide, 9.1, 0, 4.23, 3.7, NAVY2, radius=True)
    rect(slide, 10.1, 0.9, 3.2, 0.04, CYAN, radius=False)
    for i, height in enumerate([0.75, 1.18, 1.66, 2.2, 1.45, 1.95, 1.12]):
        rect(slide, 9.7 + i * 0.46, 5.85 - height, 0.22, height, [BLUE, CYAN, ORANGE][i % 3], radius=True)
    text(slide, 0.92, 0.78, 4.8, 0.45, "IT와 비즈니스 혁신", 17, "B6C7DF", True)
    text(slide, 0.92, 1.68, 8.6, 1.5, "SMP 예측 프로그램", 38, WHITE, True)
    rect(slide, 0.96, 3.45, 1.15, 0.055, CYAN, radius=False)
    text(slide, 0.92, 3.84, 5.0, 0.55, "4-2조 발표", 22, WHITE, True)
    text(slide, 0.92, 6.62, 7.3, 0.32, "한국중부발전 · 시간별 전력시장가격 예측", 12, "B6C7DF")
    text(slide, 12.1, 7.04, 0.55, 0.24, "01", 10, "B6C7DF", True, PP_ALIGN.RIGHT)


def slide_project(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_background(slide)
    title(slide, "PROJECT OVERVIEW", "무엇을 만들었나요?",
          "전력시장 가격 예측을 처음 접하는 청중을 위한 프로젝트 소개", 2)

    rect(slide, 0.66, 1.78, 12.0, 1.0, NAVY, radius=True)
    text(slide, 0.95, 1.92, 2.15, 0.26, "한국중부발전", 13, "8BD9D4", True)
    text(slide, 0.95, 2.22, 10.9, 0.38,
         "매 시간 달라지는 전력시장가격(SMP)을 예측하는 모델과 결과 대시보드를 구현했습니다.",
         18, WHITE, True)

    cards = [
        ("01", "데이터", "시간별 SMP 실적·수요예측·달력 등\n과거 정보를 정리", BLUE),
        ("02", "예측·경보", "다음 날 육지 SMP를 시간별로 예측하고\n급등·급락 위험을 분류", CYAN),
        ("03", "대시보드", "실적과 예측, 불확실성 범위,\n사건별 판정·혼동행렬을 확인", ORANGE),
    ]
    for i, (num, heading, desc, accent) in enumerate(cards):
        x = 0.66 + i * 4.08
        rect(slide, x, 3.08, 3.82, 1.64, WHITE, radius=True, line_color=LINE)
        rect(slide, x + 0.22, 3.31, 0.48, 0.4, accent, radius=True)
        text(slide, x + 0.23, 3.34, 0.46, 0.3, num, 10, WHITE, True, PP_ALIGN.CENTER)
        text(slide, x + 0.82, 3.28, 2.7, 0.35, heading, 17, NAVY, True)
        text(slide, x + 0.24, 3.88, 3.35, 0.63, desc, 12, MUTED)

    rect(slide, 0.66, 4.98, 12.0, 1.38, "E8EEF7", radius=True)
    text(slide, 0.94, 5.16, 2.1, 0.28, "왜 필요한가요?", 14, NAVY, True)
    text(slide, 0.94, 5.54, 11.25, 0.5,
         "SMP는 전력 수급과 발전 여건에 따라 시간대별로 변합니다. 예측과 경보를 한 화면에서 살펴보면 "
         "가격 변동의 패턴과 모델이 놓치거나 잘 잡는 사건을 빠르게 파악할 수 있습니다.",
         13, INK)
    text(slide, 0.68, 6.7, 11.8, 0.28,
         "범위: 시간별 육지 가격 예측 + 육지 급등·급락 및 제주 급락의 일 단위 경보",
         10, MUTED)


def slide_results(prs, table1, false_alarms):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_background(slide)
    title(slide, "TEST RESULTS · PROBABILITY / ALERT MODEL",
          "시험 구간 사건 판정과 헛경보",
          "시험: 2025.01.01–2026.10.31 · 학습: 2020–2024 · 경보 예측은 사건 전날",
          3)

    left_x, left_w = 0.65, 7.7
    right_x, right_w = 8.56, 4.12
    rect(slide, left_x, 1.75, left_w, 4.98, WHITE, radius=True, line_color=LINE)
    rect(slide, right_x, 1.75, right_w, 4.98, WHITE, radius=True, line_color=LINE)
    text(slide, left_x + 0.24, 1.91, left_w - 0.45, 0.36, "표 1 · 사건 판정  |  건수(사건 구간 비율)", 15, NAVY, True)
    text(slide, right_x + 0.23, 1.91, right_w - 0.4, 0.36, "표 2 · 헛경보", 15, NAVY, True)

    headers = ["사건", "구분", "구간 수", "성공\n(경보)", "부분\n(주의)", "실패\n(놓침)"]
    widths = [1.32, 0.76, 0.95, 1.54, 1.54, 1.25]
    x = left_x + 0.2
    y = 2.47
    for header, width in zip(headers, widths):
        add_cell(slide, x, y, width, 0.5, header, fill=NAVY, color=WHITE, size=10, bold=True)
        x += width
    labels = [("spike", "육지 급등"), ("crash", "육지 급락"), ("jeju", "제주 급락")]
    row_y = y + 0.5
    for key, name in labels:
        for group in ("휴일", "평일"):
            c = table1[(key, group)]
            values = [name, group, str(c["n"])]
            values.extend(f"{c[k]} ({(100*c[k]/c['n']):.1f}%)" if c["n"] else "0 (—)"
                          for k in ("success", "part", "fail"))
            x = left_x + 0.2
            for idx, (value, width) in enumerate(zip(values, widths)):
                fill = "F6F8FC" if row_y % 2 else WHITE
                if idx == 0 and group == "평일":
                    value = "↳"
                add_cell(slide, x, row_y, width, 0.49, value, fill=fill,
                         color=BLUE if idx == 3 else INK, size=10, bold=idx in (0, 2, 3))
                x += width
            row_y += 0.49

    text(slide, left_x + 0.25, 6.03, left_w - 0.45, 0.46,
         "경보=성공 · 주의=부분 · 미경보=실패  |  한 사건이 여러 시간 이어지면 1구간",
         9, MUTED)

    small_headers = ["사건", "휴일", "평일", "계"]
    small_widths = [1.28, 0.92, 0.92, 0.62]
    x = right_x + 0.18
    for header, width in zip(small_headers, small_widths):
        add_cell(slide, x, 2.47, width, 0.48, header, fill=NAVY, color=WHITE, size=10, bold=True)
        x += width
    row_y = 2.95
    for key, name in labels:
        row_values = [name]
        for group in ("휴일", "평일"):
            v = false_alarms[(key, group)]
            row_values.append(f"{v['watch'] + v['alarm']}\n주의 {v['watch']} · 경보 {v['alarm']}")
        total_watch = sum(false_alarms[(key, g)]["watch"] for g in ("휴일", "평일"))
        total_alarm = sum(false_alarms[(key, g)]["alarm"] for g in ("휴일", "평일"))
        row_values.append(f"{total_watch + total_alarm}\n{total_watch} · {total_alarm}")
        x = right_x + 0.18
        for value, width in zip(row_values, small_widths):
            add_cell(slide, x, row_y, width, 0.82, value,
                     fill="F6F8FC" if row_y < 4 else WHITE,
                     size=9 if "\n" in value else 10, bold=True)
            x += width
        row_y += 0.82
    text(slide, right_x + 0.22, 5.55, right_w - 0.42, 0.83,
         "주의·경보를 냈지만 해당 사건이 없었던 날 수\n(표기: 전체 · 주의 · 경보)",
         10, MUTED)
    text(slide, 0.69, 6.86, 11.9, 0.2,
         "자료: events_segments.csv · events_false_alarms.csv · events_report.json | 기간·사건 정의는 팀 재현 안내 기준",
         8, MUTED)


def add_matrix_panel(slide, y, key, name, group, cm, example, accent):
    x, w, h = 0.66, 12.0, 2.72
    rect(slide, x, y, w, h, WHITE, radius=True, line_color=LINE)
    rect(slide, x, y, 0.09, h, accent, radius=False)
    text(slide, x + 0.25, y + 0.12, 5.7, 0.34, f"{name}  ·  {group}", 16, NAVY, True)
    c = cm[(key, group)]
    total = sum(c[k] for k in ("TP", "FN", "FP", "TN"))
    precision = c["TP"] / (c["TP"] + c["FP"]) * 100 if c["TP"] + c["FP"] else None
    recall = c["TP"] / (c["TP"] + c["FN"]) * 100 if c["TP"] + c["FN"] else None
    metric = f"표본 {total}일   ·   정밀도 {fmt(precision, 1) + '%' if precision is not None else '—'}   ·   재현율 {fmt(recall, 1) + '%' if recall is not None else '—'}"
    text(slide, x + 0.28, y + 0.49, 5.5, 0.22, metric, 9, MUTED)

    mx, my = x + 0.28, y + 0.82
    row_label_w, cell_w, cell_h = 1.05, 1.26, 0.55
    text(slide, mx + row_label_w, my - 0.12, cell_w * 2, 0.18, "예측", 8, MUTED, True, PP_ALIGN.CENTER)
    add_cell(slide, mx, my + 0.05, row_label_w, 0.34, "실제", fill="EAF0F8", size=9, bold=True)
    add_cell(slide, mx + row_label_w, my + 0.05, cell_w, 0.34, "있음", fill="EAF0F8", size=9, bold=True)
    add_cell(slide, mx + row_label_w + cell_w, my + 0.05, cell_w, 0.34, "없음", fill="EAF0F8", size=9, bold=True)
    matrix_rows = [
        ("있음", ("TP", c["TP"], "적중"), ("FN", c["FN"], "놓침")),
        ("없음", ("FP", c["FP"], "헛경보"), ("TN", c["TN"], "정상")),
    ]
    for ri, (actual_label, first, second) in enumerate(matrix_rows):
        cy = my + 0.39 + ri * cell_h
        add_cell(slide, mx, cy, row_label_w, cell_h, actual_label, fill="EAF0F8", size=9, bold=True)
        for ci, (label, value, descr) in enumerate((first, second)):
            color = GREEN if label == "TP" else (ORANGE if label == "FP" else (BLUE if label == "FN" else NAVY2))
            add_cell(slide, mx + row_label_w + ci * cell_w, cy, cell_w, cell_h,
                     f"{value}  {descr}", fill="F5F8FC", color=color, size=12, bold=True)

    dx = x + 5.85
    rect(slide, dx, y + 0.55, 5.82, 1.9, "F4F7FB", radius=True)
    text(slide, dx + 0.2, y + 0.68, 5.36, 0.24, "실제 발생일 사례 · 사건 시간대 요약", 10, accent, True)
    fields = [
        ("날짜", example["date"]),
        ("SMP 실적", example["price"]),
        ("수요예측", example["demand"]),
        ("예측 근거", example["reason"]),
    ]
    row_y = y + 0.99
    for label, value in fields:
        text(slide, dx + 0.2, row_y, 0.93, 0.28, label, 9, MUTED, True)
        text(slide, dx + 1.15, row_y, 4.43, 0.34, value, 9 if label != "예측 근거" else 8.5, INK,
             label == "날짜")
        row_y += 0.36


def slide_event_matrices(prs, page, entries, cm, examples):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_background(slide)
    event_name = entries[0][1]
    title(slide, f"EVENT REVIEW · {event_name}", f"{event_name} 혼동행렬과 실제 사례",
          "확률(경보) 모델 · 예측 있음=주의 또는 경보 · 실제 있음=해당 날짜에 사건 구간 1개 이상",
          page)
    colors = {"spike": ORANGE, "crash": BLUE, "jeju": PURPLE}
    for idx, entry in enumerate(entries):
        key, name, group, *_ = entry
        add_matrix_panel(slide, 1.72 + idx * 2.77, key, name, group, cm,
                         examples[(key, group)], colors[key])
    text(slide, 0.72, 7.22, 11.6, 0.16,
         "가격·수요는 실제 사건 시간대 평균. 육지 예측 기여는 일평균 모델 설명(SHAP)이며 인과관계 증명은 아닙니다. 제주 근거는 전날 경보확률.",
         7.5, MUTED)


def main():
    examples_list = load_examples()
    examples = {(e["key"], e["group"]): e for e in examples_list}
    table1, false_alarms, cm = calculate_tables()
    expected_events = {
        ("spike", "휴일"): (109, 64, 33, 12),
        ("spike", "평일"): (429, 353, 72, 4),
        ("crash", "휴일"): (36, 29, 0, 7),
        ("crash", "평일"): (3, 0, 0, 3),
        ("jeju", "휴일"): (52, 39, 0, 13),
        ("jeju", "평일"): (56, 44, 0, 12),
    }
    for key, values in expected_events.items():
        actual = table1[key]
        if (actual["n"], actual["success"], actual["part"], actual["fail"]) != values:
            raise ValueError(f"Unexpected event score for {key}: {actual}")
    expected_cm = {
        ("spike", "휴일"): (60, 74, 11, 78),
        ("spike", "평일"): (222, 197, 3, 24),
        ("crash", "휴일"): (29, 60, 7, 127),
        ("crash", "평일"): (0, 0, 3, 443),
        ("jeju", "휴일"): (37, 54, 13, 119),
        ("jeju", "평일"): (44, 150, 12, 240),
    }
    for key, values in expected_cm.items():
        actual = cm[key]
        if tuple(actual[k] for k in ("TP", "FP", "FN", "TN")) != values:
            raise ValueError(f"Unexpected confusion matrix for {key}: {actual}")
    expected_false_alarms = {
        ("spike", "휴일"): (56, 18),
        ("spike", "평일"): (102, 95),
        ("crash", "휴일"): (0, 60),
        ("crash", "평일"): (0, 0),
        ("jeju", "휴일"): (0, 54),
        ("jeju", "평일"): (0, 150),
    }
    for key, values in expected_false_alarms.items():
        actual = false_alarms[key]
        if (actual["watch"], actual["alarm"]) != values:
            raise ValueError(f"Unexpected false alarm count for {key}: {actual}")

    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)
    prs.core_properties.title = "SMP 예측 프로그램 · 4-2조 발표"
    prs.core_properties.subject = "IT와 비즈니스 혁신"
    prs.core_properties.author = "4-2조"

    slide_cover(prs)
    slide_project(prs)
    slide_results(prs, table1, false_alarms)
    slide_event_matrices(prs, 4, EVENTS[0:2], cm, examples)
    slide_event_matrices(prs, 5, EVENTS[2:4], cm, examples)
    slide_event_matrices(prs, 6, EVENTS[4:6], cm, examples)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(f"Saved {OUT} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
