#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""충북 보은군 회인면 면소재지 3D 지도 — 시안 v0.6 (건물 + 지형·하천·옹벽 + 교량 + 띄운 고속도로 + 입체 일반 도로).

목원대 생활권 3D 지도(clients/인큐넥스트/2026년 하반기 목원대학교 교육/3d-map)의 빌더·템플릿을 가져왔다.
원본 (건물): GIS건물통합정보 AL_D010 충북 (국토교통부, 2026-09-06 기준, EPSG:5186, cp949)
      필드는 A0~A28 로 익명화돼 있다. 표준 순서로 읽는다:
      A4 법정동 · A5 지번 · A9 용도 · A16 높이(m) · A24 건물명 · A25 동명 · A26 지상층수
      1.5GB 라 한 번 읽어 회인면 둘레만 data/ 에 캐시한다.
      건물 모양은 수치지도 2025 건물(B001)을 쓰고 대장은 속성만 붙인다 (merge_buildings — 네이버 최신 지도와 대조한 결과).
원본 (지형·도로): 수치지도 2.0 1:5,000 (국토지리정보원, 2025 제작, EPSG:5186, euc_kr)
      도엽 36711004(보은004, 면소재지) · 36711005(보은005, 동쪽) → data/raw/ngii/<도엽>/
      F001 등고선 · F002 표고점 → 지형 / A002 도로중심선 × 도로폭 → 입체 도로(고속국도·일반) / A001 도로면 → 포장면 그림 /
      D001 논밭 · E001 하천 구역(둔치) · E003 수면 → 지면 그림 / F004 옹벽 · B002 담장 → 벽 / A007 교량 → 다리 상판.
      지도 범위는 두 도엽이 덮는 곳 안으로만 잡는다 — 북쪽(36707094·095)은 받지 않아 그 구역 건물도 뺐다.
출력: 회인면_3D지도_시안.html (데이터·UI 인라인, three.js 만 CDN)

의존: pyshp numpy scipy shapely  (python3 -m venv .venv && .venv/bin/pip install pyshp numpy scipy shapely)
실행: .venv/bin/python build_map.py [AL_D010_43 zip 경로]
"""
import json
import math
import os
import re
import sys
import tempfile
import zipfile

import numpy as np
import shapefile
from scipy.interpolate import griddata
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/projects/products/블렌더 3D 모델링/지도/대전-충남-충북 건물 지도/AL_D010_43_20260909.zip")
CACHE = os.path.join(HERE, "data", "AL_D010_43_회인면.json")
OUT = os.path.join(HERE, "회인면_3D지도_시안.html")

CENTER = ("충청북도 보은군 회인면 중앙리", "172-2")   # 회인면사무소 — 지도 원점
# 캐시 범위(EPSG:5186): 회인면 건물 전체 범위 + 1km
CACHE_BOX = (249800, 258200, 428200, 439900)
# 원점 기준 잘라낼 범위(m): 서·동·남·북 — 면소재지(중앙리) + 둘레 마을. 북쪽은 도엽 끝(+573m)에서 멈춘다
BOX = (-1200, 1300, -1900, 520)
NGII = os.path.join(HERE, "data", "raw", "ngii")
SHEETS = ("36711004", "36711005")               # 두 도엽 합친 범위: 서 -1918 · 동 +2579 · 남 -2214 · 북 +573
# 지면 범위(m): 건물 범위보다 넓게, 단 도엽 밖으로는 나가지 않게 (밖은 등고선이 없어 보간이 무너진다)
GROUND = (-1800, 1790, -2200, 570)             # 동쪽은 고속도로 본선이 회인대교 상·하행으로 갈라지기 직전(+1803m)까지
CELL = 10                                       # 높이 격자 간격(m)
MATCH_IOU = 0.2                                 # 수치지도 건물에 대장 속성을 붙일 최소 겹침(IoU)
OVERLAP_M2 = 1.0                                # 건물끼리 이 넓이(㎡) 넘게 겹치면 안 된다 — 벽이 맞닿은 정도의 오차만 허용
BRIDGE_MIN, BRIDGE_PAD = 15, 8                  # 띄울 교량 최소 연장(m) · 교대 높이를 잴 다리 끝 바깥 거리(m)
BRIDGE_CLEAR, BRIDGE_HIGH = 3, 5                # 다리 밑 지형이 상판보다 최소 이만큼 낮게(m) · 이만큼 넘게 뜬 곳이 있어야 높은 다리
HW_MAIN, HW_STEP, HW_SIGMA = 23, 5, 40         # 고속도로 본선 최소 폭(m) · 높이 표본 간격(m) · 높이 평활 폭(m)
HW_SIDE = 40                                    # 고속도로 밑 성토를 걷어 낼 때 자연 지면을 재는 거리(도로 가장자리 밖 m)
RAMP_CLEAR, RAMP_BLEND = 2.3, 25                # 램프 밑 지형을 상판면보다 이만큼 아래로 (상판 두께 1.8m + 0.5m) · 일반 도로와 만나는 끝에서 이 거리(m) 동안 0까지 줄인다
ROAD_W0, ROAD_STEP, ROAD_SIGMA = 3, 4, 15       # 폭 정보 없는 도로 폭(m) · 높이 표본 간격(m) · 노면 높이 평활 폭(m)
ROAD_LIFT, ROAD_BLEND = 0.15, 20                # 노면을 지형보다 이만큼 위로(m) · 교차점 높이 맞춤을 잇는 거리(m)
FLOOR_H = 3.3                                   # 층수만 있을 때 한 층 높이(m)
MAX_FLOOR_H = 6.0                               # 원본 높이가 층당 이보다 크면 입력 오류로 보고 층수로 추정
SNAPSHOT = "2026-09-06"

# 이름을 보여 줄 공공·마을 시설. 건물명에 소유자 개인 이름이 많아(○○씨 주택) 그 밖은 이름을 숨긴다.
PUBLIC = re.compile(r"면사무소|행정복지|지구대|파출소|보건소|보건지소|보건진료|우체국|학교|유치원|서당|경로당|마을회|복지|회관|"
                    r"농협|교회|성당|전시관|터널|주유소|센터|도서관|향교|객사")
PUBLIC_USE = r"교육|업무|노유자|문화|종교|운동|의료"

AXDS = os.path.expanduser("~/projects/internal/design-systems/axds")
sys.path.insert(0, AXDS)
from theme_doc import CSS as AXDS_CSS  # noqa: E402

ROOT_TOKENS = re.search(r":root\s*\{[^}]*\}", AXDS_CSS).group(0)


def num(v):
    return v if isinstance(v, (int, float)) else 0


def cache():
    """충북 전체(zip)에서 회인면 둘레 건물만 골라 JSON 으로 저장한다. 한 번만 돈다."""
    keys = ("A4", "A5", "A9", "A16", "A24", "A25", "A26")
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(SRC) as z:
            names = [n for n in z.namelist() if n.endswith((".shp", ".shx", ".dbf"))]
            z.extractall(tmp, names)
        base = os.path.join(tmp, names[0][:-4])
        r = shapefile.Reader(base, encoding="cp949")
        rows = []
        for sr in r.iterShapeRecords():
            b = sr.shape.bbox
            if b[2] < CACHE_BOX[0] or b[0] > CACHE_BOX[1] or b[3] < CACHE_BOX[2] or b[1] > CACHE_BOX[3]:
                continue
            d = sr.record.as_dict()
            rows.append({"pts": [[round(x, 2), round(y, 2)] for x, y in sr.shape.points],
                         "parts": list(sr.shape.parts), **{k: d[k] for k in keys}})
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    return rows


def load():
    rows = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else cache()
    hit = [x for x in rows if (x["A4"], x["A5"]) == CENTER]
    if not hit:
        sys.exit("회인면사무소(중앙리 172-2) 건물을 찾지 못했습니다 — 원본 파일을 확인하세요.")
    pts = [p for x in hit for p in x["pts"]]
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return rows, cx, cy


def height(d):
    """(높이 m, 출처 0=원본 높이 · 1=층수 추정 · 2=기본값 · 3=수치지도 층수 추정(merge_buildings))"""
    floors = num(d["A26"])
    if num(d["A16"]) > 0 and num(d["A16"]) <= max(floors, 1) * MAX_FLOOR_H:
        return num(d["A16"]), 0
    if floors > 0:
        return floors * FLOOR_H, 1
    return FLOOR_H, 2


def label(d):
    """공공·마을 시설 이름만. '회인면 오동리 131-9 제1종근린생활시설 (오동리마을회)' → '오동리마을회'"""
    name = (d["A24"] or "").strip()
    m = re.search(r"\(([^)]+)\)\s*$", name)
    if m:
        name = m.group(1).strip()
    return name if PUBLIC.search(name) else ""


def public(d, name):
    return bool(name) or bool(re.search(PUBLIC_USE, d["A9"] or ""))


def extract(rows, cx, cy):
    out = []
    for d in rows:
        pts = d["pts"]
        mx = (min(p[0] for p in pts) + max(p[0] for p in pts)) / 2 - cx
        my = (min(p[1] for p in pts) + max(p[1] for p in pts)) / 2 - cy
        if not (BOX[0] <= mx <= BOX[1] and BOX[2] <= my <= BOX[3]):
            continue
        parts = d["parts"] + [len(pts)]
        rings = [[[round(px - cx, 1), round(py - cy, 1)] for px, py in pts[parts[i]:parts[i + 1]]]
                 for i in range(len(parts) - 1)]
        h, src = height(d)
        name = label(d)
        out.append({
            "r": rings,
            "h": round(h, 1), "s": src,
            "f": num(d["A26"]),
            "n": name,
            "u": (d["A9"] or "").strip(),
            "p": 1 if public(d, name) else 0,
        })
    return out


def verify(bs):
    """빈 도형·음수 높이·범위 밖 좌표·깨진 도형이 있으면 빌드를 멈춘다."""
    assert bs, "잘라낸 건물이 0동"
    for b in bs:
        assert b["r"] and all(len(ring) >= 3 for ring in b["r"]), "꼭짓점 3개 미만 도형"
        assert b["h"] > 0, "높이 0 이하"
        assert Polygon(b["r"][0]).area > 0, "넓이 0 도형"
        for ring in b["r"]:
            for px, py in ring:
                assert GROUND[0] <= px <= GROUND[1] and GROUND[2] <= py <= GROUND[3], "지면 밖 좌표"
    shapes = [poly(b["r"]) for b in bs]
    tree = STRtree(shapes)
    for i, P in enumerate(shapes):                # 한 자리에 두 건물이 서면 안 된다
        for j in tree.query(P):
            assert j <= i or P.intersection(shapes[j]).area <= OVERLAP_M2, \
                f"건물 겹침 {P.intersection(shapes[j]).area:.1f}㎡ ({bs[i]['u']} / {bs[j]['u']})"
    return {
        "total": len(bs),
        "public": sum(b["p"] for b in bs),
        "src": [sum(1 for b in bs if b["s"] == k) for k in (0, 1, 2, 3)],
    }


def ngii(code, cx, cy):
    """두 도엽에서 레이어 하나를 읽어 지면 범위(GROUND)에 걸리는 것만 (로컬 좌표 부분들, 속성) 으로."""
    for sheet in SHEETS:
        path = os.path.join(NGII, sheet, code)
        if not os.path.exists(path + ".shp"):
            continue
        r = shapefile.Reader(path, encoding="euc_kr")
        for sr in r.iterShapeRecords():
            pts = sr.shape.points
            if not pts:
                continue
            xs = [p[0] - cx for p in pts]
            ys = [p[1] - cy for p in pts]
            if max(xs) < GROUND[0] or min(xs) > GROUND[1] or max(ys) < GROUND[2] or min(ys) > GROUND[3]:
                continue
            parts = list(getattr(sr.shape, "parts", None) or [0]) + [len(pts)]   # 점 도형은 parts 가 없다
            yield ([[[round(xs[k], 1), round(ys[k], 1)] for k in range(parts[i], parts[i + 1])]
                     for i in range(len(parts) - 1)], sr.record.as_dict())


def terrain(cx, cy):
    """등고선 꼭짓점 + 표고점 → 선형 보간 높이 격자. 볼록 껍질 밖은 가장 가까운 값."""
    xy, z = [], []
    for parts, d in ngii("N3L_F0010000", cx, cy):
        for part in parts:
            xy += part
            z += [d["등고수치"]] * len(part)
    n_contour = len(z)
    for parts, d in ngii("N3P_F0020000", cx, cy):
        xy.append(parts[0][0])
        z.append(d["수치"])
    xy, z = np.array(xy), np.array(z, dtype=float)
    gx = np.arange(GROUND[0], GROUND[1] + 1, CELL)
    gy = np.arange(GROUND[2], GROUND[3] + 1, CELL)
    X, Y = np.meshgrid(gx, gy)
    H = griddata(xy, z, (X, Y), method="linear")
    hole = np.isnan(H)
    H[hole] = griddata(xy, z, (X[hole], Y[hole]), method="nearest")
    z0 = float(np.floor(H.min()))
    grid = {"x0": GROUND[0], "y0": GROUND[2], "cell": CELL, "nx": len(gx), "ny": len(gy), "z0": z0,
            "h": [int(round((v - z0) * 10)) for v in H.ravel()]}     # 10cm 단위 정수, 행 = 남→북
    return grid, H - z0, (n_contour, len(z) - n_contour)


def sampler(H):
    """로컬 좌표 (x, y) → 지면 높이 (격자 쌍선형 보간)."""
    ny, nx = H.shape

    def at(x, y):
        fx = min(max((x - GROUND[0]) / CELL, 0), nx - 1.001)
        fy = min(max((y - GROUND[2]) / CELL, 0), ny - 1.001)
        i, j = int(fx), int(fy)
        a, b = fx - i, fy - j
        return ((H[j, i] * (1 - a) + H[j, i + 1] * a) * (1 - b)
                + (H[j + 1, i] * (1 - a) + H[j + 1, i + 1] * a) * b)
    return at


def fields(cx, cy):
    """운동장: 수치지도엔 면이 없고 위치 점(C038 '운동장') 과 둘레 스탠드(C039) 만 있다.
    점 주변 스탠드를 감싼 윤곽 = 운동장, 안쪽 띠 = 트랙, 그 안 = 필드. 스탠드가 둘러싸지 않은 운동장은 건너뛴다."""
    stands = [poly(p) for p, d in ngii("N3A_C0390000", cx, cy) if d["구조"] == "스탠드"]
    ring = lambda g: [[[round(x, 1), round(y, 1)] for x, y in r.coords] for r in [g.exterior] + list(g.interiors)]
    out, seen = [], []
    for p, d in ngii("N3P_C0380000", cx, cy):
        pt = Point(p[0][0])
        near = [s for s in stands if s.distance(pt) < 120]
        if d["구분"] != "운동장" or len(near) < 2:
            continue
        hull = unary_union(near).convex_hull
        if not hull.contains(pt) or any(hull.intersects(h) for h in seen):
            continue                                 # 같은 운동장에 점이 둘 찍힌 경우
        seen.append(hull)
        ground = hull.difference(unary_union(near).buffer(1)).buffer(-1)
        ground = max(getattr(ground, "geoms", [ground]), key=lambda g: g.area)
        pitch = ground.buffer(-7, join_style=1)
        out.append({"track": ring(ground), "pitch": ring(pitch)})
    return out


def ground_layers(cx, cy):
    """지면에 그릴 면·선, 세울 벽."""
    polys = lambda code: [parts for parts, _ in ngii(code, cx, cy)]
    lines = lambda code: [p for parts, _ in ngii(code, cx, cy) for p in parts]
    walls = [{"p": p, "h": d["높이"] if d["높이"] and d["높이"] > 0 else 1.5}
             for parts, d in ngii("N3L_F0040000", cx, cy) for p in parts]
    walls += [{"p": p, "h": 2.0 if d["구분"] == "담" else 2.5}
              for parts, d in ngii("N3L_B0020000", cx, cy) for p in parts]
    farm = lambda kind: [parts for parts, d in ngii("N3A_D0010000", cx, cy) if d["구분"] == kind]
    width = {"지방하천": 6, "소하천": 3}                   # 그 밖(세류) 1.5m
    return {
        "road": polys("N3A_A0010000"), "walk": polys("N3A_A0033320"),   # 도로면은 포장면으로만 칠한다 — 도로 자체는 local_roads 입체
        "stair": polys("N3A_C0390000"),                     # 계단·스탠드
        "sports": fields(cx, cy),
        "dry": farm("밭"), "paddy": farm("논"), "orchard": farm("과수원"),
        "river": polys("N3A_E0010001"), "surface": polys("N3A_E0032111"),   # 하천 구역(둔치 포함) · 수면
        "marsh": polys("N3A_G0020000"),
        "water": polys("N3A_E0052114"),
        "stream": [{"p": p, "w": width.get(d["구분"], 1.5)} for parts, d in ngii("N3L_E0020000", cx, cy) for p in parts],
        "bike": lines("N3L_A0033330"),
        "wall": walls,
    }


def poly(parts):
    p = Polygon(parts[0], parts[1:])
    return p if p.is_valid else p.buffer(0)


NGII_SKIP = ("무벽건물", "공사중건물", "가건물")   # 네이버 지도도 그리지 않는 차양·공사장·가설 건물


def merge_buildings(daejang, cx, cy):
    """건물 = 수치지도 2025(항공측량) 모양 + 건물 대장 속성.
    네이버 최신 지도와 대조한 결과(2026-09-29) 건물 위치·크기는 수치지도와 약 1m 안에서 같고(면적비 1.01),
    대장 모양은 평균 1.7m(최대 12m) 어긋나 있었다. 그래서 모양은 수치지도를 쓰고, 용도·층수·높이·이름은
    가장 많이 겹치는 대장 기록에서 가져온다(겹침 IoU MATCH_IOU 이상).
    수치지도에 없는 대장 기록은 용도가 있으면 측량 뒤 신축일 수 있어 대장 모양 그대로 남기고,
    용도가 비었거나 수치지도 건물 자리를 조금이라도 침범하면(OVERLAP_M2) 헐린·어긋난 옛 기록으로 보고 뺀다
    (네이버와 대조 시 용도 없는 기록의 58% 가 없었고, 학교 운동장의 옛 교사가 새 교사와 겹쳐 서는 일이 있었다).
    → (건물 목록, {모양: 수치지도 수, 속성 붙음 수, 대장만 남김 수, 옛 기록 뺌 수})"""
    shapes = [poly(b["r"]) for b in daejang]
    used, out, n_attr = set(), [], 0
    for parts, d in ngii("N3A_B0010000", cx, cy):
        if d["종류"] in NGII_SKIP or len(parts[0]) < 3:
            continue
        xs, ys = [p[0] for p in parts[0]], [p[1] for p in parts[0]]
        if not (BOX[0] <= (min(xs) + max(xs)) / 2 <= BOX[1] and BOX[2] <= (min(ys) + max(ys)) / 2 <= BOX[3]):
            continue                             # 대장 건물과 같은 범위 규칙 (extract)
        P = poly(parts)
        if P.is_empty:
            continue
        hits = [(P.intersection(S).area / P.union(S).area, k) for k, S in enumerate(shapes)
                if S.intersects(P)]
        hits = [(iou, k) for iou, k in hits if iou >= MATCH_IOU]
        floors = d["층수"] if d["층수"] and d["층수"] > 0 else 0
        if hits:                                 # 대장 속성을 붙인다 — 이름은 겹치는 기록 중 공공시설 이름이 있으면 그것
            b = daejang[max(hits)[1]]
            used.update(k for _, k in hits)
            n_attr += 1
            name = next((daejang[k]["n"] for _, k in sorted(hits, reverse=True) if daejang[k]["n"]), "")
            h, s, fl = b["h"], b["s"], b["f"]
            if s == 2 and floors:                # 대장에 높이·층수가 없으면 수치지도 층수
                h, s, fl = floors * FLOOR_H, 3, floors
            out.append({**b, "r": parts, "h": round(h, 1), "s": s, "f": fl, "n": name,
                        "p": 1 if name or b["p"] else 0, "g": 0})
        else:
            use = (d["용도"] or d["종류"] or "").strip()
            fl = floors or 1
            out.append({"r": parts, "h": round(fl * FLOOR_H, 1), "s": 3, "f": fl, "n": "", "u": use,
                        "p": 1 if re.search(PUBLIC_USE, use) else 0, "g": 0})
    n_ngii = len(out)
    now = unary_union([poly(b["r"]) for b in out])
    kept = gone = 0
    for k, b in enumerate(daejang):
        if k in used:
            continue
        if not b["u"] or shapes[k].intersection(now).area > OVERLAP_M2:
            gone += 1                            # 용도 없는 옛 기록, 또는 항공측량에 건물이 있는 자리를 침범하는 기록(한 자리에 두 건물은 없다)
            continue
        out.append({**b, "g": 1})
        kept += 1
    return out, {"ngii": n_ngii, "attr": n_attr, "daejang_only": kept, "gone": gone}


def seat(bs, at):
    """건물 바닥을 지면에 앉힌다: 바닥 = 꼭짓점 중 가장 낮은 지면, 비탈만큼 벽을 더 내린다."""
    for b in bs:
        g = [at(x, y) for x, y in b["r"][0]]
        b["z"] = round(min(g), 2)
        b["e"] = round(max(g) - min(g), 2)


def bridges(cx, cy, at):
    """교량(A007)을 지면에서 띄운다. 수치지도 등고선은 다리 밑 골짜기 지면을 따르므로 도로를 지면에 칠하면
    고속도로가 논바닥에 붙는다. 상판 높이 = 다리 양 끝(교대) 바로 바깥 지면 높이를 잇는 직선.
    다리 축 = 최소 외접 직사각형의 긴 변 방향. 짧은 암거형 다리(BRIDGE_MIN m 미만)는 지면 그림으로 둔다."""
    pieces = [(poly(parts), d) for parts, d in ngii("N3A_A0070000", cx, cy) if (d["연장"] or 0) >= BRIDGE_MIN]
    merged = []                                  # 도엽 경계에서 잘린 같은 이름 다리 조각을 하나로
    for P, d in pieces:
        name = (d["명칭"] or "").strip()
        hit = next((m for m in merged if name and m[1] == name and m[0].distance(P) < 1), None)
        if hit:
            hit[0] = unary_union([hit[0], P.buffer(0.5)]).buffer(-0.5)
        else:
            merged.append([P, name])
    area = Polygon([(GROUND[0], GROUND[2]), (GROUND[1], GROUND[2]), (GROUND[1], GROUND[3]), (GROUND[0], GROUND[3])])
    out = []
    for P, name in merged:
        if not area.buffer(BRIDGE_PAD * 3).contains(P):
            continue                             # 지면 밖으로 멀리 나가는 다리는 교대 높이를 잴 수 없다
        rect = list(P.minimum_rotated_rectangle.exterior.coords)[:4]
        e = sorted([(rect[i], rect[(i + 1) % 4]) for i in range(4)], key=lambda s: -math.dist(*s))
        mid = lambda s: ((s[0][0] + s[1][0]) / 2, (s[0][1] + s[1][1]) / 2)
        # 긴 변 둘의 끝끼리 짝지어 짧은 변(= 다리 양 끝)의 가운데를 구한다
        ends = sorted([s for s in [(rect[i], rect[(i + 1) % 4]) for i in range(4)] if s not in e[:2]],
                      key=lambda s: mid(s))
        a, b = mid(ends[0]), mid(ends[1])
        L = math.dist(a, b)
        ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
        za = at(a[0] - ux * BRIDGE_PAD, a[1] - uy * BRIDGE_PAD)
        zb = at(b[0] + ux * BRIDGE_PAD, b[1] + uy * BRIDGE_PAD)
        out.append({"p": [[[round(x, 1), round(y, 1)] for x, y in P.intersection(area).exterior.coords]], "a": [round(v, 1) for v in a], "b": [round(v, 1) for v in b],
                    "za": round(float(za), 2), "zb": round(float(zb), 2),
                    "w": round(math.dist(*ends[0]), 1), "n": name})
    return out


def chain(segs):
    """끝점이 맞닿은 선 조각들을 이어 하나의 선으로. 끝점(차수 1)이 정확히 둘이어야 한다."""
    key = lambda p: (round(p[0]), round(p[1]))
    deg = {}
    for s in segs:
        for p in (s[0], s[-1]):
            deg[key(p)] = deg.get(key(p), 0) + 1
    ends = [k for k, v in deg.items() if v == 1]
    assert len(ends) == 2, f"고속도로 본선이 한 줄로 이어지지 않음 (끝점 {len(ends)}개)"
    left, line, cur = list(segs), [], ends[0]
    while left:
        s = next(s for s in left if key(s[0]) == cur or key(s[-1]) == cur)
        left.remove(s)
        s = s if key(s[0]) == cur else s[::-1]
        line += s if not line else s[1:]
        cur = key(s[-1])
    return line


def densify(line, ws, step):
    """선을 step m 간격 점으로. ws = 각 꼭짓점의 폭. → (N×2 좌표, 누적거리, 폭)"""
    L = np.r_[0, np.cumsum([math.dist(a, b) for a, b in zip(line, line[1:])])]
    S = np.linspace(0, L[-1], max(2, int(L[-1] // step) + 1))
    xy = np.c_[np.interp(S, L, [p[0] for p in line]), np.interp(S, L, [p[1] for p in line])]
    return xy, S, np.interp(S, L, ws)


def smooth(S, raw, wt, sigma):
    """거리 S 위의 값 raw 를 가중치 wt 로 가우시안 평활."""
    K = np.exp(-((S[:, None] - S[None, :]) / sigma) ** 2 / 2) * wt[None, :]
    return (K @ raw) / K.sum(1)


def expressway(cx, cy, at, brs):
    """고속국도(A002 도로구분)를 지면에서 띄운 연속 상판으로. 본선(폭 HW_MAIN m 이상)은 한 줄로 이어 높이를 정하고,
    램프·진입로는 본선과 만나는 끝 = 본선 높이, 일반 도로와 만나는 끝 = 지면 높이로 맞춰 경사지게 잇는다.
    높이 원천: 등고선이 성토(둑) 높이를 담고 있어 중심선 아래 지형 = 도로면. 높은 다리 구간은 교량 상판 직선.
    → (선 목록 [{p: [[x, y, z, 폭]…], m: 본선 여부}], 이 선들에 속한 교량 목록)"""
    main, other = [], []
    for parts, d in ngii("N3L_A0020000", cx, cy):
        if d["도로구분"] != "고속국도":
            continue
        (main if (d["도로폭"] or 0) >= HW_MAIN else other).append((parts[0], d["도로폭"] or 7))
    high = [br for br in brs if lift(br, at) >= BRIDGE_HIGH]

    def raw(xy):
        z, w = np.array([at(x, y) for x, y in xy]), np.ones(len(xy))
        for br in high:
            P = Polygon(br["p"][0]).buffer(3)
            for k, (x, y) in enumerate(xy):
                if P.contains(Point(x, y)):
                    z[k], w[k] = deck(br, x, y), 20
        return z, w

    line = chain([s for s, _ in main])
    wmap = {(round(p[0]), round(p[1])): w for s, w in main for p in s}
    ws = [wmap.get((round(p[0]), round(p[1])), HW_MAIN) for p in line]
    mxy, mS, mw = densify(line, ws, HW_STEP)
    mz = smooth(mS, *raw(mxy), HW_SIGMA)
    out = [{"p": np.c_[mxy, mz, mw, np.full(len(mS), BRIDGE_CLEAR)], "m": 1}]   # 열: x, y, 상판 높이, 폭, 밑 비울 깊이
    main_at = lambda x, y: mz[np.argmin(np.hypot(mxy[:, 0] - x, mxy[:, 1] - y))]
    on_main = lambda x, y: np.hypot(mxy[:, 0] - x, mxy[:, 1] - y).min() < 2

    # 램프·진입로: 차수 2 인 점에서 이어 붙여 사슬로, 사슬 끝은 본선 위 · 막다른 끝(지면) · 램프끼리 갈림점
    key = lambda p: (round(p[0]), round(p[1]))
    deg = {}
    for s, _ in other:
        for p in (s[0], s[-1]):
            deg[key(p)] = deg.get(key(p), 0) + 1
    stop = lambda p: deg[key(p)] != 2 or on_main(*p)
    left, chains = list(other), []
    while left:
        s, w = left.pop(0)
        c, cw = list(s), [w] * len(s)
        for end in (0, 1):                       # 양쪽으로 늘린다
            while not stop(c[-1] if end else c[0]):
                tip = key(c[-1] if end else c[0])
                nxt = next(((t, tw) for t, tw in left if key(t[0]) == tip or key(t[-1]) == tip), None)
                if nxt is None:
                    break
                left.remove(nxt)
                t, tw = nxt
                if end:
                    t = t if key(t[0]) == tip else t[::-1]
                    c, cw = c + t[1:], cw + [tw] * (len(t) - 1)
                else:
                    t = t if key(t[-1]) == tip else t[::-1]
                    c, cw = t[:-1] + c, [tw] * (len(t) - 1) + cw
        chains.append((c, cw))
    ramps = []
    for c, cw in chains:
        xy, S, w = densify(c, cw, HW_STEP)
        z = smooth(S, *raw(xy), HW_SIGMA / 2) if S[-1] > HW_STEP else raw(xy)[0]
        ramps.append([xy, S, w, z])
    ends = {}                                    # 갈림점 높이 = 거기서 끝나는 램프들 평균
    for xy, S, w, z in ramps:
        for k in (0, -1):
            ends.setdefault(key(xy[k]), []).append(z[k])
    for r in ramps:
        xy, S, w, z = r
        tgt, clr = [], np.full(len(S), RAMP_CLEAR)
        for k, far in ((0, S), (-1, S[-1] - S)):
            x, y = xy[k]
            if on_main(x, y):
                tgt.append(main_at(x, y))
            elif deg.get(key(xy[k]), 1) == 1:
                tgt.append(at(x, y))             # 막다른 끝 = 일반 도로와 만나는 지면
                clr = np.minimum(clr, RAMP_CLEAR * np.minimum(1, far / RAMP_BLEND))   # 끝에선 상판이 지면에 얹힌다
            else:
                tgt.append(np.mean(ends[key(xy[k])]))
        t = S / S[-1] if S[-1] > 0 else np.zeros(len(S))
        r[3] = z + (tgt[0] - z[0]) * (1 - t) + (tgt[1] - z[-1]) * t
        out.append({"p": np.c_[xy, r[3], w, clr], "m": 0})
    foot = unary_union([LineString(c["p"][:, :2]).buffer(float(c["p"][:, 3].max()) / 2 + 2, cap_style=2)
                        for c in out if len(c["p"]) > 1])
    mine = [br for br in brs if Polygon(br["p"][0]).intersection(foot).area > 0.3 * Polygon(br["p"][0]).area]
    inside = lambda p: p[(p[:, 0] >= GROUND[0]) & (p[:, 0] <= GROUND[1]) & (p[:, 1] >= GROUND[2]) & (p[:, 1] <= GROUND[3])]
    out = [{**c, "p": inside(c["p"])} for c in out]
    return [c for c in out if len(c["p"]) > 1], foot, mine


def lower_under(H, grid, hw, at0):
    """고속도로 밑 성토를 걷어 낸다: 선에 직각으로 양옆 (폭/2 + 여유) 지점의 지면 높이를 직선으로 이은 값보다
    높으면 낮춘다(자연 지면 추정). 상판 밑은 본선 BRIDGE_CLEAR · 램프 RAMP_CLEAR m 아래까지 비운다(깎기 구간 포함).
    밑을 지나는 일반 도로·하천이 끊기지 않게 하려는 것. 지형은 낮추기만 하고 올리지 않는다."""
    ny, nx = H.shape
    n = 0
    for c in hw:
        p = c["p"]
        if len(p) < 3:
            continue
        E = HW_SIDE if c["m"] else HW_SIDE / 3
        t = np.gradient(p[:, :2], axis=0)
        t /= np.linalg.norm(t, axis=1)[:, None]
        nrm = np.c_[-t[:, 1], t[:, 0]]
        for (x, y, z, w, clear), (ux, uy) in zip(p, nrm):
            R = w / 2 + E
            zl, zr = at0(x - ux * R, y - uy * R), at0(x + ux * R, y + uy * R)
            for o in np.arange(-R, R + 0.1, CELL / 2):
                cap = zl + (zr - zl) * (o + R) / (2 * R)
                if abs(o) <= w / 2 + CELL:           # 상판 밑은 확실히 비운다 (본선은 높이 띄우고 램프는 상판 바로 아래까지)
                    cap = min(cap, z - clear)
                i = int(round((x + ux * o - GROUND[0]) / CELL))
                j = int(round((y + uy * o - GROUND[2]) / CELL))
                if 0 <= i < nx and 0 <= j < ny and H[j, i] > cap:
                    H[j, i] = cap
                    n += 1
    grid["h"] = [int(round(v * 10)) for v in H.ravel()]
    return n


def local_roads(cx, cy, at, brs):
    """일반 도로(고속국도 밖 A002 중심선)를 도로폭 띠의 입체로. 도로는 사람이 높이를 맞춘 구조물이라
    10m 지형 격자로는 노면을 나타낼 수 없다 — 지면에 칠하면 둑길이 하천 쪽 비탈로 흘러내려 잠겨 보인다.
    노면 높이 = 폭 방향 세 점(좌·중·우) 지형 중 가장 높은 곳 + ROAD_LIFT, 다리 위는 교량 상판 높이.
    교차점은 만나는 도로 중 가장 높은 끝에 맞추고 ROAD_BLEND m 동안 서서히 잇는다.
    → [{p: [[x, y, 노면 높이, 폭, 옆면 깊이]…]}]"""
    key = lambda p: (round(p[0] * 2), round(p[1] * 2))
    segs = []
    for parts, d in ngii("N3L_A0020000", cx, cy):
        if d["도로구분"] == "고속국도":
            continue
        w = d["도로폭"] if d["도로폭"] and d["도로폭"] > 0 else ROAD_W0
        segs += [(p, w) for p in parts if len(p) >= 2]
    deg = {}
    for p, _ in segs:
        for q in (p[0], p[-1]):
            deg[key(q)] = deg.get(key(q), 0) + 1
    bp = [(Polygon(br["p"][0]).buffer(2), br) for br in brs]
    lines = []
    for p, w in segs:
        xy, S, _ = densify(p, [w] * len(p), ROAD_STEP)
        if S[-1] < 1:
            continue
        t = np.gradient(xy, axis=0)
        t /= np.maximum(np.linalg.norm(t, axis=1), 1e-9)[:, None]
        n = np.c_[-t[:, 1], t[:, 0]] * (w / 2)
        side = np.array([[at(*(c + o)) for o in (-n[k], 0 * n[k], n[k])] for k, c in enumerate(xy)])
        ground = side.max(1) + ROAD_LIFT
        raw, wt, on = ground.copy(), np.ones(len(S)), np.zeros(len(S), bool)
        for P, br in bp:
            for k, (x, y) in enumerate(xy):
                if P.contains(Point(x, y)):
                    raw[k], wt[k], on[k] = deck(br, x, y) + ROAD_LIFT, 20, True
        z = smooth(S, raw, wt, ROAD_SIGMA) if len(S) > 2 else raw
        z = np.where(on, raw, np.maximum(z, ground))
        lines.append([xy, S, w, z, side.min(1), t, (key(p[0]), key(p[-1]))])
    node = {}
    for xy, S, w, z, low, t, ends in lines:
        for k, e in ((0, ends[0]), (-1, ends[1])):
            node[e] = max(node.get(e, -1e9), z[k])
    out = []
    for xy, S, w, z, low, t, (e0, e1) in lines:
        z = (z + (node[e0] - z[0]) * np.clip(1 - S / ROAD_BLEND, 0, 1)
               + (node[e1] - z[-1]) * np.clip(1 - (S[-1] - S) / ROAD_BLEND, 0, 1))
        pts = np.c_[xy, z, np.full(len(S), w), z - low + 0.3]
        ext = min(w / 2, 3)                      # 교차점에서는 끝을 조금 늘려 꺾인 모서리 틈을 메운다
        if deg[e0] > 1:
            pts = np.r_[[[*(xy[0] - t[0] * ext), z[0], w, pts[0, 4]]], pts]
        if deg[e1] > 1:
            pts = np.r_[pts, [[*(xy[-1] + t[-1] * ext), z[-1], w, pts[-1, 4]]]]
        pts = pts[(pts[:, 0] >= GROUND[0]) & (pts[:, 0] <= GROUND[1]) & (pts[:, 1] >= GROUND[2]) & (pts[:, 1] <= GROUND[3])]
        if len(pts) > 1:
            out.append({"p": np.round(pts, 2).tolist()})
    return out


def deck(br, x, y):
    """교량 상판 높이: 다리 축 위 위치 비율로 양 끝(교대) 높이를 잇는다."""
    (ax, ay), (bx, by) = br["a"], br["b"]
    t = min(1, max(0, ((x - ax) * (bx - ax) + (y - ay) * (by - ay)) / ((bx - ax) ** 2 + (by - ay) ** 2)))
    return br["za"] + (br["zb"] - br["za"]) * t


def lift(br, at):
    """다리 축을 따라 상판이 지면보다 가장 높이 뜬 값(m)."""
    (ax, ay), (bx, by) = br["a"], br["b"]
    L = math.dist(br["a"], br["b"])
    return max(deck(br, x, y) - at(x, y) for x, y in
               ((ax + (bx - ax) * s / L, ay + (by - ay) * s / L) for s in np.arange(0, L, CELL)))


def carve(H, grid, brs):
    """다리 밑 지형을 상판보다 BRIDGE_CLEAR m 아래로 깎는다. 등고선 보간은 다리 밑 능선 끝자락을 그대로 이어
    상판을 뚫고 올라오는 일이 있는데, 다리가 지나는 자리의 땅이 상판보다 높을 수는 없다."""
    n = 0
    ny, nx = H.shape
    at = sampler(H)
    for br in brs:                               # 낮은 다리도 깎는다: 입체 도로가 상판 위로 지나가므로 밑 하천 골짜기가 드러나야 한다
        clear = BRIDGE_CLEAR if lift(br, at) >= BRIDGE_HIGH else RAMP_CLEAR
        P = Polygon(br["p"][0]).buffer(CELL)
        x0, y0, x1, y1 = P.bounds
        for j in range(max(0, int((y0 - GROUND[2]) // CELL)), min(ny, int((y1 - GROUND[2]) // CELL) + 2)):
            for i in range(max(0, int((x0 - GROUND[0]) // CELL)), min(nx, int((x1 - GROUND[0]) // CELL) + 2)):
                x, y = GROUND[0] + i * CELL, GROUND[2] + j * CELL
                top = deck(br, x, y) - clear
                if P.contains(Point(x, y)) and H[j, i] > top:
                    H[j, i] = top
                    n += 1
    grid["h"] = [int(round(v * 10)) for v in H.ravel()]
    return n


def build():
    rows, cx, cy = load()
    bs, merged = merge_buildings(extract(rows, cx, cy), cx, cy)
    stats = verify(bs)
    stats["merge"] = merged
    grid, H, n_pts = terrain(cx, cy)
    at = sampler(H)
    brs = bridges(cx, cy, at)                    # 교대 높이·고속도로 높이는 깎기 전 지형에서 잰다
    hw, foot, hw_brs = expressway(cx, cy, at, brs)
    brs = [br for br in brs if br not in hw_brs]  # 고속도로 교량은 고속도로 상판으로 그린다
    at0 = sampler(H.copy())
    stats["carved"] = carve(H, grid, brs) + lower_under(H, grid, hw, at0)
    seat(bs, at)                                 # at 은 H 를 참조하므로 깎은 지형 기준
    layers = ground_layers(cx, cy)
    layers["bridge"] = brs
    layers["hw"] = [{"p": np.round(c["p"][:, :4], 1).tolist(), "m": c["m"]} for c in hw]
    layers["hwfoot"] = [[[round(x, 1), round(y, 1)] for x, y in r.coords]
                        for g in getattr(foot, "geoms", [foot]) for r in [g.exterior, *g.interiors]]
    layers["roads"] = local_roads(cx, cy, at, brs)
    assert layers["road"], "도로면이 0개 — data/raw/ngii 에 도엽이 풀려 있는지 확인하세요"
    stats["ground"] = {"z0": grid["z0"], "rise": round(float(H.max()), 1), "pts": n_pts,
                       **{k: len(v) for k, v in layers.items()}}
    html = open(os.path.join(HERE, "template.html"), encoding="utf-8").read()
    html = (html.replace("/*__TOKENS__*/", ROOT_TOKENS)
                .replace("__GROUND__", json.dumps({"grid": grid, **layers}, separators=(",", ":")))
                .replace("__DATA__", json.dumps(bs, ensure_ascii=False, separators=(",", ":")))
                .replace("__STATS__", json.dumps(stats))
                .replace("__SNAPSHOT__", SNAPSHOT))
    assert "__DATA__" not in html and "__STATS__" not in html and "__GROUND__" not in html
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    g = stats["ground"]
    print(OUT)
    mg = stats["merge"]
    print(f"건물 {stats['total']}동 (공공·마을 {stats['public']}) · 모양: 수치지도 {mg['ngii']} (대장 속성 붙음 {mg['attr']}) + "
          f"대장만 {mg['daejang_only']} · 옛 기록 제외 {mg['gone']} · 높이 원본 {stats['src'][0]} / 층수 추정 {stats['src'][1]} / "
          f"기본값 {stats['src'][2]} / 수치지도 층수 {stats['src'][3]} · {os.path.getsize(OUT) // 1024}KB")
    print(f"지형 해발 {g['z0']}~{g['z0'] + g['rise']}m (등고선 점 {g['pts'][0]} · 표고점 {g['pts'][1]}) · "
          f"입체 도로 {g['roads']}줄 {sum(math.dist(a[:2], b[:2]) for r in layers['roads'] for a, b in zip(r['p'], r['p'][1:])) / 1000:.1f}km · 포장면 {g['road']} · 보도 {g['walk']} · "
          f"옹벽·담장 {g['wall']} · 교량 {g['bridge']} · 고속도로 {g['hw']}줄 · 지형 {stats['carved']}칸 낮춤 · 하천 {g['river']} · 저수지 {g['water']} · 논 {g['paddy']} · 밭 {g['dry']}")


if __name__ == "__main__":
    build()
