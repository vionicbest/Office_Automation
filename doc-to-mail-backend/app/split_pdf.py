import fitz
from PIL import Image
import numpy as np
import cv2
from collections import defaultdict
import os
import json

pdf_path = "temp.pdf"

def find_horizontal_lines(img: Image.Image,
                          black_thresh: int = 220,
                          row_fill_ratio: float = 0.5):
    """
    픽셀 기반 수평선 탐지 함수
    """
    gray = np.array(img.convert("L"))
    mask = gray < black_thresh
    h, w = mask.shape
    filled = np.sum(mask, axis=1) / w
    rows = np.where(filled > row_fill_ratio)[0]
    if not len(rows):
        return []
    lines, start, prev = [], rows[0], rows[0]
    for y in rows[1:]:
        if y == prev + 1:
            prev = y
        else:
            lines.append((start, prev))
            start = prev = y
    lines.append((start, prev))
    return lines

def get_top_line_y_from_pdf(doc: fitz.Document, dpi: int = 300) -> int:
    """
    PDF에서 가장 위에 있는 수평선의 y좌표를 반환
    """
    page = doc[0]
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    lines = find_horizontal_lines(img)
    if not lines:
        raise ValueError("수평선을 찾지 못했습니다.")
    
    top_y = lines[0][0]
    print(f"가장 위 수평선 y좌표: {top_y}")
    return img, top_y

def find_last_blank_block_between(img: Image.Image,
                                   start_y: int,
                                   end_y: int,
                                   white_thresh: int = 240,
                                   blank_ratio: float = 0.999,
                                   min_height_px: int = 5) -> tuple[int, int] | None:
    """
    start_y ~ end_y 사이에서 공백 블록을 찾되,
    블록 높이가 min_height_px 이상인 것 중 가장 아래에 있는 것 반환
    """
    gray = np.array(img.convert("L"))
    h, w = gray.shape
    end_y = min(end_y, h)

    region = gray[start_y:end_y]
    is_blank_row = (np.sum(region > white_thresh, axis=1) / w) > blank_ratio
    blank_rows = np.where(is_blank_row)[0]

    if not len(blank_rows):
        print("공백 라인 없음")
        return None

    # 연속된 공백 블록 구하기
    blocks = []
    block_start = block_end = blank_rows[0]
    for y in blank_rows[1:]:
        if y == block_end + 1:
            block_end = y
        else:
            if (block_end - block_start + 1) >= min_height_px:
                blocks.append((block_start, block_end))
            block_start = block_end = y
    if (block_end - block_start + 1) >= min_height_px:
        blocks.append((block_start, block_end))

    if not blocks:
        print("min_height_px 이상인 공백 블록 없음")
        return None

    # 마지막 블록 선택
    last_start, last_end = blocks[-2]
    abs_start = start_y + last_start
    abs_end = start_y + last_end
    print(f"마지막 공백 블록: y = {abs_start} ~ {abs_end}")
    return abs_start, abs_end

def test_title_crop_from_pdf(doc: fitz.Document, output_path: str = "title_final_crop.png"):
    """
    PDF에서 가장 위 수평선(top_line_y)을 기준으로,
    그 위쪽에서 가장 아래 공백 줄(last_blank_y)을 찾아
    제목 영역만 정확히 크롭하여 저장
    """
    # 1. PDF 렌더링 및 수평선 y좌표 추출
    img, top_line_y = get_top_line_y_from_pdf(doc)

    # 2. 공백 줄 중 가장 아래 y좌표 찾기
    _, last_blank_y = find_last_blank_block_between(img, 0, top_line_y)

    # 3. 제목 크롭 (공백 라인 ~ 수평선 직전까지)
    cropped = img.crop((0, last_blank_y, img.width, top_line_y))
    cropped.save(output_path)
    print(f"[✓] 제목 이미지 저장 완료: {output_path}")

    return last_blank_y, top_line_y

def find_body_range_in_pdf(doc: fitz.Document,
                           min_blank_height_px: int = 300,
                           max_search_height: int = 8000) -> tuple[int, int] | None:
    """
    PDF 전체에서 본문 시작 (top_line_y from page 0) ~ 본문 끝 (처음 나타나는 큰 공백) 위치를 찾음
    반환값: (end_page_idx, y좌표), 즉 본문이 해당 페이지 y좌표까지 포함됨
    """
    import fitz
    from PIL import Image
    import numpy as np

    def find_first_sufficient_blank_block_below(
        img: Image.Image,
        start_y: int,
        max_search_height: int = 8000,
        white_thresh: int = 240,
        blank_ratio: float = 0.95,
        min_height_px: int = 100
    ) -> tuple[int, int] | None:
        """
        start_y 이후에서 충분히 넓은 공백 한 개만 찾아 반환
        """
        gray = np.array(img.convert("L"))
        h, w = gray.shape
        end_y = min(start_y + max_search_height, h)
        region = gray[start_y:end_y]

        is_blank_row = (np.sum(region > white_thresh, axis=1) / w) > blank_ratio
        blank_rows = np.where(is_blank_row)[0]
        if not len(blank_rows): return None

        block_start = block_end = blank_rows[0]
        for y in blank_rows[1:]:
            if y == block_end + 1:
                block_end = y
            else:
                if (block_end - block_start + 1) >= min_height_px:
                    return start_y + block_start, start_y + block_end
                block_start = block_end = y

        # 마지막 블록 검사
        if (block_end - block_start + 1) >= min_height_px:
            return start_y + block_start, start_y + block_end

        return None

    top_line_y = None

    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        if i == 0:
            # 첫 페이지에서 시작 지점 찾기
            _, top_line_y = get_top_line_y_from_pdf(doc)
            start_y = top_line_y
        else:
            start_y = 0  # 이후 페이지는 맨 위부터 탐색

        blank_block = find_first_sufficient_blank_block_below(
            img,
            start_y=start_y,
            max_search_height=max_search_height,
            min_height_px=min_blank_height_px
        )

        if blank_block:
            print(f"[✓] 본문 종료 지점 탐지됨: 페이지 {i}, y = {blank_block[0]}")
            return (i, blank_block[0])

    print("[!] 본문 종료 공백을 찾지 못함")
    return None

def stitch_body_region(doc: fitz.Document,
                       start_y: int,
                       end_page: int,
                       end_y: int,
                       dpi: int = 300) -> Image.Image:
    """
    주어진 start_y (page 0) ~ (end_page, end_y) 범위 본문을 이미지로 추출하여 이어붙임
    """
    images_to_stitch = []

    for i in range(end_page + 1):
        pix = doc[i].get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
        if i == 0:
            # 첫 페이지는 start_y부터 끝까지
            cropped = img.crop((0, start_y, img.width, img.height))
        elif i == end_page:
            # 마지막 페이지는 0부터 end_y까지
            cropped = img.crop((0, 0, img.width, end_y))
        else:
            # 중간 페이지는 전부
            cropped = img

        images_to_stitch.append(cropped)

    # 스티칭
    total_height = sum(img.height for img in images_to_stitch)
    final_img = Image.new("RGB", (images_to_stitch[0].width, total_height), (255, 255, 255))
    y_offset = 0
    for img in images_to_stitch:
        final_img.paste(img, (0, y_offset))
        y_offset += img.height

    return final_img

def split_by_smart_blank_lines(img: Image.Image,
                               white_thresh: int = 240,
                               blank_ratio: float = 0.999,
                               min_blank_group_height: int = 10) -> list[tuple[int, int]]:
    """
    박스 내부 공백은 무시하고, 진짜 분리 목적의 공백만 블록 나누기
    """
    gray = np.array(img.convert("L"))
    h, w = gray.shape
    is_blank = (np.sum(gray > white_thresh, axis=1) / w) > blank_ratio
    blank_rows = np.where(is_blank)[0]

    blocks = []
    prev_y = 0
    if len(blank_rows) == 0:
        return [(0, h)]

    # 공백 그룹으로 나누기
    group_start = group_end = blank_rows[0]
    for y in blank_rows[1:]:
        if y == group_end + 1:
            group_end = y
        else:
            if (group_end - group_start + 1) >= min_blank_group_height:
                blocks.append((prev_y, group_start))
                prev_y = group_end + 1
            group_start = group_end = y

    # 마지막 공백 그룹 처리
    if (group_end - group_start + 1) >= min_blank_group_height:
        blocks.append((prev_y, group_start))
        prev_y = group_end + 1

    if prev_y < h:
        blocks.append((prev_y, h))

    return blocks

def merge_blocks_with_tag(blocks: list[tuple[int, int]], height_thresh: int = 70) -> list[dict]:
    """
    블록 리스트에서 height 기준으로 텍스트를 판별.
    연속된 텍스트 블록은 병합하고, non-text는 그대로 유지.
    결과는 {"y0": .., "y1": .., "type": "text"/"nontext"} 형식.
    """
    tagged_blocks = []
    current_start, current_end = None, None

    for y0, y1 in blocks:
        height = y1 - y0
        if height < height_thresh:
            if current_start is None:
                current_start, current_end = y0, y1
            else:
                current_end = y1
        else:
            # 현재 병합 중인 텍스트 블록이 있다면 먼저 저장
            if current_start is not None:
                tagged_blocks.append({"y0": current_start, "y1": current_end, "type": "text"})
                current_start, current_end = None, None
            # 그 다음 이건 nontext로 따로 저장
            tagged_blocks.append({"y0": y0, "y1": y1, "type": "nontext"})

    # 마지막 텍스트 블록 처리
    if current_start is not None:
        tagged_blocks.append({"y0": current_start, "y1": current_end, "type": "text"})

    return tagged_blocks

def mapping_blocks(blocks, start_y, doc):
    
    dpi = 300
    scale = dpi / 72
    mat = fitz.Matrix(scale, scale)
    pix = doc[0].get_pixmap(matrix=mat)

    page_height = pix.height
    print(page_height)

    result = []

    for b in blocks:
        y0 = b['y0'] + start_y
        y1 = b['y1'] + start_y

        result.append({
            'page_s': 1 + y0 // page_height, 'y0': y0 % page_height,
            'page_e': 1 + y1 // page_height, 'y1': y1 % page_height,
            'type': b['type']
        })
    
    return result

def crop_pdf_blocks_with_fitz(doc, blocks, output_dir, dpi=300):
    """
    PDF에서 PyMuPDF(fitz)를 사용해 페이지를 이미지로 렌더링한 뒤,
    지정된 y0, y1 좌표 구간을 기준으로 블록 단위로 잘라내어 저장합니다.
    
    Parameters:
    - pdf_path: 입력 PDF 파일 경로
    - blocks: 크롭 블록 정보 리스트. 각 요소는 dict로, keys: page_s, page_e, y0, y1
      * page_s, page_e: 1-based 페이지 번호
      * y0, y1: 상단 y좌표, 하단 y좌표 (픽셀 단위)
    - output_dir: 잘라낸 이미지를 저장할 폴더 경로
    - dpi: 렌더링 해상도 (기본 300dpi)
    """
    os.makedirs(output_dir, exist_ok=True)
    page_count = doc.page_count
    print(f"[DEBUG] 문서 전체 페이지 수: {page_count}")
    zoom = dpi / 72  # 72 DPI가 기본이므로 비율 계산
    mat = fitz.Matrix(zoom, zoom)
    
    for idx, blk in enumerate(blocks, start=1):
        ps = int(blk['page_s']) - 1
        pe = int(blk['page_e']) - 1
        y0, y1 = blk['y0'], blk['y1']

        # 단일 페이지
        if ps == pe:
            page = doc.load_page(ps)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            cropped = img.crop((0, y0, pix.width, y1))
        # 여러 페이지에 걸친 블록
        else:
            # 시작 페이지 크롭
            page1 = doc.load_page(ps)
            pix1 = page1.get_pixmap(matrix=mat, alpha=False)
            img1 = Image.frombytes("RGB", (pix1.width, pix1.height), pix1.samples)
            part1 = img1.crop((0, y0, pix1.width, pix1.height))
            
            # 종료 페이지 크롭
            page2 = doc.load_page(pe)
            pix2 = page2.get_pixmap(matrix=mat, alpha=False)
            img2 = Image.frombytes("RGB", (pix2.width, pix2.height), pix2.samples)
            part2 = img2.crop((0, 0, pix2.width, y1))
            
            # 두 이미지를 세로로 병합
            w = pix1.width
            merged = Image.new("RGB", (w, part1.height + part2.height))
            merged.paste(part1, (0, 0))
            merged.paste(part2, (0, part1.height))
            cropped = merged
        
        out_path = os.path.join(output_dir, f"block_{idx}_p{blk['page_s']}_to_p{blk['page_e']}.png")
        cropped.save(out_path)
        print(f"Saved: {out_path}")
        
def segments_intersect(a: dict, b: dict) -> bool:
    # 점을 튜플로 변환
    p0 = (a["x0"], a["y0"])
    p1 = (a["x1"], a["y1"])
    p2 = (b["x0"], b["y0"])
    p3 = (b["x1"], b["y1"])

    # 방향(orientation) 계산: p→q와 p→r의 외적
    def orientation(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    # q가 p와 r 사이에 놓여 있는지 확인
    def on_segment(p, q, r):
        return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

    o1 = orientation(p0, p1, p2)
    o2 = orientation(p0, p1, p3)
    o3 = orientation(p2, p3, p0)
    o4 = orientation(p2, p3, p1)

    # 일반 교차 판정
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True

    # 특수 케이스: 한 점이 다른 선분 위에 놓인 경우
    if o1 == 0 and on_segment(p0, p2, p1):
        return True
    if o2 == 0 and on_segment(p0, p3, p1):
        return True
    if o3 == 0 and on_segment(p2, p0, p3):
        return True
    if o4 == 0 and on_segment(p2, p1, p3):
        return True

    return False

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[ry] = rx
from collections import defaultdict

def group_lines_in_cell(chars_in_cell, y_tol=8.0):
    line_groups = defaultdict(list)
    for ch in chars_in_cell:
        line_key = round(ch["cy"] / y_tol) * y_tol
        line_groups[line_key].append(ch)

    lines = []
    for y in sorted(line_groups):
        sorted_line = sorted(line_groups[y], key=lambda c: c["cx"])
        styled_chars = []

        for c in sorted_line:
            style = f'font-family:{c["font"]}; font-size:{c["size"]}pt; color:{c["color"]};'
            if c.get("bold"):
                style += ' font-weight:bold;'
            if c.get("italic"):
                style += ' font-style:italic;'

            styled_char = f'<span style="{style}">{c["text"]}</span>'
            styled_chars.append(styled_char)

        lines.append(''.join(styled_chars))

    return "<br>".join(lines)


def drawings_to_html_table_with_text(
    drawings,
    chars,
    cell_bg_candidates,
    margin: float = 0.5
) -> str:
    scale = 1.67

    # 1. 선 정리
    h_lines = sorted([d for d in drawings if abs(d["y0"] - d["y1"]) < margin], key=lambda l: l["y0"])
    v_lines = sorted([d for d in drawings if abs(d["x0"] - d["x1"]) < margin], key=lambda l: l["x0"])
    y_coords = sorted(set(round(l["y0"], 2) for l in h_lines))
    x_coords = sorted(set(round(l["x0"], 2) for l in v_lines))
    n_rows, n_cols = len(y_coords) - 1, len(x_coords) - 1

    # 2. 셀 초기화
    cell_texts = [[[] for _ in range(n_cols)] for _ in range(n_rows)]

    for ch in chars:
        cx, cy = ch["cx"], ch["cy"]
        for row in range(n_rows):
            if y_coords[row] <= cy < y_coords[row + 1]:
                for col in range(n_cols):
                    if x_coords[col] <= cx < x_coords[col + 1]:
                        cell_texts[row][col].append(ch)
                        break
                break

    # 셀 배경 색상 후보 색상 배정
    def find_cell_bg(row: int, col: int) -> str:
        x0, x1 = x_coords[col], x_coords[col + 1]
        y0, y1 = y_coords[row], y_coords[row + 1]
        for bg in cell_bg_candidates:
            bx0, by0, bx1, by1 = bg["x0"], bg["y0"], bg["x1"], bg["y1"]
            if bx0 <= (x0 + x1) / 2 <= bx1 and by0 <= (y0 + y1) / 2 <= by1:
                return bg['color']
        return ""

    # 3. 선 유무 확인 함수
    def has_bottom_line(row, col):
        y = y_coords[row + 1]
        x0, x1 = x_coords[col], x_coords[col + 1]
        return any(abs(l["y0"] - y) < margin and l["x0"] <= x0 and l["x1"] >= x1 for l in h_lines)

    def has_right_line(row, col):
        x = x_coords[col + 1]
        y0, y1 = y_coords[row], y_coords[row + 1]
        return any(abs(l["x0"] - x) < margin and l["y0"] <= y0 and l["y1"] >= y1 for l in v_lines)

    visited = [[False for _ in range(n_cols)] for _ in range(n_rows)]

    # 4. 테이블 생성
    html = ['<table style="border-collapse: collapse;">']
    for row in range(n_rows):
        html.append("  <tr>")
        for col in range(n_cols):
            if visited[row][col]:
                continue

            # 병합 확장
            rowspan, colspan = 1, 1
            while row + rowspan < n_rows and not has_bottom_line(row + rowspan - 1, col):
                rowspan += 1
            while col + colspan < n_cols and not has_right_line(row, col + colspan - 1):
                colspan += 1

            # 병합 범위 내용 흡수 + visited 처리
            base_cell = cell_texts[row][col]
            for r in range(row, row + rowspan):
                for c in range(col, col + colspan):
                    if (r, c) != (row, col):
                        base_cell.extend(cell_texts[r][c])
                    visited[r][c] = True

            # 셀 내용, 정렬
            x0, x1 = x_coords[col], x_coords[col + colspan]
            y0, y1 = y_coords[row], y_coords[row + rowspan]
            if not base_cell:
                content = ""
                align = "left"
                valign = "top"
            else:
                lines = {}
                for ch in base_cell:
                    cy = round(ch["cy"], 1)
                    if cy not in lines:
                        lines[cy] = []
                    lines[cy].append((ch["cx"], ch["text"]))
                content = group_lines_in_cell(base_cell).strip().replace("\n", "<br>")

                avg_cx = sum(c['cx'] for c in base_cell) / len(base_cell)
                avg_cy = sum(c['cy'] for c in base_cell) / len(base_cell)
                cell_cx = (x0 + x1) / 2
                cell_cy = (y0 + y1) / 2

                align = (
                    "center" if abs(avg_cx - cell_cx) < (x1 - x0) * 0.1 else
                    "left" if avg_cx < cell_cx else
                    "right"
                )
                valign = (
                    "middle" if abs(avg_cy - cell_cy) < (y1 - y0) * 0.1 else
                    "top" if avg_cy < cell_cy else
                    "bottom"
                )

            # 배경색
            bg_color = find_cell_bg(row, col)
            bg_style = f" background-color: {bg_color};" if bg_color else ""

            span_attr = ""
            if rowspan > 1:
                span_attr += f' rowspan="{rowspan}"'
            if colspan > 1:
                span_attr += f' colspan="{colspan}"'

            style = (
                f'style="width: {(x1 - x0) * scale}px; height: {(y1 - y0)}px; '
                f'border: 1px solid black; padding: 4px; '
                f'text-align: {align}; vertical-align: {valign};{bg_style}"'
            )

            html.append(f'    <td{span_attr} {style}>{content}</td>')
        html.append("  </tr>")
    html.append("</table>")

    return "\n".join(html)

def int_to_rgb_string(color_int: int) -> str:
    r = (color_int >> 16) & 255
    g = (color_int >> 8) & 255
    b = color_int & 255
    return f"rgb({r}, {g}, {b})"

def extract_text_in_range_from_pdf(doc: fitz.Document,
                                   page_index: int,
                                   y0_px: float,
                                   y1_px: float,
                                   dpi: int = 300,
                                   left_margin_ratio: float = 0.15) -> str:
    """
    PDF 특정 페이지에서 y0~y1 범위의 텍스트를 추출하되,
    각 줄의 가장 왼쪽 항목(보통 제목)을 제거하고 나머지 줄은 공백으로 연결함.
    """
    from collections import defaultdict
    import fitz

    zoom = dpi / 72
    scale = 1 / zoom

    page = doc.load_page(page_index)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    w_pt = pix.width * scale
    x_threshold = w_pt * left_margin_ratio  # 왼쪽 가장자리 판단 기준

    clip = fitz.Rect(0, y0_px * scale, w_pt, y1_px * scale)
    char_dict = page.get_text("rawdict", clip=clip)
    char_lines = defaultdict(list)
    y_tol = 8.0

    for block in char_dict["blocks"]:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                for ch in span.get("chars", []):
                    x0, y0 = ch["bbox"][0], ch["bbox"][1]
                    y_key = round(y0 / y_tol) * y_tol
                    char_lines[y_key].append((x0, ch["c"]))

    sorted_lines = []
    for y in sorted(sorted(char_lines.keys())):
        line = sorted(char_lines[y], key=lambda t: t[0])
        # 왼쪽 텍스트 제거
        filtered = [t for t in line if t[0] > x_threshold]
        if not filtered:
            continue

        # 공백 복원
        buffer = []
        xs = [x for x, _ in filtered]
        for i, (x, ch) in enumerate(filtered):
            if i > 0:
                prev_x = xs[i - 1]
                if x - prev_x > 20:
                    buffer.append(" ")
            buffer.append(ch)
        sorted_lines.append("".join(buffer))

    return " ".join(sorted_lines)

def trim_horizontal_whitespace(image: Image.Image, threshold=245) -> Image.Image:
    """
    좌우 흰 여백을 잘라낸 이미지를 반환합니다.
    threshold: 밝기 기준 (0~255), 이 값 이상이면 '흰색'으로 간주.
    """
    # 흑백 이미지로 변환
    gray = image.convert('L')
    # 흰색(255) 기준으로 이진화: 흰색은 0, 나머지는 255
    bw = gray.point(lambda x: 0 if x > threshold else 255, mode='1')
    # 박스 추출
    bbox = bw.getbbox()  # (left, upper, right, lower)
    if bbox:
        left, upper, right, lower = bbox
        # 상하 여백은 유지하고 좌우만 자름
        return image.crop((left, 0, right, image.height))
    else:
        return image  # 내용이 없으면 원본 그대로

def process_pdf_blocks(doc, title, blocks, output_dir, dpi=300):
    """
    PDF 블록 처리 (char 기반 bbox 정렬):
    - text 블록: 문자 단위로 x좌표 정렬 + 공백 복원하여 .txt로 저장
    - nontext 블록: 이미지를 .png로 저장
    """
    os.makedirs(output_dir, exist_ok=True)
    zoom = dpi / 72
    scale = 1 / zoom
    mat = fitz.Matrix(zoom, zoom)

    meta = [{
        "type": "title",
        "data": title
    }]

    for idx, blk in enumerate(blocks, start=1):
        ps = int(blk['page_s']) - 1
        pe = int(blk['page_e']) - 1
        y0_px, y1_px = blk['y0'], blk['y1']
        btype = blk.get('type', 'nontext')

        if btype == 'text':
            full_text = []
            for pi in range(ps, pe + 1):
                page = doc.load_page(pi)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                w, h = pix.width, pix.height

                x0_pt, x1_pt = 0, w * scale
                if pi == ps == pe:
                    y0_pt, y1_pt = y0_px * scale, y1_px * scale
                elif pi == ps:
                    y0_pt, y1_pt = y0_px * scale, h * scale
                elif pi == pe:
                    y0_pt, y1_pt = 0, y1_px * scale
                else:
                    y0_pt, y1_pt = 0, h * scale

                clip = fitz.Rect(x0_pt, y0_pt, x1_pt, y1_pt)
                char_dict = page.get_text("rawdict", clip=clip)
                char_lines = defaultdict(list)

                y_tol = 12.0
                for block in char_dict["blocks"]:
                    if "lines" not in block:
                        continue
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if "chars" not in span:
                                continue
                            for ch in span["chars"]:
                                c = ch["c"]
                                x0, y0 = ch["bbox"][0], ch["bbox"][1]
                                y_key = round(y0 / y_tol) * y_tol
                                char_lines[y_key].append((x0, c))
                
                sorted_y = sorted(char_lines.keys())
                line_x_starts = {y: min(x for x, _ in char_lines[y]) for y in sorted_y}
                min_x_start = min(line_x_starts.values())
                
                line_texts = []
                for y in sorted_y:
                    chars = sorted(char_lines[y], key=lambda t: t[0])
                    buffer = []
                    xs = [x for x, _ in chars]
                    if len(xs) > 1:
                        diffs = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
                        avg_gap = sum(diffs) / len(diffs)
                        std_gap = (sum((g - avg_gap) ** 2 for g in diffs) / len(diffs)) ** 0.5
                        threshold = avg_gap + std_gap * 1.6
                    else:
                        threshold = 3.0

                    x_offset = line_x_starts[y] - min_x_start
                    num_spaces = int(x_offset / 7.0)  # 글자당 픽셀 너비 (평균 6~8 정도로 가정)
                    buffer.extend([" "] * num_spaces)

                    for i, (x, ch) in enumerate(chars):
                        if i > 0:
                            prev_x = chars[i - 1][0]
                            gap = x - prev_x
                            if gap > 13:
                                buffer.append(" ")
                        buffer.append(ch)
                    line_texts.append("".join(buffer))

                page_text = "\n".join(line_texts)
                full_text.append(page_text)

            raw = "\n".join(full_text)
            txt_path = os.path.join(output_dir, f"text_block_{idx}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(raw)
            meta.append({
                "type": "text",
                "filename": os.path.basename(txt_path)
            })

        else:
            page1 = doc.load_page(ps)
            pix1 = page1.get_pixmap(matrix=mat, alpha=False)
            img1 = Image.frombytes("RGB", (pix1.width, pix1.height), pix1.samples)
            part1 = img1.crop((0, y0_px, pix1.width, pix1.height))

            if ps != pe:
                page2 = doc.load_page(pe)
                pix2 = page2.get_pixmap(matrix=mat, alpha=False)
                img2 = Image.frombytes("RGB", (pix2.width, pix2.height), pix2.samples)
                part2 = img2.crop((0, 0, pix2.width, y1_px))
                merged = Image.new("RGB", (pix1.width, part1.height + part2.height))
                merged.paste(part1, (0, 0))
                merged.paste(part2, (0, part1.height))
                cropped = merged
                suffix = f"_p{blk['page_s']}_to_p{blk['page_e']}"
            else:
                cropped = img1.crop((0, y0_px, pix1.width, y1_px))
                suffix = ""
            cropped = trim_horizontal_whitespace(cropped)
            img_path = os.path.join(output_dir, f"image_block_{idx}{suffix}.png")
            cropped.save(img_path)

            y0_pt = y0_px * scale
            y1_pt = y1_px * scale

            chars = []
            clip = fitz.Rect(x0_pt, y0_pt, x1_pt, y1_pt)
            char_dict = page.get_text("rawdict", clip=clip)

            for block in char_dict["blocks"]:
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        if "chars" not in span:
                            continue
                        for ch in span["chars"]:
                            font = span["font"]
                            size = span["size"]
                            flags = span["flags"]
                            color_val = span["color"]
                            color_rgb_string = int_to_rgb_string(color_val)
                            c = ch["c"]
                            x0, y0 = ch["bbox"][0], ch["bbox"][1]
                            x1, y1 = ch["bbox"][2], ch["bbox"][3]
                            chars.append({
                                "text": c,
                                "cx": (round(x0, 2) + round(x1, 2)) / 2,
                                "cy": (round(y0, 2) + round(y1, 2)) / 2,
                                "font": font,
                                "size": size,
                                "bold": bool(flags & 2),     # Bold
                                "italic": bool(flags & 1),   # Italic
                                "color":  color_rgb_string
                            })


            for block in char_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue  # 텍스트 블록만 처리
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "")
                        bbox = span.get("bbox", [])
                        if len(bbox) != 4 or not text:
                            continue

                        span_x0, span_y0, span_x1, span_y1 = bbox
                        num_chars = len(text)
                        if num_chars == 0:
                            continue

                        # 한 글자당 너비 비례 계산
                        avg_width = (span_x1 - span_x0) / num_chars

                        for i, ch in enumerate(text):
                            x0 = span_x0 + i * avg_width
                            x1 = x0 + avg_width
                            chars.append({
                                "text": ch,
                                "x0": round(x0, 2),
                                "y0": round(span_y0, 2),
                                "x1": round(x1, 2),
                                "y1": round(span_y1, 2)
                            })
            page = doc.load_page(ps)
            drawings = page.get_drawings()
            segs = []
            for d in drawings:
                y_center = (d["rect"].y0 + d["rect"].y1) / 2
                if not (y0_pt <= y_center <= y1_pt):
                    continue
                x0 = round(d['rect'].x0, 2); y0 = round(d['rect'].y0, 2)
                x1 = round(d['rect'].x1, 2); y1 = round(d['rect'].y1, 2)
                if abs(x0 - x1) > 1e-2 and abs(y0 - y1) > 1e-2:
                    continue
                segs.append({
                    'type': d['type'],
                    'x0': x0, 'y0': y0,
                    'x1': x1, 'y1': y1,
                })
            cell_bg_candidates = []
            for d in drawings:
                for item in d["items"]:
                    if item[0] == "re":
                        x0, y0, x1, y1 = item[1]
                        fill = d.get("fill")
                        r, g, b = [int(c * 255) for c in fill[:3]]
                        cell_bg_candidates.append({'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1, 'color': f"rgb({r}, {g}, {b})"})
            for c in cell_bg_candidates:
                print(c)
            # 2) Union-Find 초기화
            n = len(segs)
            uf = UnionFind(n)

            # 3) 모든 쌍 검사해서 교차하면 union
            for i in range(n):
                for j in range(i+1, n):
                    if segments_intersect(segs[i], segs[j]):
                        uf.union(i, j)

            # 4) 최종 그룹화
            drawings_group = {}
            for idx, seg in enumerate(segs):
                root = uf.find(idx)
                drawings_group.setdefault(root, []).append(seg)

            # 리스트 형태로 꺼내기
            drawings_group = list(drawings_group.values())
            valid_drawings_group = [gp for gp in drawings_group if len(gp) >= 4]
            if len(valid_drawings_group) == 1:
                table = drawings_to_html_table_with_text(valid_drawings_group[0], chars, cell_bg_candidates)
                meta.append({
                    "type": "table",
                    "html": table,
                })
            else:
                meta.append({
                    "type": "image",
                    "filename": os.path.basename(img_path),
                    "drawings": valid_drawings_group,
                    "width": cropped.width / zoom * 1.67,
                    "height": cropped.height / zoom * 1.67
                })
    

    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved meta: {meta_path}")

def save_pdf_blocks(pdf_bytes, uuid):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    title_start_y, title_end_y = test_title_crop_from_pdf(doc)
    _, start_y = get_top_line_y_from_pdf(doc)
    end_page, end_y = find_body_range_in_pdf(doc)

    print(f"제목은 y={title_start_y} 부터 y={title_end_y}까지로 판단됨")
    #print(f"본문은 y={start_y} 부터 페이지 {end_page}의 y={end_y}까지로 판단됨")

    title = extract_text_in_range_from_pdf(doc, page_index=0, y0_px=title_start_y, y1_px=title_end_y)
    print(title)
    body_img = stitch_body_region(doc, start_y, end_page, end_y)
    body_img.save("final_body.png")

    bs = split_by_smart_blank_lines(body_img)
    bs = merge_blocks_with_tag(bs)

    #print(bs)

    result = mapping_blocks(bs, start_y, doc)
    #print(result)

    crop_pdf_blocks_with_fitz(doc, result, "output")
    process_pdf_blocks(doc, title, result, os.path.join("static", uuid))