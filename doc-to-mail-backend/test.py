import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import pytesseract
import os
from PIL import ImageDraw

def trim_white_top_bottom(img: Image.Image, threshold=240) -> Image.Image:
    """
    이미지에서 상하단 흰 여백 제거 (완전 흰 줄만 기준)
    """
    np_img = np.array(img)
    mean_vals = np.mean(np_img, axis=1)  # 각 줄의 평균 밝기

    y_nonwhite = np.where(mean_vals < threshold)[0]
    if len(y_nonwhite) == 0:
        return img  # 모두 흰색이면 그대로 반환

    top = y_nonwhite[0]
    bottom = y_nonwhite[-1] + 1
    return img.crop((0, top, img.width, bottom))

def split_pdf_page_by_y_gap_with_ocr(
    pdf_path: str,
    dpi: int = 200,
    min_gap_height: int = 40,
    lang: str = "kor+eng"
):
    """
    PDF의 지정 페이지를 이미지로 변환한 뒤,
    수평 공백(y 간격)을 기준으로 이미지 crop하고 OCR로 텍스트 추출

    Returns: List of (cropped_image, ocr_text) 튜플
    """
    doc = fitz.open(pdf_path)    
    trimmed_images = []    
    
    # 1. 각 페이지를 이미지로 렌더링
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        trimmed = trim_white_top_bottom(img)
        trimmed_images.append(trimmed)

    # 2. 세로로 이어붙이기
    width = trimmed_images[0].width
    total_height = sum(p.height for p in trimmed_images)
    stitched = Image.new("L", (width, total_height))
    
    offset = 0
    for p in trimmed_images:
        stitched.paste(p, (0, offset))
        offset += p.height

    np_img = np.array(stitched)

    # 공백 탐지
    white_rows = np.mean(np_img < 200, axis=1) < 0.01

    # 공백 구간 기준 crop 영역 추출
    start_y = 0
    gap_start = None
    gap_indices = []

    for i, is_white in enumerate(white_rows):
        if is_white:
            if gap_start is None:
                gap_start = i
        else:
            if gap_start is not None and (i - gap_start) > min_gap_height:
                gap_indices.append((start_y, gap_start))
                start_y = i
            gap_start = None
    if start_y < np_img.shape[0] - 5:
        gap_indices.append((start_y, np_img.shape[0]))

    # Crop + OCR
    results = []
    img_height = img.height

    for idx, (y0, y1) in enumerate(gap_indices):
        y0 = max(0, int(y0))
        y1 = min(img_height, int(y1))
        if y1 <= y0:
            continue

        print(f"[{idx+1}] Crop from y={y0} to y={y1} (height={y1 - y0})")
        cropped = stitched.crop((0, y0, stitched.width, y1))
        text = pytesseract.image_to_string(cropped, lang=lang)
        results.append((cropped, text))
    # cleanup
    #os.remove(pdf_path)
    draw_y_lines_on_image(stitched, gap_indices)
    return results

def draw_y_lines_on_image(image, gap_indices, output_path="debug_page.png"):
    debug_img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(debug_img)

    for y0, y1 in gap_indices:
        draw.line([(0, y0), (image.width, y0)], fill=(255, 0, 0), width=2)
        draw.line([(0, y1), (image.width, y1)], fill=(0, 0, 255), width=2)

    debug_img.save(output_path)
    print(f"🖼️ 시각화 이미지 저장됨: {output_path}")

results = split_pdf_page_by_y_gap_with_ocr("temp.pdf")

for i, (img, text) in enumerate(results):
    img.save(f"crop_{i+1}.png")
    #print(f"🧾 OCR 결과 ({i+1}번):\n{text}\n{'-'*50}")