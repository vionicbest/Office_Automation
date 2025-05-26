from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import fitz  # PyMuPDF

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.post("/upload", response_class=HTMLResponse)
async def upload_pdf(request: Request, pdf: UploadFile = File(...)):
    # PDF 열기
    doc = fitz.open(stream=await pdf.read(), filetype="pdf")
    page = doc[0]  # 첫 페이지만 대상으로

    # 텍스트 + 좌표 추출
    blocks = page.get_text("dict")["blocks"]
    lines = []

    for block in blocks:
        if block["type"] == 0:
            for line in block["lines"]:
                line_text = " ".join([span["text"] for span in line["spans"]])
                lines.append(line_text)

    # 단순 <pre>로 구성한 HTML 본문
    html_body = "<br>".join(lines)

    return templates.TemplateResponse("preview.html", {
        "request": request,
        "html_body": html_body
    })
