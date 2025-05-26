from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pdfplumber

app = FastAPI()

# CORS 허용 (Next.js 클라이언트용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영 시 origin 제한하는 게 안전
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/extract-pdf-text")
async def extract_pdf_text(file: UploadFile = File(...)):
    contents = await file.read()
    with open("temp.pdf", "wb") as f:
        f.write(contents)

    extracted_text = ""
    with pdfplumber.open("temp.pdf") as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n\n"

    return JSONResponse(content={"text": extracted_text.strip()})
