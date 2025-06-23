from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import fitz  # PyMuPDF
from split_pdf import save_pdf_blocks
from util import generate_doc_id_from_bytes, collect_blocks

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://wise-pots-lick.loca.lt/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    pdf_bytes = await file.read()

    doc_id = generate_doc_id_from_bytes(pdf_bytes)

    save_pdf_blocks(pdf_bytes, doc_id)
    print(123)

    return {"docId": doc_id}


@app.get("/api/docs/{docId}")
def get_document(doc_id: str):
    try:
        blocks = collect_blocks(doc_id)
        return {
            "docId": doc_id,
            "blocks": blocks
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Document not found")