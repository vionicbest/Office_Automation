import hashlib
import os
import re

def generate_doc_id_from_bytes(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()[:16]  # 앞 16자리만 사용해도 충분

def collect_blocks(uuid: str):
    base_path = f"{uuid}"
    files = os.listdir(base_path)

    blocks = []
    for f in sorted(files, key=lambda x: int(re.search(r'_(\d+)', x).group(1))):
        block_id = int(re.search(r'_(\d+)', f).group(1))
        if f.endswith(".txt"):
            blocks.append({
                "type": "text",
                "id": block_id,
                "src": f"{uuid}/{f}"
            })
        elif f.endswith(".png"):
            blocks.append({
                "type": "image",
                "id": block_id,
                "src": f"{uuid}/{f}"
            })

    return blocks
