import os
from typing import List
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, status

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".csv", ".xlsx"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

@app.post("/upload-single/", status_code=status.HTTP_201_CREATED)
async def upload_single_file(file: UploadFile = File(...)):
    """
    Accepts a single file, validates size/extension, and saves it locally.
    """
    # Validate File Extension
    _, ext = Path(file.filename).name
    if ext.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Extension {ext} not allowed. Choose from: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Save the file to disk
    file_path = Path(file.filename).name
    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File too large"
        )

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "message": "File uploaded successfully."
    }

@app.post("/upload-multiple/", status_code=status.HTTP_201_CREATED)
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    """
    Accepts multiple files simultaneously and lists their filenames.
    """
    uploaded_filenames = []
    for file in files:
        file_path = Path(file.filename).name
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            uploaded_filenames.append(file.filename)
        finally:
            await file.close()
            
    return {"uploaded_files": uploaded_filenames}