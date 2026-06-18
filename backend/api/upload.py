import os
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".csv", ".xlsx"}

#also add a max file limit

@app.post("/upload-single/", status_code=status.HTTP_201_CREATED)
async def upload_single_file(file: UploadFile = File(...)):
    """
    Accepts a single file, validates size/extension, and saves it locally.
    """
    # 1. Validate File Extension
    _, ext = os.path.splitext(file.filename)
    if ext.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Extension {ext} not allowed. Choose from: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # 2. Save the file to disk in chunks to optimize memory
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": file_size,
        "message": "File uploaded successfully."
    }

@app.post("/upload-multiple/", status_code=status.HTTP_201_CREATED)
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    """
    Accepts multiple files simultaneously and lists their filenames.
    """
    uploaded_filenames = []
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        try:
            contents = await file.read()
            with open(file_path, "wb") as buffer:
                buffer.write(contents)
            uploaded_filenames.append(file.filename)
        finally:
            await file.close()
            
    return {"uploaded_files": uploaded_filenames}