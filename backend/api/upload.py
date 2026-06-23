import os
import shutil
from typing import List
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException, status

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".csv", ".xlsx"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

from backend.storage.session_store import SessionStore

@router.post("/upload-single/", status_code=status.HTTP_201_CREATED)
async def upload_single_file(file: UploadFile = File(...)):
    """
    Accepts a single file, validates size/extension, saves it locally,
    and initializes a new AutoML session with the file path.
    
    Args:
        file (UploadFile): File to upload
    
    Returns:
        Dict[str, Any]: Dictionary containing session ID, filename, content type, and message
    """
    # Validate File Extension
    ext = Path(file.filename).suffix
    if ext.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Extension {ext} not allowed. Choose from: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File too large"
        )

    # Save the file to disk
    file_path = os.path.join(UPLOAD_DIR, Path(file.filename).name)
    with open(file_path, "wb") as buffer:
        buffer.write(contents)

    # Initialize a new session and save the file path
    session_store = SessionStore()
    session_id = session_store.create_session()
    session_store.set(session_id, "file_path", file_path)

    return {
        "session_id": session_id,
        "filename": file.filename,
        "content_type": file.content_type,
        "message": "File uploaded successfully and session initialized."
    }

@router.post("/upload-multiple/", status_code=status.HTTP_201_CREATED)
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    """
    Accepts multiple files simultaneously and lists their filenames.

    Args:
        files (List[UploadFile]): List of files to upload
    
    Returns:
        Dict[str, List[str]]: Dictionary containing list of uploaded filenames
    """
    uploaded_filenames = []
    for file in files:
        ext = Path(file.filename).suffix
        if ext.lower() not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Extension {ext} not allowed. Choose from: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        file_path = os.path.join(UPLOAD_DIR, Path(file.filename).name)
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            uploaded_filenames.append(file.filename)
        finally:
            await file.close()
            
    return {"uploaded_files": uploaded_filenames}