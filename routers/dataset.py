"""
Dataset Router - Handles dataset upload and validation
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import os
import shutil
from pathlib import Path

router = APIRouter()

UPLOAD_DIR = "uploads"
MAX_FILE_SIZE = 1024 * 1024 * 1024  # 1GB
ALLOWED_EXTENSIONS = {".csv", ".jpg", ".jpeg", ".png", ".bmp"}


@router.post("/upload")
async def upload_dataset(files: List[UploadFile] = File(...)):
    """
    Upload dataset files (CSV or images)
    - Validates file types and size
    - Stores in /uploads folder
    """
    uploaded_files = []

    for file in files:
        # Validate file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file_ext}. Allowed: {ALLOWED_EXTENSIONS}"
            )

        # Read file content
        content = await file.read()
        file_size = len(content)

        # Validate file size
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large: {file_size} bytes. Max: {MAX_FILE_SIZE} bytes"
            )

        # Save file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(content)

        uploaded_files.append({
            "filename": file.filename,
            "size": file_size,
            "path": file_path
        })

    return {
        "message": "Files uploaded successfully",
        "files": uploaded_files,
        "count": len(uploaded_files)
    }


@router.get("/list")
async def list_datasets():
    """List all uploaded datasets"""
    if not os.path.exists(UPLOAD_DIR):
        return {"files": []}

    files = []
    for filename in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.isfile(file_path):
            files.append({
                "filename": filename,
                "size": os.path.getsize(file_path),
                "path": file_path
            })

    return {"files": files, "count": len(files)}


@router.delete("/delete/{filename}")
async def delete_dataset(filename: str):
    """Delete a specific dataset"""
    file_path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    os.remove(file_path)
    return {"message": f"File {filename} deleted successfully"}


@router.delete("/clear")
async def clear_all_datasets():
    """Clear all uploaded datasets"""
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
        os.makedirs(UPLOAD_DIR)

    return {"message": "All datasets cleared"}