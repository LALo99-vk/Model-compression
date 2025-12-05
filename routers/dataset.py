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
ALLOWED_EXTENSIONS = {".csv", ".txt", ".jpg", ".jpeg", ".png", ".bmp"}


@router.post("/upload")
async def upload_dataset(files: List[UploadFile] = File(...)):
    """
    Upload dataset files (CSV or images) or folders
    - Validates file types and size
    - Stores in /uploads folder
    - Preserves folder structure for image datasets
    """
    uploaded_files = []
    folders = set()

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

        # Preserve folder structure (extract folder name from path)
        # Browser sends paths like "my_images/cat/img1.jpg"
        file_parts = Path(file.filename).parts
        
        if len(file_parts) > 1:
            # This is part of a folder upload
            folder_name = file_parts[0]
            folders.add(folder_name)
            file_path = os.path.join(UPLOAD_DIR, *file_parts)
        else:
            # Single file upload
            file_path = os.path.join(UPLOAD_DIR, file.filename)

        # Create parent directories if needed
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save file
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
        "folders": list(folders),
        "count": len(uploaded_files)
    }


@router.get("/list")
async def list_datasets():
    """List all uploaded datasets (files and folders)"""
    if not os.path.exists(UPLOAD_DIR):
        return {"files": []}

    files = []
    for filename in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        if os.path.isfile(file_path):
            # Single file
            files.append({
                "filename": filename,
                "size": os.path.getsize(file_path),
                "path": file_path,
                "type": "file"
            })
        elif os.path.isdir(file_path):
            # Folder (image dataset)
            # Calculate total size of all files in folder
            total_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, _, filenames in os.walk(file_path)
                for filename in filenames
            )
            # Count total images
            image_count = sum(
                1 for dirpath, _, filenames in os.walk(file_path)
                for f in filenames if Path(f).suffix.lower() in ALLOWED_EXTENSIONS
            )
            
            files.append({
                "filename": filename,
                "size": total_size,
                "path": file_path,
                "type": "folder",
                "image_count": image_count
            })

    return {"files": files, "count": len(files)}


@router.delete("/delete/{filename}")
async def delete_dataset(filename: str):
    """Delete a specific dataset (file or folder)"""
    file_path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File/folder not found")

    if os.path.isfile(file_path):
        os.remove(file_path)
    elif os.path.isdir(file_path):
        shutil.rmtree(file_path)
    
    return {"message": f"Dataset {filename} deleted successfully"}


@router.delete("/clear")
async def clear_all_datasets():
    """Clear all uploaded datasets"""
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
        os.makedirs(UPLOAD_DIR)

    return {"message": "All datasets cleared"}