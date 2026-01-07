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


@router.get("/preview/{filename:path}")
async def preview_dataset(filename: str, rows: int = 10):
    """
    Get preview and statistics for a dataset
    - CSV: Returns first N rows, column names, row count, data types
    - TXT: Returns first N lines, character count, line count
    - Image folder: Returns class names, image counts per class, sample paths
    """
    import pandas as pd
    from PIL import Image
    
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    result = {
        "filename": filename,
        "path": file_path,
        "file_type": "unknown",
        "size_bytes": 0,
    }
    
    # Handle folder (image dataset)
    if os.path.isdir(file_path):
        result["file_type"] = "image_folder"
        
        # Get class folders and image counts
        classes = {}
        sample_images = []
        total_images = 0
        total_size = 0
        
        for item in os.listdir(file_path):
            item_path = os.path.join(file_path, item)
            if os.path.isdir(item_path):
                # This is a class folder
                images = [f for f in os.listdir(item_path) 
                         if Path(f).suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
                classes[item] = len(images)
                total_images += len(images)
                
                # Get sample image from this class
                if images and len(sample_images) < 5:
                    sample_path = os.path.join(item_path, images[0])
                    try:
                        with Image.open(sample_path) as img:
                            sample_images.append({
                                "class": item,
                                "filename": images[0],
                                "size": f"{img.width}x{img.height}",
                                "mode": img.mode
                            })
                    except:
                        pass
                
                # Calculate folder size
                for img_file in images:
                    img_path = os.path.join(item_path, img_file)
                    total_size += os.path.getsize(img_path)
            elif Path(item).suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                # Images directly in the folder (no class subfolders)
                classes["unlabeled"] = classes.get("unlabeled", 0) + 1
                total_images += 1
                total_size += os.path.getsize(item_path)
                
                if len(sample_images) < 5:
                    try:
                        with Image.open(item_path) as img:
                            sample_images.append({
                                "class": "unlabeled",
                                "filename": item,
                                "size": f"{img.width}x{img.height}",
                                "mode": img.mode
                            })
                    except:
                        pass
        
        result["size_bytes"] = total_size
        result["classes"] = classes
        result["num_classes"] = len(classes)
        result["total_images"] = total_images
        result["sample_images"] = sample_images
        
        return result
    
    # Handle files
    file_ext = Path(filename).suffix.lower()
    result["size_bytes"] = os.path.getsize(file_path)
    
    # CSV file
    if file_ext == ".csv":
        result["file_type"] = "csv"
        try:
            # Read CSV with pandas
            df = pd.read_csv(file_path, nrows=rows + 1)  # +1 for header detection
            full_df = pd.read_csv(file_path)
            
            result["columns"] = list(df.columns)
            result["num_columns"] = len(df.columns)
            result["num_rows"] = len(full_df)
            result["preview_rows"] = rows
            
            # Get data types
            result["dtypes"] = {col: str(dtype) for col, dtype in full_df.dtypes.items()}
            
            # Get missing values count
            result["missing_values"] = {col: int(full_df[col].isna().sum()) for col in full_df.columns}
            result["total_missing"] = int(full_df.isna().sum().sum())
            
            # Get unique values for target (last column)
            target_col = df.columns[-1]
            result["target_column"] = target_col
            result["unique_targets"] = int(full_df[target_col].nunique())
            result["target_values"] = full_df[target_col].value_counts().head(10).to_dict()
            
            # Preview data as list of lists (header + rows)
            preview_data = [list(df.columns)]
            for _, row in df.head(rows).iterrows():
                preview_data.append([str(v) if pd.notna(v) else "" for v in row.values])
            
            result["preview"] = preview_data
            
        except Exception as e:
            result["error"] = str(e)
            result["preview"] = []
        
        return result
    
    # Text file
    if file_ext == ".txt":
        result["file_type"] = "text"
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            lines = content.split('\n')
            result["num_lines"] = len(lines)
            result["num_characters"] = len(content)
            result["num_words"] = len(content.split())
            
            # Get unique characters (vocabulary for char-level models)
            unique_chars = sorted(set(content))
            result["vocab_size"] = len(unique_chars)
            result["sample_vocab"] = unique_chars[:50]  # First 50 chars
            
            # Preview first N lines
            result["preview"] = [[line] for line in lines[:rows]]
            
            # Check if it's tab-separated (classification format)
            if '\t' in lines[0] if lines else '':
                result["format"] = "tab_separated"
                # Try to parse as label\ttext format
                try:
                    labels = set()
                    for line in lines[:100]:
                        if '\t' in line:
                            label = line.split('\t')[0]
                            labels.add(label)
                    result["detected_labels"] = list(labels)[:20]
                except:
                    pass
            else:
                result["format"] = "plain_text"
            
        except Exception as e:
            result["error"] = str(e)
            result["preview"] = []
        
        return result
    
    # Other file types - just basic info
    result["file_type"] = file_ext.replace(".", "")
    result["preview"] = []
    
    return result