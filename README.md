# 🔬 Model Compression Pipeline

<div align="center">

**A complete end-to-end machine learning solution for training, evaluating, and compressing neural network models with an intuitive web interface.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18.3-61dafb.svg)](https://reactjs.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-ee4c2c.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Detailed Installation](#-detailed-installation)
- [User Guide](#-user-guide)
- [Supported Models](#-supported-models)
- [Dataset Formats](#-dataset-formats)
- [Compression Methods](#-compression-methods)
- [API Reference](#-api-reference)
- [Project Structure](#-project-structure)
- [Production Deployment](#-production-deployment)
- [Troubleshooting](#-troubleshooting)
- [FAQ](#-faq)

---

## 🎯 Overview

Model Compression Pipeline is a full-stack application that simplifies the entire machine learning workflow:

```
📁 Upload Data → 🤖 Select Model → ✅ Validate → 🏋️ Train → 📦 Compress → 📊 Compare
```

The application automatically handles data preprocessing, model training, and applies state-of-the-art compression techniques to reduce model size while maintaining accuracy.

---

## ✨ Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| 🤖 **Multi-Model Support** | Decision Tree, CNN, RNN/LSTM architectures |
| 🔄 **Auto Preprocessing** | Automatic validation, normalization, and encoding |
| 📦 **Model Compression** | Pruning, Quantization, Knowledge Distillation |
| 📈 **Real-Time Training** | Live progress, loss curves, and metrics |
| ⚖️ **Model Comparison** | Side-by-side original vs compressed analysis |
| 💾 **Export Models** | Download trained and compressed models |
| 🎨 **Modern UI** | Dark-themed, responsive React interface |

### Additional Features

- **Training History** - View and manage all your trained models
- **Dataset Preview** - Visualize your data before training
- **Analytics Dashboard** - Charts and insights about your models
- **Comparison Reports** - Downloadable detailed reports

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** - [Download](https://python.org/downloads)
- **Node.js 18+** - [Download](https://nodejs.org)
- **Git** - [Download](https://git-scm.com)

### One-Minute Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd Model-compression

# 2. Start Backend (Terminal 1)
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py

# 3. Start Frontend (Terminal 2)
cd frontend2
npm install
npm run dev

# 4. Open browser
# Navigate to: http://localhost:5173
```

✅ **Backend API**: http://localhost:8000  
✅ **Frontend App**: http://localhost:5173  
✅ **API Docs**: http://localhost:8000/docs

---

## 📦 Detailed Installation

### Backend Setup (Python/FastAPI)

#### Step 1: Create Virtual Environment

**macOS / Linux:**
```bash
cd Model-compression
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
cd Model-compression
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
cd Model-compression
python -m venv .venv
.\.venv\Scripts\activate.bat
```

#### Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

<details>
<summary>📋 <strong>View All Python Dependencies</strong></summary>

| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | 0.104.1 | Web framework |
| uvicorn | 0.24.0 | ASGI server |
| torch | 2.1.0 | Deep learning (PyTorch) |
| torchvision | 0.16.0 | Image processing |
| tensorflow | 2.15.0 | Deep learning (TensorFlow) |
| scikit-learn | 1.3.2 | Machine learning utilities |
| pandas | 2.1.3 | Data manipulation |
| numpy | 1.26.2 | Numerical computing |
| Pillow | 10.1.0 | Image processing |
| pydantic | 2.5.0 | Data validation |
| python-multipart | 0.0.6 | File uploads |
| aiofiles | 23.2.1 | Async file operations |

</details>

#### Step 3: Start the Server

**Development mode (with auto-reload):**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Production mode:**
```bash
python main.py
```

#### Step 4: Verify Installation

```bash
# Check server health
curl http://localhost:8000/health
# Expected: {"status": "healthy"}

# Or visit in browser:
# http://localhost:8000/docs (Swagger UI)
```

---

### Frontend Setup (React/TypeScript)

#### Step 1: Navigate to Frontend Directory

```bash
cd frontend2
```

#### Step 2: Install Dependencies

```bash
npm install
```

<details>
<summary>📋 <strong>View Frontend Dependencies</strong></summary>

| Package | Purpose |
|---------|---------|
| react | UI library |
| axios | HTTP client |
| zustand | State management |
| lucide-react | Icons |
| tailwindcss | CSS framework |
| vite | Build tool |
| typescript | Type safety |

</details>

#### Step 3: Start Development Server

```bash
npm run dev
```

#### Step 4: Access the Application

Open **http://localhost:5173** in your browser.

---

## 📖 User Guide

### Complete Workflow

The application guides you through 6 simple steps:

<table>
<tr>
<td align="center" width="16%">
<strong>1️⃣ Upload</strong><br/>
Import your dataset
</td>
<td align="center" width="16%">
<strong>2️⃣ Model</strong><br/>
Select model type
</td>
<td align="center" width="16%">
<strong>3️⃣ Validate</strong><br/>
Check data quality
</td>
<td align="center" width="16%">
<strong>4️⃣ Train</strong><br/>
Train the model
</td>
<td align="center" width="16%">
<strong>5️⃣ Compress</strong><br/>
Reduce model size
</td>
<td align="center" width="16%">
<strong>6️⃣ Results</strong><br/>
Compare & download
</td>
</tr>
</table>

---

### Step 1: Upload Dataset

1. Click **"Upload"** in the sidebar
2. Drag & drop files or click to browse
3. Supported formats:
   - **CSV files** (.csv) - Tabular data
   - **Text files** (.txt) - Text/sequence data
   - **Image folders** (ZIP) - Image classification

> 💡 **Tip:** For large datasets, use the preview feature to verify your data was uploaded correctly.

---

### Step 2: Select Model

1. Click **"Model"** in the sidebar
2. Choose mode:

| Mode | Description |
|------|-------------|
| **Auto** | System recommends the best model |
| **Manual** | Choose your preferred model |

3. Available models:

| Model | Best For |
|-------|----------|
| **Decision Tree** | Tabular data, fast training |
| **CNN** | Images, spatial patterns |
| **RNN/LSTM** | Text, sequences, time series |

---

### Step 3: Validate Dataset

1. Click **"Validation"** in the sidebar
2. System automatically validates your dataset
3. Common checks:
   - ✅ File format
   - ✅ Missing values
   - ✅ Data types
   - ✅ Class balance
4. If issues are found, click **"Auto-Fix"**

---

### Step 4: Train Model

1. Click **"Training"** in the sidebar
2. Configure parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Epochs** | 20 | Training iterations |
| **Batch Size** | 32 | Samples per batch |
| **Learning Rate** | 0.001 | Optimization speed |
| **Validation Split** | 0.2 | Validation data portion |

3. Click **"Start Training"**
4. Monitor real-time:
   - Live epoch updates
   - Loss curves
   - Accuracy/metrics

---

### Step 5: Compress Model

1. Click **"Compression"** in the sidebar
2. View original model statistics
3. Click **"Start Compression"**
4. System applies:
   - ✂️ Weight Pruning
   - 🔢 INT8 Quantization
   - 📚 Knowledge Distillation
5. Download buttons available for both models

---

### Step 6: View Results

1. Click **"Results"** in the sidebar
2. Compare metrics:

| Metric | Original | Compressed |
|--------|----------|------------|
| Size | X MB | Y MB |
| Accuracy | X% | Y% |
| Parameters | X | Y |

3. View visualizations:
   - Bar chart comparison
   - Radar chart (multi-metric)
4. Download comparison report

---

## 🤖 Supported Models

### Decision Tree

<table>
<tr>
<td width="200">
<strong>Best For</strong><br/>
Tabular/structured data
</td>
<td>

| Attribute | Value |
|-----------|-------|
| Framework | Scikit-learn |
| Tasks | Classification, Regression |
| Input | CSV tabular data |
| Speed | ⚡ Very Fast |
| Interpretability | 🟢 High |

</td>
</tr>
</table>

**Compression Methods:**
- Cost-complexity pruning
- Depth/node reduction
- Tree distillation
- Gzip compression

---

### CNN (Convolutional Neural Network)

<table>
<tr>
<td width="200">
<strong>Best For</strong><br/>
Images, spatial patterns
</td>
<td>

| Attribute | Value |
|-----------|-------|
| Framework | PyTorch |
| Tasks | Classification, Regression |
| Input | Images, reshaped tabular |
| Speed | ⚡ Medium |
| Interpretability | 🔴 Low |

</td>
</tr>
</table>

**Compression Methods:**
- L1 unstructured pruning
- INT8 dynamic quantization
- Knowledge distillation

---

### RNN/LSTM (Recurrent Neural Network)

<table>
<tr>
<td width="200">
<strong>Best For</strong><br/>
Sequences, text, time series
</td>
<td>

| Attribute | Value |
|-----------|-------|
| Framework | PyTorch |
| Tasks | Classification, Generation |
| Input | Text files, sequential CSV |
| Speed | ⚡ Medium-Slow |
| Interpretability | 🔴 Low |

</td>
</tr>
</table>

**Compression Methods:**
- Weight pruning
- INT8 quantization
- Knowledge distillation

---

## 📁 Dataset Formats

### CSV Files (Tabular Data)

**Structure Requirements:**
- Header row with column names
- Each row = one sample
- Last column = target variable
- Minimum 10 rows

**Example:**
```csv
feature_1,feature_2,feature_3,target
5.1,3.5,1.4,setosa
4.9,3.0,1.4,setosa
7.0,3.2,4.7,versicolor
```

---

### Text Files (For RNN)

**Option 1: Plain Text (Character Generation)**
```text
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune...
```

**Option 2: Tab-Separated (Classification)**
```text
positive	This movie was absolutely fantastic!
negative	Terrible experience, would not recommend.
neutral	It was okay, nothing special.
```

---

### Image Folders (For CNN)

**Option 1: Classified Structure**
```
my_dataset/
├── cats/
│   ├── cat_001.jpg
│   ├── cat_002.jpg
│   └── ...
├── dogs/
│   ├── dog_001.jpg
│   └── ...
```

**Option 2: Flat Folder**
```
my_images/
├── image_001.jpg
├── image_002.png
└── ...
```

**Supported Formats:** `.jpg`, `.jpeg`, `.png`, `.bmp`

---

## 📦 Compression Methods

### ✂️ Weight Pruning

Removes less important weights from the neural network.

| Parameter | Value |
|-----------|-------|
| Type | L1 Unstructured |
| Sparsity | 30-50% |
| Size Reduction | 20-40% |
| Accuracy Impact | < 2% |

---

### 🔢 INT8 Quantization

Reduces weight precision from 32-bit to 8-bit.

| Parameter | Value |
|-----------|-------|
| Type | Dynamic Quantization |
| Size Reduction | 50-75% |
| Accuracy Impact | < 1% |
| Speed Improvement | 2-4x |

---

### 📚 Knowledge Distillation

Trains a smaller "student" model to mimic the original "teacher".

| Parameter | Value |
|-----------|-------|
| Type | Soft-label Distillation |
| Size Reduction | 40-70% |
| Accuracy Impact | 1-5% |
| Benefit | Smaller, faster model |

---

## 🔌 API Reference

### Base URL
```
http://localhost:8000
```

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/dataset/upload` | POST | Upload dataset |
| `/api/dataset/list` | GET | List all datasets |
| `/api/dataset/preview/{name}` | GET | Preview dataset contents |
| `/api/model/select` | POST | Select model type |
| `/api/model/trained` | GET | Get training history |
| `/api/validation/validate` | POST | Validate dataset |
| `/api/training/start` | POST | Start training |
| `/api/training/status` | GET | Get training status |
| `/api/training/stop` | POST | Stop training |
| `/api/compression/compress` | POST | Compress model |
| `/api/comparison/compare` | GET | Get model comparison |

### Interactive Documentation

📖 **Swagger UI:** http://localhost:8000/docs  
📖 **ReDoc:** http://localhost:8000/redoc

---

## 📂 Project Structure

```
Model-compression/
│
├── 📄 main.py                    # Application entry point
├── 📄 requirements.txt           # Python dependencies
├── 📄 README.md                  # This file
│
├── 📁 routers/                   # API Route Handlers
│   ├── dataset.py                # Dataset upload/list/preview
│   ├── model.py                  # Model selection/history
│   ├── validation.py             # Dataset validation
│   ├── training.py               # Model training
│   ├── compression.py            # Model compression
│   ├── evaluation.py             # Model evaluation
│   └── comparison.py             # Model comparison
│
├── 📁 services/                  # Business Logic
│   ├── training_service.py       # Training orchestration
│   ├── compression_service.py    # Compression algorithms
│   ├── evaluation_service.py     # Metric calculation
│   ├── universal_dataset_normalizer.py  # Data preprocessing
│   ├── dataset_validation_service.py    # Data validation
│   └── preprocessing_service.py  # Feature engineering
│
├── 📁 utils/                     # Utility Functions
│   ├── model_builder.py          # Model architecture factory
│   ├── data_loader.py            # Data loading utilities
│   └── validation.py             # Validation helpers
│
├── 📁 frontend2/                 # React Frontend
│   ├── src/
│   │   ├── components/           # UI Components
│   │   │   ├── pages/            # Page components
│   │   │   ├── layout/           # Layout components
│   │   │   └── ui/               # Reusable UI elements
│   │   ├── api/                  # API client services
│   │   ├── hooks/                # Custom React hooks
│   │   ├── store/                # Zustand state management
│   │   └── types/                # TypeScript definitions
│   ├── package.json              # NPM dependencies
│   └── vite.config.ts            # Vite configuration
│
├── 📁 uploads/                   # Uploaded datasets (runtime)
├── 📁 models/                    # Saved models (runtime)
└── 📁 results/                   # Training logs (runtime)
```

---

## 🚀 Production Deployment

### Backend Deployment

```bash
# Run with multiple workers for production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Frontend Build

```bash
cd frontend2

# Create production build
npm run build

# Serve with any static server
npx serve dist -l 3000
```

### Docker Deployment (Optional)

**docker-compose.yml:**
```yaml
version: '3.8'
services:
  backend:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./uploads:/app/uploads
      - ./models:/app/models
      - ./results:/app/results
      
  frontend:
    build: ./frontend2
    ports:
      - "80:80"
    depends_on:
      - backend
```

### Nginx Configuration (Example)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Frontend
    location / {
        root /var/www/frontend/dist;
        try_files $uri /index.html;
    }

    # Backend API
    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # WebSocket support (if needed)
    location /ws {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

---

## 🔧 Troubleshooting

### Backend Issues

| Problem | Solution |
|---------|----------|
| **Port 8000 in use** | `uvicorn main:app --port 8001` |
| **Import errors** | Ensure venv is activated: `which python` |
| **Missing packages** | `pip install -r requirements.txt --force-reinstall` |
| **Permission denied** | Run as administrator or check folder permissions |

### Frontend Issues

| Problem | Solution |
|---------|----------|
| **Node modules issues** | `rm -rf node_modules && npm install` |
| **Build errors** | `npm cache clean --force && npm install` |
| **Port 5173 in use** | `npm run dev -- --port 3000` |
| **TypeScript errors** | `npm run typecheck` to see details |

### Training Issues

| Problem | Solution |
|---------|----------|
| **Out of memory** | Reduce batch size to 16 or 8 |
| **Training stuck** | Check backend terminal for errors |
| **NaN loss** | Reduce learning rate or check data |
| **Low accuracy** | Try more epochs or different model |

### Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| "Backend Disconnected" | Server not running | Start backend with `python main.py` |
| "Dataset validation failed" | Invalid data format | Check dataset format requirements |
| "Model not found" | No trained model | Complete training first |
| "Target out of bounds" | Label encoding issue | Ensure labels are 0-indexed |

---

## ❓ FAQ

<details>
<summary><strong>Q: What Python version do I need?</strong></summary>

Python 3.10 or higher is required. Check with `python3 --version`.
</details>

<details>
<summary><strong>Q: Can I use GPU for training?</strong></summary>

Yes! If you have an NVIDIA GPU with CUDA installed, PyTorch will automatically use it. No configuration needed.
</details>

<details>
<summary><strong>Q: What's the maximum dataset size?</strong></summary>

There's no hard limit, but for best performance:
- CSV: Up to 100MB
- Images: Up to 10,000 images
- Text: Up to 10MB
</details>

<details>
<summary><strong>Q: Can I export models for mobile deployment?</strong></summary>

Yes! Compressed models (especially quantized) are optimized for edge deployment. Download the `.pt` file and convert using PyTorch Mobile or ONNX.
</details>

<details>
<summary><strong>Q: How do I add custom models?</strong></summary>

Add your model architecture to `utils/model_builder.py` and register it in the model selection service.
</details>

<details>
<summary><strong>Q: Is my data stored securely?</strong></summary>

All data is stored locally on your server. No data is sent to external services.
</details>

---

## 📊 System Requirements Summary

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Windows 10, macOS 10.15, Linux | Latest versions |
| **Python** | 3.10 | 3.11+ |
| **Node.js** | 18.0 | 20.0+ |
| **RAM** | 8 GB | 16 GB |
| **Storage** | 2 GB | 5 GB |
| **GPU** | None | NVIDIA with CUDA |

---

## 📧 Support

For issues, questions, or feature requests:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Review the [FAQ](#-faq)
3. Check existing issues in the repository
4. Contact the development team

---

## 📜 License

This project is provided under the MIT License. See LICENSE file for details.

---

<div align="center">

**Built with ❤️ using Python, FastAPI, React, and PyTorch**

</div>
