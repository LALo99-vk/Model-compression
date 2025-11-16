Backend Setup

Language: Python; framework: FastAPI with uvicorn (main.py entrypoint at Model-compression/main.py:51-53)
Storage: local directories uploads/, models/, results created at startup (Model-compression/main.py:11-15)
Default API URL: http://localhost:8000
No database or .env required; CORS allows all origins
Prerequisites

Install Python 3.10+ and pip
macOS/Linux: ensure python3 and pip commands are available
Windows: ensure python and pip are in PATH
Optional: Node.js 18+ if you will run the frontend later
Get the Code

git clone <repo-url>
cd Model-compression
Create Virtual Environment

macOS/Linux: python3 -m venv .venv && source .venv/bin/activate
Windows (PowerShell): python -m venv .venv ; .\.venv\Scripts\Activate.ps1
Windows (CMD): python -m venv .venv && .\.venv\Scripts\activate.bat
Install Python Dependencies

pip install --upgrade pip
pip install -r requirements.txt
If build wheels fail on torch/tensorflow, prefer CPU-only environments or install system build tools
Start the Backend (Development)

uvicorn main:app --reload --host 0.0.0.0 --port 8000
Alternative: python main.py (runs uvicorn with reload; Model-compression/main.py:51-53)
Visit http://localhost:8000/docs for interactive API docs
Verify API Works

Health check: curl http://localhost:8000/health → {"status":"healthy"}
Root: curl http://localhost:8000/ → contains docs link and status: "running"
Run Full Workflow Test (Optional)

Ensure server is running at http://localhost:8000
python test_client.py
The script exercises upload → select model → train → evaluate → compress → compare (Model-compression/test_client.py:279-285)
Production-Like Run (Optional)

uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
Behind reverse proxy: configure Nginx/Apache to forward to localhost:8000
Bind IPs/ports per your environment; open firewall port 8000 if needed
Data Expectations

CSV upload: last column is target; non-numeric targets are label-encoded
Image datasets: classes inferred from parent folder names; auto-resized to 32x32
GPU vs CPU Notes

CPU-only: dependencies in requirements.txt work out of the box
GPU: install torch and tensorflow versions matching your CUDA; verify with python -c "import torch; print(torch.cuda.is_available())"
Windows-Specific Notes

If Activate.ps1 is blocked: run Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
Use curl.exe or Invoke-WebRequest for HTTP checks if curl is unavailable
Troubleshooting

Port already in use: change --port to a free port (then update frontend if used)
Import errors: confirm venv is activated (which python should point to .venv)
SSL/HTTPS: terminate TLS at a reverse proxy; uvicorn typically serves HTTP locally
CORS: default allows *; restrict allow_origins in main.py for production (Model-compression/main.py:22-29)
Key Files

Backend entrypoint: main.py (routes included at Model-compression/main.py:31-37)
Routers: routers/*.py (dataset, model, training, evaluation, compression, comparison)
Requirements: requirements.txt
Test client: test_client.py