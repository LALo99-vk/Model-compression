# Manual Testing Commands for ML Compression Backend

## Quick Start Guide

### 1. Start the Server
```bash
# Terminal 1 - Start the backend server
cd /Users/test/PycharmProjects/Model-compression
source .venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Create Test Data
```bash
# Terminal 2 - Create real datasets
cd /Users/test/PycharmProjects/Model-compression
source .venv/bin/activate
python create_real_dataset.py --type iris
```

### 3. Run Complete Test
```bash
# Terminal 2 - Run the complete test suite
python demo_test_script.py --dataset real_iris_dataset.csv
```

---

## Step-by-Step Manual Testing with curl

### Step 1: Health Check
```bash
curl http://localhost:8000/health
```
**Expected:** `{"status": "healthy"}`

### Step 2: Upload Dataset
```bash
curl -X POST "http://localhost:8000/api/upload/" \
  -F "file=@real_iris_dataset.csv"
```

### Step 3: List Available Models
```bash
curl http://localhost:8000/api/models/
```

### Step 4: Select Model
```bash
curl -X POST "http://localhost:8000/api/select/" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "decision_tree",
    "task_type": "classification",
    "config": {
      "max_depth": 10,
      "min_samples_split": 2,
      "min_samples_leaf": 1,
      "criterion": "gini"
    }
  }'
```

### Step 5: Start Training
```bash
curl -X POST "http://localhost:8000/api/train/" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_path": "uploads/real_iris_dataset.csv",
    "epochs": 5,
    "batch_size": 32
  }'
```

### Step 6: Check Training Status
```bash
curl http://localhost:8000/api/train/status
```
**Keep running until status shows "completed"**

### Step 7: Get Training Logs
```bash
curl http://localhost:8000/api/train/logs
```

### Step 8: Evaluate Original Model
```bash
curl -X POST "http://localhost:8000/api/evaluation/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "original",
    "dataset_path": "uploads/real_iris_dataset.csv"
  }'
```

### Step 9: Get Compression Methods
```bash
curl http://localhost:8000/api/compression/methods
```

### Step 10: Compress Model (Pruning)
```bash
curl -X POST "http://localhost:8000/api/compression/compress" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "pruning",
    "pruning_amount": 0.3
  }'
```

### Step 11: Evaluate Compressed Model
```bash
curl -X POST "http://localhost:8000/api/evaluation/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "compressed",
    "dataset_path": "uploads/real_iris_dataset.csv"
  }'
```

### Step 12: Compare Models
```bash
curl http://localhost:8000/api/evaluation/compare
```

### Step 13: Get All Metrics
```bash
curl http://localhost:8000/api/evaluation/metrics
```

---

## Advanced Testing Scenarios

### Test Different Compression Methods

#### Quantization
```bash
curl -X POST "http://localhost:8000/api/compression/compress" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "quantization",
    "quantization_bits": 8
  }'
```

#### Knowledge Distillation
```bash
curl -X POST "http://localhost:8000/api/compression/compress" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "distillation",
    "distillation_temperature": 3.0,
    "distillation_alpha": 0.5
  }'
```

### Test Different Datasets

#### Create Complex Classification Dataset
```bash
python create_real_dataset.py --type complex
```

#### Upload and Test Complex Dataset
```bash
curl -X POST "http://localhost:8000/api/upload/" \
  -F "file=@complex_classification_dataset.csv"

# Then repeat steps 4-12 with the new dataset
```

### Test Different Models

#### CNN Model (for image-like data)
```bash
curl -X POST "http://localhost:8000/api/select/" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "cnn",
    "task_type": "classification",
    "input_shape": [64, 64, 3],
    "num_classes": 3
  }'
```

#### RNN Model (for sequence data)
```bash
curl -X POST "http://localhost:8000/api/select/" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "rnn",
    "task_type": "classification",
    "input_shape": [50, 10],
    "num_classes": 3
  }'
```

---

## Monitoring and Debugging

### Check Server Logs
The server running in Terminal 1 will show all request logs and error messages.

### Check File System
```bash
# List uploaded datasets
ls -la uploads/

# List trained models
ls -la models/

# List results and metrics
ls -la results/

# Check compression info
cat results/compression_info.json
```

### Test Individual Endpoints
```bash
# Test health endpoint
curl -v http://localhost:8000/health

# Test model listing
curl -v http://localhost:8000/api/models/

# Test compression methods
curl -v http://localhost:8000/api/compression/methods
```

---

## Expected Results

### Successful Test Output
When everything works correctly, you should see:

1. **Health Check:** Status 200, `{"status": "healthy"}`
2. **Upload:** Status 200, file uploaded successfully
3. **Training:** Status 200, training starts and completes
4. **Evaluation:** Status 200, metrics for both original and compressed models
5. **Compression:** Status 200, compression method applied
6. **Comparison:** Status 200, side-by-side model comparison

### Typical Metrics for Iris Dataset
- **Accuracy:** 0.95+ for both original and compressed
- **File Size:** ~25KB (same for sklearn models)
- **Inference Time:** <1ms

### Common Issues and Solutions

#### Server Not Responding
```bash
# Check if server is running
ps aux | grep uvicorn

# Kill existing server if needed
pkill -f uvicorn

# Restart server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### File Upload Errors
```bash
# Check file exists
ls -la real_iris_dataset.csv

# Check uploads directory
mkdir -p uploads
chmod 755 uploads
```

#### Model Not Found Errors
```bash
# Check models directory
ls -la models/

# Check if training completed
curl http://localhost:8000/api/train/status
```

---

## Performance Testing

### Load Testing
```bash
# Install Apache Bench (if not installed)
brew install ab  # Mac

# Test health endpoint under load
ab -n 1000 -c 10 http://localhost:8000/health

# Test model evaluation under load
ab -n 100 -c 5 -p test_payload.json -T application/json \
  http://localhost:8000/api/evaluation/evaluate
```

### Memory Usage
```bash
# Monitor memory during training
top -p $(pgrep -f uvicorn)

# Or use htop
htop
```

---

## Automation Script

Create a simple test script:
```bash
#!/bin/bash
# simple_test.sh

echo "🧪 Starting ML Backend Test"

# Health check
echo "1. Health check..."
curl -s http://localhost:8000/health | jq .

# Upload dataset
echo "2. Upload dataset..."
curl -s -X POST "http://localhost:8000/api/upload/" \
  -F "file=@real_iris_dataset.csv" | jq .

# List models
echo "3. List models..."
curl -s http://localhost:8000/api/models/ | jq .

echo "✅ Basic tests complete!"
```

Make it executable:
```bash
chmod +x simple_test.sh
./simple_test.sh
```

---

## Next Steps

Once you're comfortable with basic testing:

1. **Try larger datasets** - Test with 10K+ samples
2. **Test edge cases** - Empty files, malformed CSVs, etc.
3. **Performance benchmarking** - Measure training time and memory usage
4. **Integration testing** - Test with your own datasets
5. **API documentation** - Visit `http://localhost:8000/docs` for interactive API docs

Remember to keep the server running in one terminal while running tests in another!
