# Model Compression Backend Testing Guide

## Overview
This guide will teach you how to properly test the ML model compression backend with real datasets.

## Prerequisites
- Python 3.8+
- Virtual environment activated
- All dependencies installed (`pip install -r requirements.txt`)

## Step 1: Start the Backend Server

### Option A: Using uvicorn (recommended)
```bash
cd /Users/test/PycharmProjects/Model-compression
source .venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Option B: Using python directly
```bash
cd /Users/test/PycharmProjects/Model-compression
source .venv/bin/activate
python main.py
```

**Expected Output:**
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## Step 2: Verify Server Health

```bash
curl http://localhost:8000/health
```

**Expected Response:**
```json
{"status": "healthy"}
```

## Step 3: Prepare Real Dataset

### Option A: Use a classic dataset (Iris)
```python
# Create a real dataset script
import pandas as pd
from sklearn.datasets import load_iris
import numpy as np

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Save to CSV
df.to_csv('real_iris_dataset.csv', index=False)
print(f"Dataset saved with {len(df)} samples and {df.shape[1]-1} features")
```

### Option B: Use a synthetic but realistic dataset
```python
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Create a more complex dataset
X, y = make_classification(
    n_samples=5000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=4,
    random_state=42
)

# Create feature names
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Save to CSV
df.to_csv('real_complex_dataset.csv', index=False)
print(f"Complex dataset saved with {len(df)} samples and {df.shape[1]-1} features")
```

## Step 4: Upload Dataset

```bash
curl -X POST "http://localhost:8000/api/upload/" \
  -F "file=@real_iris_dataset.csv"
```

**Expected Response:**
```json
{
  "message": "Files uploaded successfully",
  "files": [
    {
      "filename": "real_iris_dataset.csv",
      "size": 12345,
      "path": "uploads/real_iris_dataset.csv"
    }
  ],
  "count": 1
}
```

## Step 5: List Available Models

```bash
curl http://localhost:8000/api/models/
```

**Expected Response:**
```json
{
  "models": {
    "cnn": "Convolutional Neural Network",
    "rnn": "Recurrent Neural Network", 
    "decision_tree": "Decision Tree"
  }
}
```

## Step 6: Select and Configure Model

```bash
curl -X POST "http://localhost:8000/api/select/" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "decision_tree",
    "task_type": "classification",
    "config": {
      "max_depth": 15,
      "min_samples_split": 5,
      "min_samples_leaf": 2,
      "criterion": "gini"
    }
  }'
```

## Step 7: Start Training

```bash
curl -X POST "http://localhost:8000/api/train/" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_path": "uploads/real_iris_dataset.csv",
    "epochs": 10,
    "batch_size": 32
  }'
```

**Expected Response:**
```json
{
  "message": "Training started",
  "model_type": "decision_tree",
  "dataset": "uploads/real_iris_dataset.csv",
  "epochs": 10,
  "status": "training"
}
```

## Step 8: Monitor Training Progress

```bash
curl http://localhost:8000/api/train/status
```

**Keep checking until status is "completed":**
```json
{
  "status": "completed",
  "current_epoch": 10,
  "total_epochs": 10,
  "message": "",
  "timestamp": 1234567890.123
}
```

## Step 9: Get Training Results

```bash
curl http://localhost:8000/api/train/logs
```

## Step 10: Evaluate Original Model

```bash
curl -X POST "http://localhost:8000/api/evaluation/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "original",
    "dataset_path": "uploads/real_iris_dataset.csv"
  }'
```

## Step 11: Apply Compression

### Pruning
```bash
curl -X POST "http://localhost:8000/api/compression/compress" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "pruning",
    "pruning_amount": 0.3
  }'
```

### Quantization
```bash
curl -X POST "http://localhost:8000/api/compression/compress" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "quantization",
    "quantization_bits": 8
  }'
```

### Knowledge Distillation
```bash
curl -X POST "http://localhost:8000/api/compression/compress" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "distillation",
    "distillation_temperature": 3.0,
    "distillation_alpha": 0.5
  }'
```

## Step 12: Evaluate Compressed Model

```bash
curl -X POST "http://localhost:8000/api/evaluation/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "compressed",
    "dataset_path": "uploads/real_iris_dataset.csv"
  }'
```

## Step 13: Compare Models

```bash
curl http://localhost:8000/api/evaluation/compare
```

## Step 14: View All Metrics

```bash
curl http://localhost:8000/api/evaluation/metrics
```

## Advanced Testing

### Test with Different Models
Try different model types to see how they perform:

```bash
# For CNN (requires image-like data)
curl -X POST "http://localhost:8000/api/select/" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "cnn",
    "task_type": "classification",
    "input_shape": [64, 64, 3],
    "num_classes": 4
  }'
```

### Test Different Datasets
Create datasets with different characteristics:
- More features
- More classes  
- Larger sample sizes
- Different data distributions

### Performance Testing
Monitor system resources during training and compression:
```bash
# In another terminal
htop  # or Activity Monitor on Mac
```

## Troubleshooting

### Common Issues

1. **Server not responding**
   - Check if server is running: `ps aux | grep uvicorn`
   - Verify port: `lsof -i :8000`

2. **File upload errors**
   - Check file format (must be CSV)
   - Verify file permissions
   - Check uploads directory exists

3. **Training errors**
   - Verify dataset has target column named 'target'
   - Check for missing values in dataset
   - Ensure dataset has enough samples

4. **Model not found errors**
   - Check models directory
   - Verify training completed successfully
   - Look for model files: `ls -la models/`

### Debug Tips

1. **Check server logs** - Look at the terminal where server is running
2. **Verify file existence** - `ls -la uploads/ models/ results/`
3. **Test individual endpoints** - Start with health check, work sequentially
4. **Use the test client** - `python test_client.py` for comprehensive testing

## Expected Results

For a successful test, you should see:
- ✅ Server starts without errors
- ✅ Dataset uploads successfully  
- ✅ Training completes with reasonable accuracy
- ✅ Compression methods work (even if just copying for sklearn models)
- ✅ Evaluation shows metrics for both original and compressed models
- ✅ Comparison shows model performance and size differences

## Performance Benchmarks

Typical results for different datasets:

| Dataset Type | Model | Original Accuracy | Compressed Accuracy | Size Reduction |
|-------------|-------|-------------------|---------------------|----------------|
| Iris (150 samples, 4 features) | Decision Tree | 0.95+ | 0.95+ | 0% (sklearn) |
| Synthetic (5000 samples, 20 features) | Decision Tree | 0.85+ | 0.85+ | 0% (sklearn) |
| Large dataset (10k+ samples) | CNN/PyTorch | 0.90+ | 0.88-0.90 | 30-70% |

## Next Steps

Once you're comfortable with the basic testing:
1. Try larger, real-world datasets
2. Experiment with different model configurations
3. Test edge cases (empty datasets, malformed files, etc.)
4. Monitor performance metrics and system resource usage
5. Integrate with your own datasets and use cases
