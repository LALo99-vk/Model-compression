"""
Test Client for FastAPI ML Backend
Demonstrates complete workflow: upload -> select -> train -> evaluate -> compress -> compare
"""

import requests
import json
import time
import pandas as pd
import numpy as np
from io import StringIO

BASE_URL = "http://localhost:8000"


def print_section(title):
    """Print section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def create_sample_dataset():
    """Create a sample CSV dataset for testing"""
    np.random.seed(42)

    # Create synthetic classification data
    n_samples = 1000
    n_features = 10
    n_classes = 3

    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)

    # Create DataFrame
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['target'] = y

    # Save to CSV
    df.to_csv('sample_dataset.csv', index=False)
    print(f"✅ Created sample dataset: sample_dataset.csv")
    print(f"   Samples: {n_samples}, Features: {n_features}, Classes: {n_classes}")


def test_health():
    """Test health endpoint"""
    print_section("1. Health Check")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_upload_dataset():
    """Test dataset upload"""
    print_section("2. Upload Dataset")

    with open('sample_dataset.csv', 'rb') as f:
        files = {'files': ('sample_dataset.csv', f, 'text/csv')}
        response = requests.post(f"{BASE_URL}/api/dataset/upload", files=files)

    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    return response.json()['files'][0]['path']


def test_list_datasets():
    """Test list datasets"""
    print_section("3. List Datasets")
    response = requests.get(f"{BASE_URL}/api/dataset/list")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_available_models():
    """Test available models"""
    print_section("4. Available Models")
    response = requests.get(f"{BASE_URL}/api/model/available")
    print(f"Status: {response.status_code}")
    print(f"Available Models:")
    for model_name, model_info in response.json()['models'].items():
        print(f"  - {model_name}: {model_info['name']}")


def test_select_model():
    """Test model selection"""
    print_section("5. Select Model")

    data = {
        "model_type": "decision_tree",
        "task_type": "classification",
        "num_classes": 3,
        "config": {
            "max_depth": 10,
            "min_samples_split": 2
        }
    }

    response = requests.post(f"{BASE_URL}/api/model/select", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_start_training(dataset_path):
    """Test training start"""
    print_section("6. Start Training")

    data = {
        "dataset_path": dataset_path,
        "epochs": 5,
        "batch_size": 32,
        "validation_split": 0.2
    }

    response = requests.post(f"{BASE_URL}/api/training/start", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_training_status():
    """Test training status"""
    print_section("7. Check Training Status")

    # Poll status until training is complete
    max_attempts = 30
    for i in range(max_attempts):
        response = requests.get(f"{BASE_URL}/api/training/status")
        status = response.json()

        print(f"Attempt {i + 1}/{max_attempts}: {status['status']}")

        if status['status'] in ['completed', 'error']:
            print(f"\nFinal Status: {json.dumps(status, indent=2)}")
            break

        time.sleep(2)


def test_training_logs():
    """Test training logs"""
    print_section("8. Training Logs")
    response = requests.get(f"{BASE_URL}/api/training/logs")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_evaluate_original(dataset_path):
    """Test original model evaluation"""
    print_section("9. Evaluate Original Model")

    data = {
        "model_type": "original",
        "dataset_path": dataset_path
    }

    response = requests.post(f"{BASE_URL}/api/evaluation/evaluate", json=data)
    print(f"Status: {response.status_code}")
    print(f"Metrics:")
    metrics = response.json()['metrics']
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")
    print(f"  Inference Time: {metrics['inference_time'] * 1000:.2f} ms")


def test_compression_methods():
    """Test available compression methods"""
    print_section("10. Compression Methods")
    response = requests.get(f"{BASE_URL}/api/compression/methods")
    print(f"Status: {response.status_code}")
    print("Available Methods:")
    for method_name, method_info in response.json()['methods'].items():
        print(f"  - {method_name}: {method_info['name']}")


def test_compress_model():
    """Test model compression"""
    print_section("11. Compress Model (Pruning)")

    data = {
        "method": "pruning",
        "pruning_amount": 0.3
    }

    response = requests.post(f"{BASE_URL}/api/compression/compress", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_evaluate_compressed(dataset_path):
    """Test compressed model evaluation"""
    print_section("12. Evaluate Compressed Model")

    data = {
        "model_type": "compressed",
        "dataset_path": dataset_path
    }

    response = requests.post(f"{BASE_URL}/api/evaluation/evaluate", json=data)
    print(f"Status: {response.status_code}")
    print(f"Metrics:")
    metrics = response.json()['metrics']
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")
    print(f"  Inference Time: {metrics['inference_time'] * 1000:.2f} ms")


def test_comparison():
    """Test model comparison"""
    print_section("13. Compare Models")
    response = requests.get(f"{BASE_URL}/api/comparison/compare")
    print(f"Status: {response.status_code}")

    comparison = response.json()

    print("\n📊 Comparison Results:")
    print(f"\n  File Size:")
    print(f"    Original: {comparison['file_size']['original_mb']:.2f} MB")
    print(f"    Compressed: {comparison['file_size']['compressed_mb']:.2f} MB")
    print(f"    Reduction: {comparison['file_size']['reduction_percent']:.2f}%")

    print(f"\n  Accuracy:")
    print(f"    Original: {comparison['accuracy']['original']:.4f}")
    print(f"    Compressed: {comparison['accuracy']['compressed']:.4f}")
    print(f"    Change: {comparison['accuracy']['difference_percent']:+.2f}%")

    print(f"\n  Inference Speed:")
    print(f"    Original: {comparison['inference_time']['original_ms']:.2f} ms")
    print(f"    Compressed: {comparison['inference_time']['compressed_ms']:.2f} ms")
    print(f"    Speedup: {comparison['inference_time']['speedup']:.2f}x")


def test_comparison_table():
    """Test comparison table"""
    print_section("14. Comparison Table")
    response = requests.get(f"{BASE_URL}/api/comparison/table")

    table = response.json()

    print(f"\n{' | '.join(table['headers'])}")
    print("-" * 80)
    for row in table['rows']:
        print(f"{row[0]:20} | {str(row[1]):15} | {str(row[2]):15} | {row[3]}")


def run_full_workflow():
    """Run complete ML workflow"""
    try:
        # Create sample dataset
        create_sample_dataset()

        # Run workflow
        test_health()
        dataset_path = test_upload_dataset()
        test_list_datasets()
        test_available_models()
        test_select_model()
        test_start_training(dataset_path)
        test_training_status()
        test_training_logs()
        test_evaluate_original(dataset_path)
        test_compression_methods()
        test_compress_model()
        test_evaluate_compressed(dataset_path)
        test_comparison()
        test_comparison_table()

        print_section("✅ All Tests Completed Successfully!")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🧪 FastAPI ML Backend Test Client")
    print("Make sure the server is running at http://localhost:8000")
    print("\nStarting tests in 3 seconds...")
    time.sleep(3)

    run_full_workflow()