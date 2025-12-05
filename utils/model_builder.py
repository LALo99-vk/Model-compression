"""
Model Builder - Constructs models based on configuration
"""

import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleCNN(nn.Module):
    """Simple CNN for image classification"""

    def __init__(self, input_shape, num_classes, conv_layers=None, dense_units=128):
        super(SimpleCNN, self).__init__()
        
        # Ensure input_shape is a tuple of integers
        if isinstance(input_shape, list):
            input_shape = tuple(input_shape)
        if not isinstance(input_shape, tuple) or len(input_shape) != 3:
            raise ValueError(f"input_shape must be a tuple of (channels, height, width), got {input_shape}")
            
        # Convert to integers to avoid any type issues
        input_channels = int(input_shape[0])
        num_classes = int(num_classes)
        dense_units = int(dense_units)
        
        # Determine pooling strategy based on input size
        # For small inputs, use adaptive pooling to avoid dimension becoming 0
        height, width = input_shape[1], input_shape[2]
        
        # Calculate expected output sizes after standard pooling
        # After first pool: (height//2, width//2), After second pool: (height//4, width//4)
        # Use adaptive pooling if standard pooling would result in 0-sized dimensions
        if height < 4 or width < 4 or (height // 4 == 0) or (width // 4 == 0):
            # Use adaptive pooling to ensure we always have valid dimensions
            # Target sizes: at least 1x1 after each pooling stage
            target_h1 = max(1, (height + 1) // 2)  # Round up to avoid 0
            target_w1 = max(1, (width + 1) // 2)
            target_h2 = max(1, (height + 3) // 4)  # Round up to avoid 0
            target_w2 = max(1, (width + 3) // 4)
            
            self.features = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((target_h1, target_w1)),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((target_h2, target_w2)),
            )
        else:
            # Use standard pooling for larger inputs
            self.features = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        
        # Calculate the size of the flattened features
        with torch.no_grad():
            self.feature_size = self._get_conv_output(input_shape)
            
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(dense_units, num_classes)
        )

    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output = self.features(input)
        n_size = output.data.view(batch_size, -1).size(1)
        return n_size

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class SimpleRNN(nn.Module):
    """Simple RNN/LSTM for sequence data"""

    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout, bidirectional, rnn_type='LSTM'):
        super(SimpleRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size, hidden_size, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        else:
            self.rnn = nn.GRU(
                input_size, hidden_size, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )

        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.rnn(x)
        # Take the last output
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class ModelBuilder:
    """Build models based on configuration"""

    def build_pytorch_model(self, model_config):
        # Determine model type and configuration
        model_type = model_config.get("model_type")
        config = model_config.get("config", {})

        if model_type == "cnn":
            # Ensure required parameters exist
            if "input_shape" not in model_config:
                raise ValueError("input_shape is required in model_config for cnn models")
            
            # Get and validate parameters
            input_shape = model_config["input_shape"]
            num_classes = model_config.get("num_classes")
            dense_units = int(config.get("dense_units", 128))

            # Convert input_shape to tuple if it's a list
            if isinstance(input_shape, list):
                input_shape = tuple(int(x) for x in input_shape)

            # If input_shape is (H, W, C) from the frontend, convert to (C, H, W)
            if (
                isinstance(input_shape, tuple)
                and len(input_shape) == 3
                and input_shape[0] not in (1, 3)
                and input_shape[2] in (1, 3)
            ):
                input_shape = (input_shape[2], input_shape[0], input_shape[1])

            # Ensure num_classes is set and is an integer
            if num_classes is None:
                # For regression, default to 1 output
                task_type = model_config.get('task_type', 'classification')
                if task_type == 'regression':
                    num_classes = 1
                    logger.info(f"num_classes not set for regression, using default: {num_classes}")
                else:
                    raise ValueError(
                        "num_classes is required in model_config for cnn models. "
                        "For classification, please ensure num_classes is set. "
                        "For regression, it will default to 1."
                    )
            num_classes = int(num_classes)

            model = SimpleCNN(
                input_shape=input_shape,
                num_classes=num_classes,
                dense_units=dense_units,
            )

        elif model_type == "rnn":
            # Get and validate parameters
            input_shape = model_config.get("input_shape", (10,))
            
            # Handle both list and tuple formats
            # input_shape can be [seq_len, input_size] or (seq_len, input_size) or just input_size
            if isinstance(input_shape, (list, tuple)):
                # If it's a sequence, extract the input_size (last element)
                input_size = int(input_shape[-1]) if len(input_shape) > 0 else 1
            elif isinstance(input_shape, (int, float)):
                # If it's a single number, use it as input_size
                input_size = int(input_shape)
            else:
                # Default fallback
                input_size = 1

            # For RNN, default to 1 for regression, infer from config for classification
            task_type = model_config.get('task_type', 'classification')
            if model_config.get("num_classes") is None:
                if task_type == 'regression':
                    num_classes = 1
                else:
                    num_classes = 1  # Default, but should be set by auto_infer
            else:
                num_classes = int(model_config.get("num_classes", 1))

            model = SimpleRNN(
                input_size=input_size,
                hidden_size=int(config.get("hidden_size", 128)),
                num_layers=int(config.get("num_layers", 2)),
                num_classes=num_classes,
                dropout=float(config.get("dropout", 0.3)),
                bidirectional=bool(config.get("bidirectional", True)),
                rnn_type=config.get("rnn_type", "LSTM"),
            )

        else:
            raise ValueError(f"Unknown model type for PyTorch: {model_type}")

        return model