"""
Model Builder - Constructs models based on configuration
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Simple CNN for image classification"""

    def __init__(self, input_channels, num_classes, filters, kernel_size, pool_size, dense_units, dropout):
        super(SimpleCNN, self).__init__()

        layers = []
        in_channels = input_channels

        for out_channels in filters:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(pool_size)
            ])
            in_channels = out_channels

        self.features = nn.Sequential(*layers)

        # Calculate flattened size (assuming 32x32 input)
        # This is a simplification; in production, calculate dynamically
        self.flatten_size = filters[-1] * (32 // (pool_size ** len(filters))) ** 2

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
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
        """Build PyTorch model from config"""
        model_type = model_config['model_type']
        config = model_config['config']

        if model_type == 'cnn':
            # Determine input channels
            input_shape = model_config.get('input_shape', (3, 32, 32))
            if isinstance(input_shape, tuple) and len(input_shape) >= 3:
                input_channels = input_shape[0]
            else:
                input_channels = 3

            num_classes = model_config.get('num_classes', 10)

            model = SimpleCNN(
                input_channels=input_channels,
                num_classes=num_classes,
                filters=config.get('filters', [32, 64, 128]),
                kernel_size=config.get('kernel_size', 3),
                pool_size=config.get('pool_size', 2),
                dense_units=config.get('dense_units', 128),
                dropout=config.get('dropout', 0.5)
            )

        elif model_type == 'rnn':
            input_shape = model_config.get('input_shape', (10,))  # sequence length
            if isinstance(input_shape, tuple):
                input_size = input_shape[-1] if len(input_shape) > 1 else 1
            else:
                input_size = 1

            num_classes = model_config.get('num_classes', 10)

            model = SimpleRNN(
                input_size=input_size,
                hidden_size=config.get('hidden_size', 128),
                num_layers=config.get('num_layers', 2),
                num_classes=num_classes,
                dropout=config.get('dropout', 0.3),
                bidirectional=config.get('bidirectional', True),
                rnn_type=config.get('rnn_type', 'LSTM')
            )

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return model