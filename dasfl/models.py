"""
DAS-FL Project — Model Definitions
Defines the MLP architecture used for:
    - Scenario 3: Local MLP (each building trains independently)
    - Scenario 4: FedAvg (FL with Flower)
    - Scenario 5: FedProx (FL with proximal term)
    - Scenario 6: Personalised FL

The model is intentionally kept lightweight for FL efficiency —
small parameter count means fast weight transfer between
buildings and server.

Architecture choice follows thesis but extended:
    - Thesis: 2 hidden layers, 2 input features (kwh, m3h)
    - For paper: 3 hidden layers, 51 input features, dropout

Usage:
    from dasfl.models import create_mlp, compile_mlp
"""

import tensorflow as tf
from tensorflow import keras
from typing import Optional


def create_mlp(
    n_features: int,
    hidden_layers: list[int] = None,
    dropout_rate: float = 0.2,
    activation: str = "relu",
    output_dim: int = 1,
    name: str = "dh_mlp",
) -> keras.Model:
    """Create an MLP model for heat demand prediction.

    Architecture:
        Input (n_features)
        → Dense(128) + ReLU + Dropout
        → Dense(64) + ReLU + Dropout
        → Dense(32) + ReLU
        → Dense(1) — linear output (regression)

    Args:
        n_features: Number of input features.
        hidden_layers: List of hidden layer sizes.
            Default: [128, 64, 32]
        dropout_rate: Dropout rate between hidden layers.
            Set to 0.0 to disable dropout.
        activation: Activation function for hidden layers.
        output_dim: Output dimension (1 for single-step prediction).
        name: Model name.

    Returns:
        Uncompiled Keras Sequential model.
    """
    if hidden_layers is None:
        hidden_layers = [128, 64, 32]

    model = keras.Sequential(name=name)

    # Input layer
    model.add(keras.layers.InputLayer(shape=(n_features,)))

    # Hidden layers
    for i, units in enumerate(hidden_layers):
        model.add(keras.layers.Dense(
            units,
            activation=activation,
            kernel_initializer="he_normal",
            name=f"dense_{i}",
        ))
        # Add dropout after all layers except the last hidden layer
        if dropout_rate > 0 and i < len(hidden_layers) - 1:
            model.add(keras.layers.Dropout(dropout_rate, name=f"dropout_{i}"))

    # Output layer — linear activation for regression
    model.add(keras.layers.Dense(
        output_dim,
        activation="linear",
        name="output",
    ))

    return model


def compile_mlp(
    model: keras.Model,
    learning_rate: float = 0.001,
    loss: str = "mse",
) -> keras.Model:
    """Compile the MLP model with optimizer and loss.

    Args:
        model: Uncompiled Keras model.
        learning_rate: Learning rate for Adam optimizer.
        loss: Loss function ('mse' or 'mae').

    Returns:
        Compiled Keras model.
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["mae"],
    )

    return model


def get_model_summary(model: keras.Model) -> dict:
    """Get model parameter count and architecture summary.

    Args:
        model: Keras model.

    Returns:
        Dict with total_params, trainable_params, layer info.
    """
    total_params = model.count_params()
    trainable_params = sum(
        tf.keras.backend.count_params(w)
        for w in model.trainable_weights
    )

    layers = []
    for layer in model.layers:
        layers.append({
            "name": layer.name,
            "type": layer.__class__.__name__,
            "output_shape": layer.output_shape if hasattr(layer, "output_shape") else None,
            "params": layer.count_params(),
        })

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "layers": layers,
        "size_kb": total_params * 4 / 1024,  # float32 = 4 bytes
    }


# ── Convenience function for the full pipeline ──────────────

def build_and_compile(
    n_features: int,
    hidden_layers: list[int] = None,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
) -> keras.Model:
    """Create and compile an MLP in one call.

    This is the main entry point used by both local training
    and FL client code.

    Args:
        n_features: Number of input features.
        hidden_layers: Hidden layer sizes.
        dropout_rate: Dropout rate.
        learning_rate: Learning rate for Adam.

    Returns:
        Compiled Keras model ready for training.
    """
    model = create_mlp(
        n_features=n_features,
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
    )
    model = compile_mlp(model, learning_rate=learning_rate)
    return model
