"""
Neural network brain for agents using NumPy.
Simple feedforward network that evolves through genetic algorithms.
"""

import numpy as np


class Brain:
    """A simple feedforward neural network."""

    def __init__(self, input_size, hidden_size=32, output_size=2):
        """
        Initialize a neural network.

        Args:
            input_size: Number of input neurons
            hidden_size: Number of neurons in each hidden layer
            output_size: Number of output neurons
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights with Xavier initialization
        self.w1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)

        self.w2 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(hidden_size)

        self.w3 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b3 = np.zeros(output_size)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input vector (numpy array)

        Returns:
            Output vector (numpy array)
        """
        # Layer 1
        h1 = np.tanh(np.dot(x, self.w1) + self.b1)

        # Layer 2
        h2 = np.tanh(np.dot(h1, self.w2) + self.b2)

        # Output layer
        output = np.tanh(np.dot(h2, self.w3) + self.b3)

        return output

    def mutate(self, mutation_rate=0.1):
        """
        Mutate the network weights by adding Gaussian noise.

        Args:
            mutation_rate: Standard deviation of the Gaussian noise
        """
        self.w1 += np.random.randn(*self.w1.shape) * mutation_rate
        self.b1 += np.random.randn(*self.b1.shape) * mutation_rate

        self.w2 += np.random.randn(*self.w2.shape) * mutation_rate
        self.b2 += np.random.randn(*self.b2.shape) * mutation_rate

        self.w3 += np.random.randn(*self.w3.shape) * mutation_rate
        self.b3 += np.random.randn(*self.b3.shape) * mutation_rate

    def copy(self):
        """Create a deep copy of this brain."""
        new_brain = Brain(self.input_size, self.hidden_size, self.output_size)
        new_brain.w1 = self.w1.copy()
        new_brain.b1 = self.b1.copy()
        new_brain.w2 = self.w2.copy()
        new_brain.b2 = self.b2.copy()
        new_brain.w3 = self.w3.copy()
        new_brain.b3 = self.b3.copy()
        return new_brain

    def get_weights(self):
        """Get all weights as a flat array."""
        return np.concatenate([
            self.w1.flatten(), self.b1.flatten(),
            self.w2.flatten(), self.b2.flatten(),
            self.w3.flatten(), self.b3.flatten()
        ])

    def set_weights(self, weights):
        """Set all weights from a flat array."""
        idx = 0

        # w1
        size = self.w1.size
        self.w1 = weights[idx:idx+size].reshape(self.w1.shape)
        idx += size

        # b1
        size = self.b1.size
        self.b1 = weights[idx:idx+size]
        idx += size

        # w2
        size = self.w2.size
        self.w2 = weights[idx:idx+size].reshape(self.w2.shape)
        idx += size

        # b2
        size = self.b2.size
        self.b2 = weights[idx:idx+size]
        idx += size

        # w3
        size = self.w3.size
        self.w3 = weights[idx:idx+size].reshape(self.w3.shape)
        idx += size

        # b3
        size = self.b3.size
        self.b3 = weights[idx:idx+size]
