import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class DataPreprocessing(nn.Module):
    """
    This class is responsible for converting input and output tokens to vectors of dimension d_model,
    using learned embeddings and positional encoding.
    """

    def __init__(self, vocab_size: int, d_model: int, max_len: int):
        """
        Initializes the DataPreprocessing class.

        Args:
            vocab_size (int): The size of the vocabulary.
            d_model (int): The dimension of the model.
            max_len (int): The maximum length of the input sequence.
        """
        super(DataPreprocessing, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self.generate_positional_encoding(max_len, d_model)

    def generate_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """
        Generates the positional encoding.

        Args:
            max_len (int): The maximum length of the input sequence.
            d_model (int): The dimension of the model.

        Returns:
            torch.Tensor: The positional encoding tensor.
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Converts the input tokens to vectors of dimension d_model.

        Args:
            input_ids (torch.Tensor): The input token ids.

        Returns:
            torch.Tensor: The embedded input tokens with positional encoding.
        """
        embedded_input = self.embedding(input_ids)
        positional_encoding = self.positional_encoding[:input_ids.size(0), :]
        return embedded_input + positional_encoding

def main():
    # Create a dummy dataset
    vocab_size = 1000
    d_model = 512
    max_len = 100
    input_ids = torch.randint(0, vocab_size, (10, max_len))

    # Create a DataPreprocessing instance
    data_preprocessing = DataPreprocessing(vocab_size, d_model, max_len)

    # Use the DataPreprocessing instance
    output = data_preprocessing(input_ids)
    print(output.shape)

if __name__ == "__main__":
    main()