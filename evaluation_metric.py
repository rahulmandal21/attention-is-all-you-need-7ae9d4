import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import sacrebleu

class BLEUEvaluator:
    """
    A class to evaluate the BLEU score of machine translation model outputs.
    """

    def __init__(self):
        """
        Initializes the BLEUEvaluator class.
        """
        pass

    def compute_bleu(self, predictions: list, references: list) -> float:
        """
        Computes the BLEU score of the given predictions and references.

        Args:
        predictions (list): A list of predicted translations.
        references (list): A list of reference translations.

        Returns:
        float: The BLEU score of the predictions.
        """
        bleu = sacrebleu.corpus_bleu(predictions, [references])
        return bleu.score

    def evaluate(self, predictions: list, references: list) -> float:
        """
        Evaluates the BLEU score of the given predictions and references.

        Args:
        predictions (list): A list of predicted translations.
        references (list): A list of reference translations.

        Returns:
        float: The BLEU score of the predictions.
        """
        return self.compute_bleu(predictions, references)

class TokenizedTextDataset(Dataset):
    """
    A PyTorch Dataset class for tokenized text sequences.
    """

    def __init__(self, sequences: list):
        """
        Initializes the TokenizedTextDataset class.

        Args:
        sequences (list): A list of tokenized text sequences.
        """
        self.sequences = [torch.tensor(seq) for seq in sequences]

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
        int: The length of the dataset.
        """
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns the item at the given index.

        Args:
        idx (int): The index of the item.

        Returns:
        torch.Tensor: The item at the given index.
        """
        return self.sequences[idx]

def collate_fn(batch: list) -> torch.Tensor:
    """
    A collate function for padding variable-length batches.

    Args:
    batch (list): A list of tensors.

    Returns:
    torch.Tensor: The padded batch.
    """
    return pad_sequence(batch, batch_first=True, padding_value=0)

if __name__ == "__main__":
    evaluator = BLEUEvaluator()
    predictions = ["This is a test", "This is another test"]
    references = ["This is a test", "This is another test"]
    print(evaluator.evaluate(predictions, references))

    dataset = TokenizedTextDataset([[1, 2, 3], [4, 5]])
    data_loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    for batch in data_loader:
        print(batch)