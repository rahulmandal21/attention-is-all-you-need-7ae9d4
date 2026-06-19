import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLossFunction(nn.Module):
    """
    A PyTorch module implementing the cross-entropy loss function.
    """

    def __init__(self, num_classes: int, smoothing: float = 0.1):
        """
        Initializes the CrossEntropyLossFunction module.

        Args:
        - num_classes (int): The number of classes in the classification problem.
        - smoothing (float, optional): The label smoothing factor. Defaults to 0.1.
        """
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
        self.num_classes = num_classes

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the cross-entropy loss between the predictions and targets.

        Args:
        - predictions (torch.Tensor): The predicted logits.
        - targets (torch.Tensor): The true labels.

        Returns:
        - torch.Tensor: The cross-entropy loss.
        """
        return self.criterion(predictions, targets)

    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the cross-entropy loss using the nn.functional.nll_loss API.

        Args:
        - predictions (torch.Tensor): The predicted logits.
        - targets (torch.Tensor): The true labels.

        Returns:
        - torch.Tensor: The cross-entropy loss.
        """
        softmax_output = F.softmax(predictions, dim=1)
        log_softmax_output = F.log_softmax(predictions, dim=1)
        return F.nll_loss(log_softmax_output, targets)

if __name__ == "__main__":
    # Create a dummy dataset
    predictions = torch.randn(10, 5)
    targets = torch.randint(0, 5, (10,))

    # Initialize the CrossEntropyLossFunction module
    loss_function = CrossEntropyLossFunction(num_classes=5)

    # Compute the cross-entropy loss
    loss = loss_function(predictions, targets)
    print("Cross-Entropy Loss:", loss)

    # Compute the cross-entropy loss using the nn.functional.nll_loss API
    loss = loss_function.compute_loss(predictions, targets)
    print("Cross-Entropy Loss (nll_loss):", loss)