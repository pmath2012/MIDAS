import torch
from losses.fbeta_loss import FBetaLoss
from losses.bce import BCE

class FBetaCELoss(torch.nn.Module):
    """
    Combined F-beta (F1) and Binary Cross Entropy (BCE) loss.
    This loss computes the sum of FBetaLoss (with beta=1.0) and BCE for binary segmentation tasks.
    """
    def __init__(self, beta=1.0):
        super(FBetaCELoss, self).__init__()
        self.fbeta = FBetaLoss(beta=beta)
        self.bce = BCE()

    def forward(self, logits, targets):
        return self.fbeta(logits, targets) + self.bce(logits, targets) 