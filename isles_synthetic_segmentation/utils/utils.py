import numpy as np
from losses.bce import BCE
from losses.fbeta_loss import FBetaLoss
from losses.fbeta_ce_loss import FBetaCELoss
from losses.dice_loss import DiceLoss
from monai.losses.dice import DiceFocalLoss, DiceCELoss
from skimage import io


def load_image(img_name):
    if img_name.endswith("png"):
        image = io.imread(img_name)
    elif img_name.endswith("npz"): 
        image = np.load(img_name)['arr_0']
    else:
        raise ValueError("Not yet implemented")
    return image


def get_loss_function(loss_name):
    if loss_name == 'f0.5':
        loss = FBetaLoss(beta=0.5)
    elif loss_name == 'f1':
        loss = FBetaLoss(beta=1.0)
    elif loss_name == 'f2':
        loss = FBetaLoss(beta=2.0)
    elif loss_name == 'dice':
        loss = DiceLoss()
    elif loss_name == 'DiceFocalLoss':
        loss = DiceFocalLoss(include_background=False, to_onehot_y=False, sigmoid=True)
    elif loss_name == 'DiceCELoss':
        loss = DiceCELoss(include_background=False, to_onehot_y=False, sigmoid=True)
    elif loss_name == "BCE" or loss_name == "bce":
        loss = BCE()
    elif loss_name == "fce_0.5":
        loss = FBetaCELoss(beta=0.5)
    elif loss_name == "fce_1":
        loss = FBetaCELoss(beta=1)
    elif loss_name == "fce_2":
        loss = FBetaCELoss(beta=2)
    else:
        raise ValueError("Unsupported  loss")

    return loss