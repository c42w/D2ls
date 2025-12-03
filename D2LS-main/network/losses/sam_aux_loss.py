# network/losses/sam_aux_loss.py
import torch
import torch.nn.functional as F
from torch import Tensor

class ObjectConsistencyLoss:
    def __init__(self, lambda_obj: float = 0.5):
        self.lambda_obj = lambda_obj

    def __call__(self, pred_logits: Tensor, sgo_masks: Tensor) -> Tensor:
        if sgo_masks is None:
            return 0.0
        pred = F.softmax(pred_logits, dim=1)
        loss = 0.0
        B = pred.shape[0]
        for b in range(B):
            obj_ids = torch.unique(sgo_masks[b])
            obj_ids = obj_ids[obj_ids != 0]
            if len(obj_ids) == 0: continue
            for obj_id in obj_ids:
                mask = (sgo_masks[b] == obj_id).float()
                if mask.sum() < 10: continue
                pred_obj = pred[b:b+1, :, mask > 0]
                mean_pred = pred_obj.mean(dim=2, keepdim=True)
                loss += F.mse_loss(pred_obj, mean_pred.expand_as(pred_obj))
        return self.lambda_obj * loss / B if B > 0 else 0.0

class SAMBoundaryLoss:
    def __init__(self, lambda_bound: float = 0.5):
        self.lambda_bound = lambda_bound

    def __call__(self, boundary_pred: Tensor, sgb: Tensor) -> Tensor:
        if boundary_pred is None or sgb is None:
            return 0.0
        weight = 1.0 + 5.0 * sgb  # 可调权重
        loss = F.binary_cross_entropy_with_logits(
            boundary_pred.squeeze(1),
            (sgb > 0.1).float(),
            weight=weight
        )
        return self.lambda_bound * loss