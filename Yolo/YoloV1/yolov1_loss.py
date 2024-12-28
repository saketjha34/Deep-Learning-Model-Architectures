"""
YOLOv1 Loss Module

This module implements the loss function for the YOLOv1 model, which combines multiple 
loss components to optimize bounding box predictions, object confidence, and class probabilities.

Parameters:
    S (int): The grid size of the input image (e.g., 7 for a 7x7 grid).
    B (int): Number of bounding boxes per grid cell.
    C (int): Number of classes for object classification.
    lambda_coord (float): Weight for the bounding box coordinate loss.
    lambda_noobj (float): Weight for the no-object confidence loss.

Attributes:
    mse (nn.MSELoss): Mean Squared Error loss for computing individual loss components.
    S (int): Grid size.
    B (int): Number of bounding boxes per grid cell.
    C (int): Number of classes.
    lambda_coord (float): Weight for the bounding box coordinate loss.
    lambda_noobj (float): Weight for the no-object confidence loss.

Methods:
    forward(predictions, target):
        Computes the YOLOv1 loss given the predictions and ground truth targets.

        Parameters:
            predictions (torch.Tensor): Tensor of shape (BATCH_SIZE, S*S*(C+B*5)) containing predictions.
            target (torch.Tensor): Tensor of the same shape as predictions containing ground truth labels.

        Returns:
            torch.Tensor: The computed YOLOv1 loss.
"""

import torch
import torch.nn as nn

def intersection_over_union(boxes_preds: torch.Tensor, boxes_labels: torch.Tensor, box_format: str="midpoint")->torch.Tensor:
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


class YoloV1Loss(nn.Module):
    """
    Calculate the loss for yolo (v1) model
    """

    def __init__(self, S=7, B=2, C=20, lambda_coord = 5, lambda_noobj=0.5):
        super(YoloV1Loss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        """
        S is split size of image (in paper 7),
        B is number of boxes (in paper 2),
        C is number of classes (in paper and VOC dataset is 20),
        """
        self.S = S
        self.B = B
        self.C = C

        # These are from Yolo paper, signifying how much we should
        # pay loss for no object (noobj) and the box coordinates (coord)
        self.lambda_noobj = lambda_noobj
        self.lambda_coord = lambda_coord

    def forward(self, predictions, target):
        # predictions are shaped (BATCH_SIZE, S*S(C+B*5) when inputted
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # Calculate IoU for the two predicted bounding boxes with target bbox
        iou_b1 = intersection_over_union(predictions[..., C:C+4], target[..., C:C+4])
        iou_b2 = intersection_over_union(predictions[..., C+5:C+9], target[..., C:C+4])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        # Adjusted bestbox indices
        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., C-1].unsqueeze(3)  # C-1 corresponds to the objectness score

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        # Set boxes with no object in them to 0. We only take out one of the two 
        # predictions, which is the one with highest Iou calculated previously.
        box_predictions = exists_box * (
            (
                bestbox * predictions[..., 26:30]
                + (1 - bestbox) * predictions[..., 21:25]
            )
        )

        box_targets = exists_box * target[..., 21:25]

        # Take sqrt of width, height of boxes to ensure that
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # pred_box is the confidence score for the bbox with highest IoU
        pred_box = (
            bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21]),
        )

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        #max_no_obj = torch.max(predictions[..., 20:21], predictions[..., 25:26])
        #no_object_loss = self.mse(
        #    torch.flatten((1 - exists_box) * max_no_obj, start_dim=1),
        #    torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        #)

        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2,),
            torch.flatten(exists_box * target[..., :20], end_dim=-2,),
        )

        loss = (
            self.lambda_coord * box_loss  # first two rows in paper
            + object_loss  # third row in paper
            + self.lambda_noobj * no_object_loss  # forth row
            + class_loss  # fifth row
        )

        return loss
    
if __name__ == "__main__":
    # Constants
    N = 8  # Batch size
    S = 7  # Grid size (SxS grid)
    B = 2  # Number of bounding boxes per grid cell
    C = 2  # Number of classes

    loss_fn = YoloV1Loss(S, B, C)
    # Create dummy tensors for predictions and targets
    predictions = torch.rand(N, S, S, B * 5 + C)  # Random predictions
    targets = predictions.clone()  # Start with identical targets

    # Add slight perturbation to ensure intersection
    targets[..., :B * 5] += torch.rand(N, S, S, B * 5) * 0.1  # Add small noise to bounding box predictions

    # Verify shapes
    print("Predictions shape:", predictions.shape)
    print("Targets shape:", targets.shape)
    print(loss_fn(predictions, targets))