import torch
import torch.nn as nn
from torch.nn.functional import mse_loss


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                     tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or(self.real_label_var.numel() != input.numel()))
            # pdb.set_trace()
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                # self.real_label_var = Variable(real_tensor, requires_grad=False)
                # self.real_label_var = torch.Tensor(real_tensor)
                self.real_label_var = real_tensor
            target_tensor = self.real_label_var
        else:
            # pdb.set_trace()
            create_label = ((self.fake_label_var is None) or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                # self.fake_label_var = Variable(fake_tensor, requires_grad=False)
                # self.fake_label_var = torch.Tensor(fake_tensor)
                self.fake_label_var = fake_tensor
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        # pdb.set_trace()
        return self.loss(input, target_tensor)

import torch
import torch.nn.functional as F

def edge_loss(pred, target):
    def sobel(x):
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3)

        grad_x = F.conv2d(x, sobel_x, padding=1, groups=1)
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=1)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

    # Apply to grayscale (convert to 1 channel first)
    pred_gray = 0.2989 * pred[:, 0:1] + 0.5870 * pred[:, 1:2] + 0.1140 * pred[:, 2:3]
    target_gray = 0.2989 * target[:, 0:1] + 0.5870 * target[:, 1:2] + 0.1140 * target[:, 2:3]

    edge_pred = sobel(pred_gray)
    edge_target = sobel(target_gray)

    return F.l1_loss(edge_pred, edge_target)
