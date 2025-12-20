import torch
import torch.nn as nn


class TransductivePortfolioLoss(nn.Module):
    def __init__(self, k_target, k_feat, sector_matrix, lambda_target=1.0, lambda_sector=1.0,
                 use_bce=False, average_sectors=True):
        super().__init__()
        self.k_target = k_target
        self.k_feat = k_feat
        self.lambda_target = lambda_target
        self.lambda_sector = lambda_sector
        self.use_bce = use_bce
        self.average_sectors = average_sectors
        self.register_buffer('sector_matrix', sector_matrix)
        self.eps = 1e-6

        if use_bce:
            self.bce = nn.BCELoss()

    def forward(self, y_hat, y_true=None):
        E_target = torch.relu(y_hat.sum() - self.k_target)
        L_target = E_target / (E_target + self.k_target + self.eps)

        sector_sums = torch.matmul(self.sector_matrix, y_hat)
        E_sector = torch.relu(sector_sums - self.k_feat)

        if self.average_sectors:
            L_sector = (E_sector / (E_sector + self.k_feat + self.eps)).mean()
        else:
            L_sector = (E_sector / (E_sector + self.k_feat + self.eps)).sum()

        L_constraint = self.lambda_target * L_target + self.lambda_sector * L_sector

        if self.use_bce and y_true is not None:
            L_pred = self.bce(y_hat, y_true)
            L_total = L_pred + L_constraint
            return L_total

        return L_constraint
