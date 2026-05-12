# tests/training/test_distill.py
import torch
from training.scripts.distill import distillation_loss


def test_distillation_loss_shape():
    student_cat_logits = torch.randn(4, 8)
    student_sev_logits = torch.randn(4, 5)
    soft_labels = torch.softmax(torch.randn(4, 8), dim=-1)
    hard_cat_labels = torch.randint(0, 8, (4,))
    hard_sev_labels = torch.randint(0, 5, (4,))

    loss = distillation_loss(
        student_cat_logits=student_cat_logits,
        student_sev_logits=student_sev_logits,
        soft_labels=soft_labels,
        hard_cat_labels=hard_cat_labels,
        hard_sev_labels=hard_sev_labels,
        temperature=4.0,
        alpha=0.7,
    )
    assert loss.shape == torch.Size([])
    assert loss.item() > 0


def test_distillation_loss_alpha_boundary():
    student_logits = torch.randn(2, 8)
    soft = torch.softmax(torch.randn(2, 8), dim=-1)
    hard = torch.zeros(2, dtype=torch.long)
    sev = torch.zeros(2, dtype=torch.long)
    sev_logits = torch.randn(2, 5)

    loss_full = distillation_loss(student_logits, sev_logits, soft, hard, sev, 4.0, alpha=1.0)
    loss_none = distillation_loss(student_logits, sev_logits, soft, hard, sev, 4.0, alpha=0.0)
    assert loss_full.item() != loss_none.item()
