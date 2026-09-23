"""Fixed training-only backbone adaptation, without validation or selection."""
import math


def adapt(model, loader, epochs, lr, emit):
    import torch
    if type(epochs) is not int or epochs<1 or not math.isfinite(lr) or lr<=0:
        raise ValueError('positive epochs and learning rate required')
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    device=next(model.parameters()).device
    planned=epochs*len(loader); applied=0
    emit(dict(event='adaptation_started',epochs=epochs,lr=lr,planned_updates=planned))
    for epoch in range(epochs):
        model.train(); total_loss=0.; count=0
        for inputs,training_labels in loader:
            inputs,training_labels=inputs.to(device),training_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(model(inputs),training_labels)
            if not torch.isfinite(loss): raise RuntimeError('nonfinite adaptation loss')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):
                raise RuntimeError('invalid adaptation gradient')
            optimizer.step(); applied+=1
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise RuntimeError('nonfinite adaptation parameter')
            total_loss+=float(loss.detach().item())*len(inputs);count+=len(inputs)
        if count==0:raise ValueError('empty training loader')
        emit(dict(event='adaptation_epoch',epoch=epoch+1,training_ce=total_loss/count,
                  examples=count,applied_updates=applied))
    if applied!=planned:raise RuntimeError('adaptation dose mismatch')
    model.eval()
    emit(dict(event='adaptation_completed',planned_updates=planned,applied_updates=applied,skipped_updates=0))


class TrainingImages:
    """Filters rows before any labels/images are exposed to the training loader."""
    def __init__(self, root, rows, transform):
        from pathlib import Path
        self.root=Path(root)
        self.rows=[r for r in rows if r['split']=='train']
        self.transform=transform
        if not self.rows:raise ValueError('empty training split')

    def __len__(self):return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image
        from .knee_experiment import digest
        row=self.rows[index];p=self.root/row['path']
        if digest(p)!=row['sha256']:raise ValueError('training image changed')
        with Image.open(p) as image:value=self.transform(image.convert('RGB'))
        return value,row['label']
