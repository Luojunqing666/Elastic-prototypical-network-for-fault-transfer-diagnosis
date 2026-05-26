"""
Training script for MAML (Model-Agnostic Meta-Learning) baseline.

Reference:
    Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks", ICML 2017.
"""

import os
import time
import numpy as np
import torch
import learn2learn as l2l

from models import MAMLNet
from datasets import MetaLearningDataset, CWRUConfig
from utils import get_device, accuracy, seed_everything


device = get_device()


def build_tasks(config, mode='train', ways=5, shots=5, num_tasks=100):
    """Build meta-learning task dataset."""
    dataset = l2l.data.MetaDataset(MetaLearningDataset(config, mode=mode))
    assert shots * 2 * ways <= len(dataset), "Reduce shots or ways!"
    tasks = l2l.data.TaskDataset(dataset, task_transforms=[
        l2l.data.transforms.FusedNWaysKShots(dataset, ways, 2 * shots),
        l2l.data.transforms.LoadData(dataset),
        l2l.data.transforms.RemapLabels(dataset, shuffle=True),
        l2l.data.transforms.ConsecutiveLabels(dataset),
    ], num_tasks=num_tasks)
    return tasks


def fast_adapt(batch, learner, loss_fn, adaptation_steps, shots, ways):
    """
    MAML inner-loop adaptation and evaluation.

    Args:
        batch: Task data (data, labels).
        learner: Cloned MAML learner.
        loss_fn: Loss function.
        adaptation_steps: Number of inner-loop gradient steps.
        shots: Support samples per class.
        ways: Number of classes.

    Returns:
        Tuple of (evaluation_loss, evaluation_accuracy).
    """
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Split into support (adaptation) and query (evaluation)
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots * ways) * 2] = True

    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)

    adaptation_data = data[adaptation_indices]
    adaptation_labels = labels[adaptation_indices]
    evaluation_data = data[evaluation_indices]
    evaluation_labels = labels[evaluation_indices]

    # Inner-loop adaptation
    for _ in range(adaptation_steps):
        train_error = loss_fn(learner(adaptation_data), adaptation_labels)
        learner.adapt(train_error)

    # Evaluate adapted model
    predictions = learner(evaluation_data)
    eval_error = loss_fn(predictions, evaluation_labels)
    eval_acc = accuracy(predictions, evaluation_labels)
    return eval_error, eval_acc


def train(config, save_dir, ways=5, shots=5, epochs=100, lr=0.005, seed=2023):
    """
    Train MAML model.

    Args:
        config: CWRUConfig instance.
        save_dir: Directory to save checkpoints.
        ways: Number of classes per task.
        shots: Support samples per class.
        epochs: Number of meta-training epochs.
        lr: Meta learning rate.
        seed: Random seed.
    """
    seed_everything(seed)
    os.makedirs(save_dir, exist_ok=True)

    model = MAMLNet(num_classes=ways, in_channels=1, hidden_channels=64, num_layers=4).to(device)
    fast_lr = 0.05
    maml = l2l.algorithms.MAML(model, lr=fast_lr)
    optimizer = torch.optim.Adam(maml.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    print(f"Training MAML: {ways}-way {shots}-shot")
    train_tasks = build_tasks(config, 'train', ways, shots, num_tasks=1000)
    valid_tasks = build_tasks(config, 'validation', ways, shots, num_tasks=1000)

    meta_batch_size = 16
    adaptation_steps = 1 if shots >= 5 else 3
    best_acc = 0.0

    for epoch in range(epochs):
        t0 = time.time()
        train_loss, train_acc = 0.0, 0.0
        val_loss, val_acc = 0.0, 0.0

        optimizer.zero_grad()
        for _ in range(meta_batch_size):
            # Meta-train
            learner = maml.clone()
            task = train_tasks.sample()
            loss, acc = fast_adapt(task, learner, loss_fn, adaptation_steps, shots, ways)
            loss.backward()
            train_loss += loss.item()
            train_acc += acc.item()

            # Meta-validate
            learner = maml.clone()
            task = valid_tasks.sample()
            loss, acc = fast_adapt(task, learner, loss_fn, adaptation_steps, shots, ways)
            val_loss += loss.item()
            val_acc += acc.item()

        # Meta-update
        for p in maml.parameters():
            if p.grad is not None:
                p.grad.data.mul_(1.0 / meta_batch_size)
        optimizer.step()

        train_loss /= meta_batch_size
        train_acc /= meta_batch_size
        val_loss /= meta_batch_size
        val_acc /= meta_batch_size

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{epochs} ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(save_dir, "maml_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  -> Best model saved (val_acc={best_acc:.4f})")

    save_path = os.path.join(save_dir, f"maml_ep{epochs}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Final model saved: {save_path}")


def test(config, load_path, ways=5, shots=5, num_tasks=200, inner_steps=10):
    """
    Test MAML model.

    Args:
        config: CWRUConfig instance.
        load_path: Path to saved model.
        ways: Number of classes per task.
        shots: Support samples per class.
        num_tasks: Number of test tasks.
        inner_steps: Number of adaptation steps at test time.
    """
    model = MAMLNet(num_classes=ways, in_channels=1, hidden_channels=64, num_layers=4).to(device)
    model.load_state_dict(torch.load(load_path, map_location=device))
    print(f"Loaded model from: {load_path}")
    print(f"Testing MAML: {ways}-way {shots}-shot, {inner_steps} inner steps")

    fast_lr = 0.05
    maml = l2l.algorithms.MAML(model, lr=fast_lr)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    test_tasks = build_tasks(config, 'test', ways, shots, num_tasks=num_tasks)

    test_loss, test_acc = 0.0, 0.0
    t0 = time.time()
    for _ in range(num_tasks):
        learner = maml.clone()
        task = test_tasks.sample()
        loss, acc = fast_adapt(task, learner, loss_fn, inner_steps, shots, ways)
        test_loss += loss.item()
        test_acc += acc.item()

    elapsed = time.time() - t0
    print(f"Test ({elapsed:.2f}s): Loss={test_loss/num_tasks:.4f}, Acc={test_acc/num_tasks:.4f}")
