"""
Training script for Elastic Prototypical Network (EProtoNet).

This is the core method proposed in the paper. It uses a CNN backbone with
SE attention and an elastic (scaled) Euclidean distance metric for
few-shot fault diagnosis under domain shift.
"""

import os
import time
import numpy as np
import torch
import learn2learn as l2l

from models import EProtoNet
from datasets import MetaLearningDataset, CWRUConfig
from utils import get_device, accuracy, weights_init, seed_everything


device = get_device()


def build_tasks(config, mode='train', ways=5, shots=5, num_tasks=100):
    """
    Build meta-learning task dataset using learn2learn.

    Args:
        config: CWRUConfig instance.
        mode: 'train', 'validation', or 'test'.
        ways: Number of classes per task.
        shots: Number of support samples per class.
        num_tasks: Number of tasks to generate.

    Returns:
        learn2learn TaskDataset.
    """
    dataset = l2l.data.MetaDataset(MetaLearningDataset(config, mode=mode))
    assert shots * 2 * ways <= len(dataset), "Reduce shots or ways!"
    tasks = l2l.data.TaskDataset(dataset, task_transforms=[
        l2l.data.transforms.FusedNWaysKShots(dataset, ways, 2 * shots),
        l2l.data.transforms.LoadData(dataset),
        l2l.data.transforms.RemapLabels(dataset, shuffle=True),
        l2l.data.transforms.ConsecutiveLabels(dataset),
    ], num_tasks=num_tasks)
    return tasks


def fast_adapt(model, batch, loss_fn, shots, ways, use_elastic=True):
    """
    Perform one episode of prototypical network evaluation.

    Args:
        model: EProtoNet model.
        batch: Tuple of (data, labels) from task sampling.
        loss_fn: Loss function (CrossEntropyLoss).
        shots: Number of support samples per class.
        ways: Number of classes.
        use_elastic: Whether to use elastic (scaled) distance.

    Returns:
        Tuple of (loss, accuracy).
    """
    data, labels = batch
    data, labels = data.to(device), labels.to(device)
    query_num = shots  # Equal number of query samples

    # Sort by labels for consistent support/query split
    sort_indices = torch.sort(labels).indices
    data = data[sort_indices]
    labels = labels[sort_indices]

    # Split into support and query sets
    support_indices = np.zeros(data.size(0), dtype=bool)
    selection = np.arange(ways) * (shots + query_num)
    for offset in range(shots):
        support_indices[selection + offset] = True

    query_indices = torch.from_numpy(~support_indices)
    support_indices = torch.from_numpy(support_indices)

    # Compute embeddings
    embeddings = model(data)
    support = embeddings[support_indices]
    query = embeddings[query_indices]
    query_labels = labels[query_indices].long()

    # Compute prototypes and logits
    prototypes = model.compute_prototypes(support, ways, shots)
    if use_elastic:
        logits = model.scaled_euclidean_distance(query, prototypes)
    else:
        logits = model.euclidean_distance(query, prototypes)

    loss = loss_fn(logits, query_labels)
    acc = accuracy(logits, query_labels)
    return loss, acc


def train(config, save_dir, ways=5, shots=5, epochs=100, episodes=30,
          lr=0.001, use_elastic=True, seed=2023):
    """
    Train EProtoNet model.

    Args:
        config: CWRUConfig instance.
        save_dir: Directory to save model checkpoints.
        ways: Number of classes per task.
        shots: Number of support samples per class.
        epochs: Number of training epochs.
        episodes: Number of episodes per epoch.
        lr: Learning rate.
        use_elastic: Whether to use elastic distance metric.
        seed: Random seed.
    """
    seed_everything(seed)
    os.makedirs(save_dir, exist_ok=True)

    model = EProtoNet(in_channels=1, hidden_channels=64, num_layers=4, use_se=True).to(device)
    model.apply(weights_init)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    loss_fn = torch.nn.CrossEntropyLoss()

    print(f"Training EProtoNet: {ways}-way {shots}-shot, elastic={use_elastic}")
    train_tasks = build_tasks(config, 'train', ways, shots, num_tasks=1000)
    valid_tasks = build_tasks(config, 'validation', ways, shots, num_tasks=50)

    best_acc = 0.0
    for epoch in range(epochs):
        t0 = time.time()

        # Training
        model.train()
        train_loss, train_acc = 0.0, 0.0
        for _ in range(episodes):
            batch = train_tasks.sample()
            loss, acc = fast_adapt(model, batch, loss_fn, shots, ways, use_elastic)
            train_loss += loss.item()
            train_acc += acc.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        for batch in valid_tasks:
            with torch.no_grad():
                loss, acc = fast_adapt(model, batch, loss_fn, shots, ways, use_elastic)
            val_loss += loss.item()
            val_acc += acc.item()

        train_loss /= episodes
        train_acc /= episodes
        val_loss /= len(valid_tasks)
        val_acc /= len(valid_tasks)

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{epochs} ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(save_dir, "eprotonet_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  -> Best model saved (val_acc={best_acc:.4f})")

    # Save final model
    save_path = os.path.join(save_dir, f"eprotonet_ep{epochs}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Final model saved: {save_path}")
    return model


def test(config, load_path, ways=5, shots=5, num_tasks=200, use_elastic=True):
    """
    Test EProtoNet model.

    Args:
        config: CWRUConfig instance.
        load_path: Path to saved model checkpoint.
        ways: Number of classes per task.
        shots: Number of support samples per class.
        num_tasks: Number of test tasks.
        use_elastic: Whether to use elastic distance metric.
    """
    model = EProtoNet(in_channels=1, hidden_channels=64, num_layers=4, use_se=True).to(device)
    model.load_state_dict(torch.load(load_path, map_location=device))
    model.eval()
    print(f"Loaded model from: {load_path}")
    print(f"Testing: {ways}-way {shots}-shot, {num_tasks} tasks")

    test_tasks = build_tasks(config, 'test', ways, shots, num_tasks=num_tasks)
    loss_fn = torch.nn.CrossEntropyLoss()

    test_loss, test_acc = 0.0, 0.0
    t0 = time.time()
    for batch in test_tasks:
        with torch.no_grad():
            loss, acc = fast_adapt(model, batch, loss_fn, shots, ways, use_elastic)
        test_loss += loss.item()
        test_acc += acc.item()

    elapsed = time.time() - t0
    print(f"Test ({elapsed:.2f}s): Loss={test_loss/num_tasks:.4f}, Acc={test_acc/num_tasks:.4f}")
