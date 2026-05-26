"""
Training script for Relation Network baseline.

Reference:
    Sung et al., "Learning to Compare: Relation Network for Few-Shot Learning", CVPR 2018.
"""

import os
import time
import numpy as np
import torch
import learn2learn as l2l

from models import EncoderNet, RelationNet
from datasets import MetaLearningDataset, CWRUConfig
from utils import get_device, accuracy, weights_init, seed_everything


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


def fast_adapt(encoder, relation, batch, loss_fn, shots, ways):
    """
    Relation Network episode evaluation.

    Args:
        encoder: Feature encoder network.
        relation: Relation scoring network.
        batch: Task data (data, labels).
        loss_fn: Loss function.
        shots: Support samples per class.
        ways: Number of classes.

    Returns:
        Tuple of (loss, accuracy).
    """
    data, labels = batch
    data, labels = data.to(device), labels.to(device)
    query_num = shots

    # Sort by labels
    sort_indices = torch.sort(labels).indices
    data = data[sort_indices]
    labels = labels[sort_indices]

    # Split support/query
    support_indices = np.zeros(data.size(0), dtype=bool)
    selection = np.arange(ways) * (shots + query_num)
    for offset in range(shots):
        support_indices[selection + offset] = True

    query_indices = torch.from_numpy(~support_indices)
    support_indices = torch.from_numpy(support_indices)

    # Encode
    embeddings = encoder(data)
    support = embeddings[support_indices]  # [n_support, C, L]
    query = embeddings[query_indices]      # [n_query, C, L]
    query_labels = labels[query_indices].long()

    # Compute prototypes and relation pairs
    support = support.reshape(ways, shots, *support.shape[-2:]).mean(dim=1)  # [ways, C, L]
    support = support.unsqueeze(0).repeat(query.shape[0], 1, 1, 1)  # [n_q, ways, C, L]
    query_expanded = query.unsqueeze(1).repeat(1, ways, 1, 1)       # [n_q, ways, C, L]

    # Concatenate and compute relation scores
    pairs = torch.cat((support, query_expanded), dim=2)  # [n_q, ways, 2C, L]
    pairs = pairs.reshape(-1, pairs.shape[2], pairs.shape[3])  # [n_q*ways, 2C, L]
    scores = relation(pairs).reshape(-1, ways)  # [n_q, ways]

    loss = loss_fn(scores, query_labels)
    acc = accuracy(scores, query_labels)
    return loss, acc


def train(config, save_dir, ways=5, shots=5, epochs=100, episodes=40,
          lr=0.001, seed=2023):
    """
    Train Relation Network.

    Args:
        config: CWRUConfig instance.
        save_dir: Directory to save checkpoints.
        ways: Number of classes per task.
        shots: Support samples per class.
        epochs: Training epochs.
        episodes: Episodes per epoch.
        lr: Learning rate.
        seed: Random seed.
    """
    seed_everything(seed)
    os.makedirs(save_dir, exist_ok=True)

    encoder = EncoderNet(in_channels=1, hidden_channels=64, num_blocks=4).to(device)
    # Compute embed_size: 1024 / 2^4 / 2^2 * 64
    sample_len = 1024
    embed_size = (sample_len // (2**4) // (2**2)) * 64
    relation = RelationNet(hidden_channels=64, embed_size=embed_size, hidden_size=256).to(device)

    encoder.apply(weights_init)
    relation.apply(weights_init)

    optimizer_e = torch.optim.Adam(encoder.parameters(), lr=lr * 0.1, weight_decay=2e-5)
    optimizer_r = torch.optim.Adam(relation.parameters(), lr=lr, weight_decay=2e-5)
    scheduler_r = torch.optim.lr_scheduler.ExponentialLR(optimizer_r, gamma=0.99)
    loss_fn = torch.nn.CrossEntropyLoss()

    print(f"Training RelationNet: {ways}-way {shots}-shot")
    train_tasks = build_tasks(config, 'train', ways, shots, num_tasks=1000)
    valid_tasks = build_tasks(config, 'validation', ways, shots, num_tasks=50)

    best_acc = 0.0
    for epoch in range(epochs):
        t0 = time.time()

        # Training
        encoder.train()
        relation.train()
        train_loss, train_acc = 0.0, 0.0
        for _ in range(episodes):
            batch = train_tasks.sample()
            loss, acc = fast_adapt(encoder, relation, batch, loss_fn, shots, ways)
            train_loss += loss.item()
            train_acc += acc.item()

            optimizer_e.zero_grad()
            optimizer_r.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(relation.parameters(), 0.5)
            optimizer_e.step()
            optimizer_r.step()

        scheduler_r.step()

        # Validation
        encoder.eval()
        relation.eval()
        val_loss, val_acc = 0.0, 0.0
        for batch in valid_tasks:
            with torch.no_grad():
                loss, acc = fast_adapt(encoder, relation, batch, loss_fn, shots, ways)
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

        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(save_dir, "relation_best.pt")
            torch.save({'encoder': encoder.state_dict(), 'relation': relation.state_dict()}, save_path)
            print(f"  -> Best model saved (val_acc={best_acc:.4f})")

    save_path = os.path.join(save_dir, f"relation_ep{epochs}.pt")
    torch.save({'encoder': encoder.state_dict(), 'relation': relation.state_dict()}, save_path)
    print(f"Final model saved: {save_path}")


def test(config, load_path, ways=5, shots=5, num_tasks=200):
    """
    Test Relation Network.

    Args:
        config: CWRUConfig instance.
        load_path: Path to saved model.
        ways: Number of classes per task.
        shots: Support samples per class.
        num_tasks: Number of test tasks.
    """
    encoder = EncoderNet(in_channels=1, hidden_channels=64, num_blocks=4).to(device)
    sample_len = 1024
    embed_size = (sample_len // (2**4) // (2**2)) * 64
    relation = RelationNet(hidden_channels=64, embed_size=embed_size, hidden_size=256).to(device)

    state = torch.load(load_path, map_location=device)
    encoder.load_state_dict(state['encoder'])
    relation.load_state_dict(state['relation'])
    encoder.eval()
    relation.eval()
    print(f"Loaded model from: {load_path}")
    print(f"Testing RelationNet: {ways}-way {shots}-shot, {num_tasks} tasks")

    test_tasks = build_tasks(config, 'test', ways, shots, num_tasks=num_tasks)
    loss_fn = torch.nn.CrossEntropyLoss()

    test_loss, test_acc = 0.0, 0.0
    t0 = time.time()
    for batch in test_tasks:
        with torch.no_grad():
            loss, acc = fast_adapt(encoder, relation, batch, loss_fn, shots, ways)
        test_loss += loss.item()
        test_acc += acc.item()

    elapsed = time.time() - t0
    print(f"Test ({elapsed:.2f}s): Loss={test_loss/num_tasks:.4f}, Acc={test_acc/num_tasks:.4f}")
