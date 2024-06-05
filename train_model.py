import torch
import torch.optim as optim
import numpy as np
from utils import rollout
from pdb import set_trace as debug
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loss_fn = torch.nn.MSELoss()

def train_single(num_epochs, num_batches, batch_size, model, optimizer, replay_buffer):
    for epoch in range(num_epochs):

        for i in range(num_batches):
            optimizer.zero_grad()
            t1_observations, t1_actions, _, t1_next_observations, _ = replay_buffer.sample(batch_size)
            oa_in = torch.cat([t1_observations, t1_actions], dim=-1)

            next_o_pred = model(oa_in)
            loss = loss_fn(next_o_pred, t1_next_observations)

            loss.backward()
            optimizer.step()


def train_model(model, replay_buffer, optimizer, num_epochs=500, batch_size=32):
    """
    Train a single model with supervised learning
    """
    idxs = np.array(range(len(replay_buffer)))
    num_batches = len(idxs) // batch_size
    if not isinstance(model, list):
        train_single(num_epochs, num_batches, batch_size, model, optimizer, replay_buffer)
    # TODO START-Ensemble models
    # Hint1: try different batch size for each model
    # hint2: check out how we define optimizer and model for ensemble models. During training, each model should have their individual optimizer and batch size to increase diversity.
    else:
        exp = int(np.log2(batch_size))
        if len(model) % 2 == 0:
            n = len(model) // 2
            exps = np.linspace(-(n+1), n, len(model)) + exp
        else:
            n = (len(model) - 1) // 2
            exps = np.linspace(-n, n, len(model)) + exp
        batch_sizes = [int(2 ** exp) for exp in exps]
        for id,model_ in enumerate(model):
            batch_size = batch_sizes[id]
            num_batches = int(len(idxs) // batch_size)
            train_single(num_epochs, num_batches, batch_size, model[id], optimizer[id], replay_buffer)
    # TODO END
