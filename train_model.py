import torch
import torch.optim as optim
import numpy as np
from utils import rollout
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loss_fn = torch.nn.MSELoss()

def train_single(num_epochs, num_batches,batch_size, model, optimizer, replay_buffer):
    for epoch in range(num_epochs):

        for i in range(num_batches):
            optimizer.zero_grad()
            # TODO START: training single model
            # Hint: Sample from the replay buffer and supervise it using MSE loss between true next observation and predicted next observation

            t1_observations, t1_actions, _, t1_next_observations, _ = replay_buffer.sample(batch_size)
            oa_in = torch.cat([t1_observations, t1_actions], dim=-1)

            next_o_pred = model(oa_in)
            loss = loss_fn(next_o_pred, t1_next_observations)
            # TODO END

            loss.backward()
            optimizer.step()

def train_model(model, replay_buffer, optimizer, num_epochs=500, batch_size=32):
    """
    Train a single model with supervised learning
    """
    idxs = np.array(range(len(replay_buffer)))
    num_batches = len(idxs) // batch_size
    if isinstance(model, list):
        # TODO START-Ensemble models
        # Hint1: try different batch size for each model
        # hint2: check out how we define optimizer and model for ensemble models. During training, each model should have their individual optimizer and batch size to increase diversity.
        for model_id, curr_model in enumerate(model):
            batch_size_curr = batch_size + (model_id+1)*32 # use different batch size for each model
            train_single(num_epochs, num_batches, batch_size_curr, curr_model, optimizer[model_id], replay_buffer)
        # TODO END
    else:
        train_single(num_epochs, num_batches,batch_size, model, optimizer, replay_buffer)

