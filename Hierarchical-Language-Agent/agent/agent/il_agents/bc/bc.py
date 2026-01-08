import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from copy import deepcopy
import wandb

wandb.login()

from agent.gameenv_single import GameEnv_Single, get_env
from agent.config import OvercookedExp1
from agent.read_demonstrations import read_multiple_files
from agent.mind.prompt_local import ALL_MOVES
from agent.il_agents.agent_base import IL_Agent
from agent.executor.low import EnvState
from agent.executor.high import HighTask
from agent.mind.prompt_local import MOVE_TO_HT, ALL_MOVES
from agent.il_agents.demonstrator_agent import get_priority_str, all_env_seeds


# Read the dataset and preprocess the values
def read_datasets(fname_list):

  # Get features and labels from the dataset
  def concatenate_values(row):
    return np.concatenate((row['f_state'], np.array([row['prev_macro_idx']])),
                          axis=None)

  demos = read_multiple_files(fname_list)
  X_tensor = None
  y_tensor = None
  for fname in demos:
    df = demos[fname]
    df = df.dropna()  # Drop rows with any missing values
    df.loc[:, 'prev_macro_idx'] = df['prev_macro_idx'].astype(int)

    df = df.assign(X=df.apply(concatenate_values, axis=1))
    X = df['X'].values
    X = [torch.tensor(arr) for arr in X]
    if X_tensor is not None:
      X_tensor = torch.cat((X_tensor, torch.stack(X).float()), dim=0)
    else:
      X_tensor = torch.stack(X).float()

    y = torch.tensor(df['macro_idx'].values)
    if y_tensor is not None:
      y_tensor = torch.cat(
          (y_tensor, F.one_hot(y, num_classes=len(ALL_MOVES)).float()), dim=0)
    else:
      y_tensor = F.one_hot(y, num_classes=len(ALL_MOVES)).float()

  X_train, X_test, y_train, y_test = train_test_split(X_tensor,
                                                      y_tensor,
                                                      test_size=0.2,
                                                      random_state=0)
  return X_train, X_test, y_train, y_test


class CustomDataset(Dataset):

  def __init__(self, features, labels):
    self.features = features
    self.labels = labels

  def __len__(self):
    return len(self.features)

  def __getitem__(self, idx):
    return self.features[idx], self.labels[idx]


# Create a Pytorch dataloader
def create_loaders(X_train, X_test, y_train, y_test, batch_size):
  trainset = CustomDataset(X_train, y_train)
  train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
  valset = CustomDataset(X_test, y_test)
  val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True)
  return train_loader, val_loader


# Define the neural network
class BC_Model(nn.Module):

  def __init__(self, input_size, output_size, layers=[128, 128]):
    super(BC_Model, self).__init__()
    self.layer1 = nn.Linear(input_size, layers[0])
    self.layer2 = nn.Linear(layers[0], layers[1])
    self.output = nn.Linear(layers[1], output_size)

  def forward(self, x):
    x = torch.relu(self.layer1(x))
    x = torch.relu(self.layer2(x))
    x = self.output(x)
    return x


def train_model(num_epochs,
                lr,
                input_size,
                output_size,
                layers,
                train_loader,
                val_loader,
                save_path,
                project_name,
                run_name,
                loss=nn.CrossEntropyLoss(),
                save=True):
  model = BC_Model(input_size, output_size, layers)

  # Define the loss function and optimizer
  criterion = loss
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  # Train the neural network
  best_accuracy = 0

  wandb.init(project=project_name, name=run_name, entity="...", reinit=True)
  for epoch in range(num_epochs):
    logs = {}
    n_correct = 0
    cumulative_loss = 0
    n_samples = 0

    model.train()
    for (batch_id, (xb, yb)) in enumerate(train_loader):
      optimizer.zero_grad()

      # Compute predictions and loss
      pred = model(xb)
      loss = criterion(pred, yb)
      cumulative_loss += loss.item()

      # Count how many correct in batch
      pred_ = pred.softmax(dim=1)
      _, pred_labels = torch.max(pred_, 1)
      n_correct += (pred_labels == torch.argmax(yb, dim=1)).sum().item()
      n_samples += xb.size(0)

      # Compute gradient
      loss.backward()
      optimizer.step()

      # Keep track of loss and accuracy
      n_batches = 1 + batch_id
      logs['loss'] = cumulative_loss / n_batches
      logs['accuracy'] = n_correct / n_samples

    n_correct = 0
    cumulative_loss = 0
    n_samples = 0

    model.eval()
    with torch.no_grad():
      for (batch_id, (xb, yb)) in enumerate(val_loader):
        # Compute predictions and loss
        pred = model(xb)
        loss = criterion(pred, yb)
        cumulative_loss += loss.item()

        # Count how many correct in batch
        pred_ = pred.softmax(dim=1)
        _, pred_labels = torch.max(pred_, 1)
        n_correct += (pred_labels == torch.argmax(yb, dim=1)).sum().item()
        n_samples += xb.size(0)

        # Keep track of loss and accuracy
        n_batches = 1 + batch_id
        logs['val_loss'] = cumulative_loss / n_batches
        logs['val_accuracy'] = n_correct / n_samples

    # Save the parameters for the best accuracy on the validation set so far.
    if logs['val_accuracy'] > best_accuracy:
      best_accuracy = logs['val_accuracy']
      if save:
        torch.save(model.state_dict(), save_path)

    wandb.log(logs)

    if (epoch + 1) % 10 == 0:
      print(f'Epoch [{epoch+1}/{num_epochs}]')
      print(f'Training loss: {logs["loss"]:.4f}')
      print(f'Training acc: {logs["accuracy"]:.4f}')
      print(f'Val loss: {logs["val_loss"]:.4f}')
      print(f'Val acc: {logs["val_accuracy"]:.4f}')

  wandb.finish()


class BC_Agent(IL_Agent):

  def __init__(self) -> None:
    super().__init__()
    self.prev_intent_idx = 0

  def load_model(self, model_path, input_size, output_size, layers):
    self.model = BC_Model(input_size, output_size, layers=layers)
    self.model.load_state_dict(torch.load(model_path))
    self.model.eval()

  def step(self, env_state: EnvState, env_tensor):
    while True:
      if self._task is None:
        pred = self.model(
            torch.cat((env_tensor, torch.tensor([self.prev_intent_idx]))))
        pred_ = pred.softmax(dim=-1)
        pred_label = torch.multinomial(pred_, num_samples=1).item()
        self.prev_intent_idx = pred_label
        self.cur_intent = ALL_MOVES[pred_label]
        print('Move: ', self.cur_intent)
        self._task = deepcopy(MOVE_TO_HT[self.cur_intent])

      state, move, msg = self._task(env_state)
      if state == HighTask.Working:  # working
        return move, None
      elif state == HighTask.Failed:  # reassign task
        print(f"Move Failed: {move}")
        self._task = None
        return (0, 0), None
      else:
        self._task = None


def main():
  num_demos_list = [1, 3, 5, 10]
  for num_demos in num_demos_list:
    print(f'Training when there are {num_demos} demos...')
    epochs = 1000 if num_demos < 5 else 700
    lr = 1.e-4
    batch_size = 32
    layers = [128, 128]

    priority = [['David Soup'], ['Alice Soup', 'Cathy Soup']]
    p = get_priority_str(priority)
    directory = f'demonstrations/{p}'
    train_env_seeds = all_env_seeds[:num_demos]
    file_names = [f'{p}_demo_env{s}_agent0.txt' for s in train_env_seeds]
    file_names = [os.path.join(directory, f) for f in file_names]
    print('Files: ', file_names)

    X_train, X_test, y_train, y_test = read_datasets(fname_list=file_names)
    train_loader, val_loader = create_loaders(X_train,
                                              X_test,
                                              y_train,
                                              y_test,
                                              batch_size=batch_size)

    run_name = f"{p}_{num_demos}demos_batch{batch_size}_nn{layers[0]}_lr{lr}"
    train_model(
        epochs,
        lr,
        X_train.shape[1],
        y_train.shape[1],
        layers,
        train_loader,
        val_loader,
        f'il_agents/bc/{p}/{str(num_demos)}demos/best_bc_{str(num_demos)}demos.pth',
        "overcooked-bc",
        run_name,
        save=True)

    eval_env_seeds = all_env_seeds[-5:]
    suffixes = ['fast0', 'fast1']
    for env_seed in eval_env_seeds:
      for suffix in suffixes:
        overcooked_env = get_env(OvercookedExp1,
                                 priority=priority,
                                 seed=env_seed)
        bc_agent = BC_Agent()
        bc_agent.load_model(
            f'il_agents/bc/{p}/{str(num_demos)}demos/best_bc_{str(num_demos)}demos.pth',
            X_train.shape[1], y_train.shape[1], layers)
        game = GameEnv_Single(env=overcooked_env,
                              max_timesteps=1000,
                              agent_type='bc',
                              agent_model=bc_agent,
                              play=False)
        game.execute_agent(
            fps=3,
            sleep_time=0.0001,
            fname=
            f'il_agents/bc/{p}/{str(num_demos)}demos/test_env{str(env_seed)}_{suffix}',
            write=True)


if __name__ == "__main__":
  main()
