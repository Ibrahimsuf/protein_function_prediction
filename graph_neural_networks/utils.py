from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
import torch_geometric 
from typing import Tuple, Optional
import torch
def split_dataset(dataset: Dataset, train_size: Optional[float] = 0.8, val_size: Optional[float] = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
  """ 
  Split a dataset into train, validation, and test sets. 

  Args:
    dataset (torch_geometric.data.Dataset): The dataset to split.
    train_size (float): The fraction of the dataset to use for training.
    val_size (float): The fraction of the dataset to use for validation.

  Returns:
    train_dataset (torch_geometric.data.Dataset): The training set.
    val_dataset (torch_geometric.data.Dataset): The validation set.
    test_dataset (torch_geometric.data.Dataset): The test set.
  """
  if train_size + val_size > 1:
    raise ValueError('train_size + val_size cannot be greater than 1')

  num_train = int(len(dataset) * train_size)
  num_val = int(len(dataset) * val_size)

  dataset = dataset.shuffle()
  train_dataset = dataset[:num_train]
  val_dataset = dataset[num_train:num_train + num_val]
  test_dataset = dataset[num_train + num_val:]


  return DataLoader(train_dataset), DataLoader(val_dataset), DataLoader(test_dataset)

def train(model: torch.nn.Module, train_loader: DataLoader, epochs: Optional[int] = 200, verbose: Optional[bool] = True) -> None:
  """
  Train a model using the given data loader.

  Args:
    model (torch.nn.Module): The model to train.
    train_loader (torch_geometric.data.DataLoader): The data loader to use.
    epochs (int): The number of epochs to train for.
  """
  optimizer = torch.optim.Adam(model.parameters(), lr=4e-3)
  model.train()
  loss_fn = torch.nn.BCEWithLogitsLoss()
  for epoch in range(epochs):
    total_loss = 0
    for data in train_loader:
      optimizer.zero_grad()
      out = model(data.x, data.edge_index, data.batch)
      loss = loss_fn(out, data.y.view(-1, 1).type_as(out))
      loss.backward()
      optimizer.step()
      total_loss += loss.item() / len(train_loader)
    if verbose:
      print(f'Epoch: {epoch + 1}, Loss: {total_loss / len(train_loader.dataset)}')

def test(model: torch.nn.Module, loader: DataLoader) -> float:
  """
  Test a model using the given data loader.

  Args:
    model (torch.nn.Module): The model to test.
    loader (torch_geometric.data.DataLoader): The data loader to use.

  Returns:
    float: The accuracy of the model on the given data.
  """
  model.eval()
  correct = 0
  for data in loader:
    out = model(data.x, data.edge_index, data.batch)
    pred = out > 0
    correct += (pred.view(-1) == data.y).sum().item()
  return correct / len(loader.dataset)
