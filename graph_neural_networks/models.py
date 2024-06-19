from torch_geometric.nn import GCNConv, GraphConv, global_mean_pool
import torch.nn.functional as F
from torch import tensor
import torch
from torch.nn import Linear
from torch import Tensor
from typing import Optional
class GCN(torch.nn.Module):
  """
  Graph Convolutional Network with 2 GCNConv layers followed by global mean pooling and then linear for classifications.
  
  Methods:
    forward(x, edge_index, batch): Forward pass of the model
  """

  def __init__(self, in_channels : int, hidden_channels : int, out_channels : int) -> None:
    """
    Args:
      in_channels (int): Number of input features
      hidden_channels (int): Number of hidden features
      out_channels (int): Number of output features
    """
    super().__init__()
    self.conv1 = GCNConv(in_channels, hidden_channels)
    self.conv2  = GCNConv(hidden_channels, hidden_channels)
    self.linear = Linear(hidden_channels, out_channels)

  def forward(self, x: Tensor, edge_index: Tensor, batch: Optional[Tensor] = None) -> Tensor:
    """
    Args:
      x (Tensor): Input features
      edge_index (Tensor): Edge indices
      batch (Optional[Tensor], optional): Batch vector. Defaults assumes all nodes belong to the same graph.

    Returns:
      Tensor: Unnormalized logits for each class
    """
    if batch is None:
      batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

    x = self.conv1(x, edge_index).relu()
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.conv2(x, edge_index)
    x = global_mean_pool(x, batch=batch)
    out = self.linear(x)
    return out


class GraphConvNet(torch.nn.Module):
  """
  Graph Convolutional Network with 2 GraphConv layers followed by global mean pooling and then linear for classifications.
  
  
  Methods:
    forward(x, edge_index, batch): Forward pass of the model
  """

  def __init__(self, in_channels : int, hidden_channels : int, out_channels : int) -> None:
    """
    Args:
      in_channels (int): Number of input features
      hidden_channels (int): Number of hidden features
      out_channels (int): Number of output features
    """
    super().__init__()
    self.conv1 = GraphConv(in_channels, hidden_channels)
    self.conv2  = GraphConv(hidden_channels, hidden_channels)
    self.linear = Linear(hidden_channels, out_channels)

  def forward(self, x: Tensor, edge_index: Tensor, batch: Optional[Tensor] = None) -> Tensor:
    """
    Args:
      x (Tensor): Input features
      edge_index (Tensor): Edge indices
      batch (Optional[Tensor], optional): Batch vector. Defaults assumes all nodes belong to the same graph.

    Returns:
      Tensor: Unnormalized logits for each class
    """
    if batch is None:
      batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

    x = self.conv1(x, edge_index).relu()
    x = self.conv2(x, edge_index)
    x = global_mean_pool(x, batch=batch)
    out = self.linear(x)
    return out
  