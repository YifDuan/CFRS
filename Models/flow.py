import torch
import torch.nn as nn

class Flip(nn.Module):
  def forward(self, x,):
    x = torch.flip(x, [1])
    # print('flip')
    return x

class WN(torch.nn.Module):
  def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
    super(WN, self).__init__()
    assert(kernel_size % 2 == 1)
    self.hidden_channels =hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels
    self.p_dropout = p_dropout

    self.in_layers = torch.nn.ModuleList()
    self.res_skip_layers = torch.nn.ModuleList()
    self.drop = nn.Dropout(p_dropout)

    if gin_channels != 0:
      cond_layer = torch.nn.Conv1d(gin_channels, 2*hidden_channels*n_layers, 1)
      self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')

    for i in range(n_layers):
      dilation = dilation_rate ** i
      padding = int((kernel_size * dilation - dilation) / 2)
      in_layer = torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size,
                                 dilation=dilation, padding=padding)
      in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
      self.in_layers.append(in_layer)

      if i < n_layers - 1:
        res_skip_channels = 2 * hidden_channels
      else:
        res_skip_channels = hidden_channels

      res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
      res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
      self.res_skip_layers.append(res_skip_layer)


  def forward(self, x):
    output = torch.zeros_like(x)

    for i in range(self.n_layers):
      x_in = self.in_layers[i](x)
      acts = self.drop(x_in)
      
      res_skip_acts = self.res_skip_layers[i](acts)
    #
      if i < self.n_layers - 1:
        res_acts = res_skip_acts[:,:self.hidden_channels,:]
        x = (x + res_acts)
        output = output + res_skip_acts[:,self.hidden_channels:,:]
      else:
        output = output + res_skip_acts
    return output

class ResidualCouplingLayer(nn.Module):
  def __init__(
          self,
          channels,
          hidden_channels,
          kernel_size,
          dilation_rate,
          n_layers,
          p_dropout=0,
          gin_channels=0,
          mean_only=False
  ):
    assert channels % 2 == 0, "channels should be divisible by 2"
    super().__init__()

    self.channels = channels
    self.hidden_channels = hidden_channels
    self.half_channels = channels // 2
    self.mean_only = mean_only

    self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
    self.enc = WN(
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      p_dropout=p_dropout,
      gin_channels=gin_channels
    )

    # 
    self.post = nn.Conv1d(hidden_channels, self.half_channels * 2, 1)

    #  
    self.post.weight.data.zero_()
    self.post.bias.data.zero_()

  def forward(self, x):
    x0, x1 = torch.split(x, [self.half_channels] * 2, dim=1)   # x0 [32,25,46]  x1 [32,25,46]

    h = self.pre(x0)     # Conv1d [32,50,46]
    h = self.enc(h)      # WaveNet [32,50,46]
    stats = self.post(h)  # [32,50,46]

    m, logs = torch.chunk(stats, 2, dim=1)    # m-->[32,25,46]  logs-->[32,25,46]
    logs = torch.zeros_like(logs)
    x1 = m + x1 * torch.exp(logs)
    x = torch.cat([x0, x1], dim=1)            # [32,50,46]

    return x

class ResidualCouplingBlock(nn.Module):
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      n_flows=2,
      gin_channels=0):
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.flows = nn.ModuleList()
    for i in range(n_flows):
      self.flows.append(ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=False))
      self.flows.append(Flip())

  def forward(self, x):

    for flow in self.flows:
        x = flow(x)
    return x

class CrossModalFlowAlign(nn.Module):


    def __init__(self, dim, n_flow_layers=4):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Dimension for Flow must be even, but got {dim}.")

        self.flow = ResidualCouplingBlock(
            channels=dim,
            hidden_channels=dim,
            kernel_size=5,
            dilation_rate=1,
            n_layers=2,
            n_flows=n_flow_layers,
        )

    def forward(self, x):
        # [B, T, D] -> [B, D, T] -> [B, T, D]
        return self.flow(x.transpose(1, 2)).transpose(1, 2)
