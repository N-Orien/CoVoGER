import torch.nn as nn

class ProjectorConv1d(nn.Module):
    def __init__(self,
                 ds_rate=5,
                 encoder_dim=1024,
                 hidden_dim=2048,
                 llm_dim=4096):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=encoder_dim,
                                out_channels=encoder_dim,
                                kernel_size=ds_rate,
                                stride=ds_rate,
                                padding=0)
        self.linear1 = nn.Linear(encoder_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, llm_dim)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = x.transpose(1, 2)
        x = self.relu1(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        return x
