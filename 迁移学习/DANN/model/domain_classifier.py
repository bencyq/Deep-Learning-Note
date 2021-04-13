import torch.nn as nn



class DomainDiscriminator(nn.Module):

    def __init__(self, in_feature, hidden_size, batch_norm=True):
        super(DomainDiscriminator, self).__init__()
        if batch_norm:
            self.domain_disciminator = nn.Sequential(
                nn.Linear(in_feature, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )
        else:
            self.domain_disciminator = nn.Sequential(
                nn.Linear(in_feature, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        return self.domain_disciminator(x)

    def get_parameters(self):
        return [{'params': self.domain_disciminator.parameters(), 'lr':1.}]