import torch
from torch import nn

class SimplePerceptron(nn.Module):
    def __init__(self, input_dim, num_classes) -> None:
        super().__init__()

        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.linear1(x.to(torch.float))
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x

class Perceptron(nn.Module):
    def __init__(self, input_dim, num_classes) -> None:
        super().__init__()
    
        self.linear1 = nn.Linear(input_dim, 50)
        self.linear2 = nn.Linear(768, 768)
        self.linear3 = nn.Linear(768, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, *args):
        x = self.linear1(x)
        # x = self.relu(x)
        # x = self.linear2(x)
        # x = self.dropout(x)
        # x = self.relu(x)
        # x = self.linear3(x)
        return x
    
class Perceptron1Class(nn.Module):
    def __init__(self, input_dim, num_classes) -> None:
        super().__init__()
    
        self.linear1 = nn.Linear(input_dim, 768)
        self.linear2 = nn.Linear(768, 768)
        self.linear3 = nn.Linear(768, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, *args):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x
    
class BERTCls(nn.Module):
    def __init__(self, bert) -> None:
        super().__init__()
        self.backbone = bert
        self.relu = nn.ReLU()
        self.linear_without_tag = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, 768),
            self.relu,
            nn.Linear(768, 256),
            self.relu,
            nn.Linear(256, 50),
        )
        self.linear = nn.Linear(self.backbone.config.hidden_size, 50)
        self.linear_layer_with_tag = nn.Linear(self.backbone.config.hidden_size+8, 50)
        
        self.linear_with_tag = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size+8, self.backbone.config.hidden_size+9),
            nn.ReLU(),
            nn.Linear(self.backbone.config.hidden_size+8, 256),
            nn.ReLU(),
            nn.Linear(256, 50),
        )
    
    def forward(self, x, x_assist=None):
        for key in x:
            x[key] = x[key].squeeze()
        
            
        out = self.backbone(**x).pooler_output
        #out = self.linear(out)
        # assert out[0][0] != out[0][1]
        if x_assist != None:
            out = torch.cat([out, x_assist], dim=-1)
            # x = self.dropout(x)
            # x = self.gelu(x)
            # out = self.linear_with_tag(out)
            out = self.linear_layer_with_tag(out)

        else:
            # x = self.dropout(x)
            out = self.linear_without_tag(out)
            
            # x = self.relu(x)
            # x = self.linear3(x)
        return out
    
class TripletLossModel(nn.Module):
    def __init__(self, bert) -> None:
        super().__init__()
        self.backbone = bert
        self.relu = nn.ReLU()
        self.linear_without_tag = nn.Sequential(
            nn.Linear(768, 768),
            self.relu,
            nn.Linear(768, 256),
            self.relu,
            nn.Linear(256, 50),
        )
        self.linear = nn.Linear(768, 50)
        self.linear_layer_with_tag = nn.Linear(768+8, 50)
        
        self.linear_with_tag = nn.Sequential(
            nn.Linear(768+9, 768+9),
            nn.ReLU(),
            nn.Linear(768+9, 256),
            nn.ReLU(),
            nn.Linear(256, 50),
        )
    
    def forward(self, x, x_assist=None):
        for key in x:
            x[key] = x[key].squeeze()
        
            
        out = self.backbone(**x).pooler_output
        #out = self.linear(out)
        # assert out[0][0] != out[0][1]
        if x_assist != None:
            out = torch.cat([out, x_assist], dim=-1)
            # x = self.dropout(x)
            # x = self.gelu(x)
            # out = self.linear_with_tag(out)
            out = self.linear_layer_with_tag(out)

        else:
            # x = self.dropout(x)
            out = self.linear_without_tag(out)
            
            # x = self.relu(x)
            # x = self.linear3(x)
        return out
    
    