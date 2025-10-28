import torch


class MLFFBlock(torch.nn.Module):
    """
    basic building block for MLFF stage of a model
    """
    def __init__(self, embedding_dims: list, fc_dim: int, num_classes: int):
        """
        Args:
            embedding_dims (list): list of integers defining size of extracted embedding vectors
            fc_dim (int): dimension of the fully connected layer 
            num_classes (int): number of classes to ouptut
        """
        super().__init__()
        assert fc_dim % len(embedding_dims) == 0

        lat_dim = int(fc_dim/len(embedding_dims))
        self.mlff_layers = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Linear(in_features=ed, out_features=lat_dim),
                    torch.nn.BatchNorm1d(num_features=lat_dim),
                    torch.nn.ReLU()
                ) for ed in embedding_dims])

        self.mlff_fc = torch.nn.Sequential(
                torch.nn.Linear(in_features=fc_dim, out_features=fc_dim),
                torch.nn.BatchNorm1d(num_features=fc_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=fc_dim, out_features=num_classes)
            )

    def forward(self, embds: list):
        return self.cat_forward(embds=embds)

    def cat_forward(self, embds: list):
        """
        forward function using fusion by concatenation

        Args:
            embds (list): list of torch.tensors with extracted embeddings

        Returns:
            tuple(torch.tensor, torch.tensor): returns the output vector y and the fused embeddings 
        """
        embds = [self.mlff_layers[i](embds[i]) for i in range(len(embds))]
        cat_embds = torch.cat(embds, dim=-1)
        return self.mlff_fc(cat_embds), cat_embds


class MLFFBaseModel(torch.nn.Module):
    """
    Base model class, that specific models should reference
    """
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module() 
        self.mlff_block = torch.nn.Module()

    def forward(self, x):
        raise NotImplementedError("forward of MLFFBaseModel not overwritten")
    
    def embedding_forward(self, x):
        raise NotImplementedError("embedding_forward of MLFFBaseModel not overwritten")
    
    def train(self, mode=True):
        if mode:
            self.model.eval()
            self.mlff_block.train()
        else:
            self.eval()
    
    def eval(self):
        self.model.eval()
        self.mlff_block.eval()

    def get_trainable_params(self):
        return self.mlff_block.parameters()
