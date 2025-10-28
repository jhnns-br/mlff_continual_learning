import torch
import torchvision
from transformers import Dinov2WithRegistersModel, MobileNetV2Model

from mlff.base import MLFFBaseModel, MLFFBlock


class MLFFResNet50(MLFFBaseModel):
    def __init__(self, num_classes:int, token:str=""):
        """
        ResNet-50 MLFF version. Backbone initialized with ImageNet1k weights provided by torchvision.

        Args:
            num_classes (int): number of classes 
            token (str): access token str. Not needed here, but args is added for consistency. 
        """
        super().__init__()
        self.model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        del self.model.fc
        for _, param in self.model.named_parameters():
            param.requires_grad = False
        self.mlff_block = MLFFBlock(embedding_dims=[256, 512, 1024, 2048],
                                  fc_dim=256*4, num_classes=num_classes)

    def forward(self, x):
        out, _ = self.embedding_forward(x)
        return out

    def embedding_forward(self, x):
        """
        Args:
            x (torch.tensor): input tensor

        Returns:
            tuple(torch.tensor, torch.tensor): returns the output vector y and the fused embeddings 
        """
        embds = []
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        embds.append(self.model.layer1(x))
        embds.append(self.model.layer2(embds[-1]))
        embds.append(self.model.layer3(embds[-1]))
        embds.append(self.model.layer4(embds[-1]))

        embds = [
            torch.nn.functional.adaptive_max_pool2d(embds[0], (1, 1)),
            torch.nn.functional.adaptive_max_pool2d(embds[1], (1, 1)),
            torch.nn.functional.adaptive_avg_pool2d(embds[2], (1, 1)),
            torch.nn.functional.adaptive_avg_pool2d(embds[3], (1, 1))
        ]
        embds = [torch.squeeze(e, dim=(2, 3)) for e in embds]

        out, _ = self.mlff_block(embds)
        return  out, embds
    
   
class MLFFDINOv2RegBase(MLFFBaseModel):
    def __init__(self, num_classes:int, token:str):
        """
        DINOv2 with registers MLFF version. Backbone initialized with weights provided by huggingface.

        Args:
            num_classes (int): number of classes 
            token (str): access token for huggingface 
        """
        super().__init__()
        self.model = Dinov2WithRegistersModel.from_pretrained(
            "facebook/dinov2-with-registers-base",
            token=token
        )
        for _, param in self.model.named_parameters():
            param.requires_grad = False

        self.mlff_block = MLFFBlock(embedding_dims=[1536, 1536, 1536, 1536],
                                    fc_dim=1536, num_classes=num_classes)

    def forward(self, x):
        out, _ = self.embedding_forward(x)
        return out
    
    def embedding_forward(self, x):
        """
        Args:
            x (torch.tensor): input tensor

        Returns:
            tuple(torch.tensor, torch.tensor): returns the output vector y and the fused embeddings 
        """
        outputs = self.model(pixel_values=x, output_hidden_states=True)
        sequence_output = outputs.last_hidden_state
        hidden_states = outputs.hidden_states

        embds = [hidden_states[0], hidden_states[4], hidden_states[8], sequence_output]
        embds = [torch.cat([e[:, 0], torch.mean(e[:, 1:], dim=1)], dim=1) for e in embds]

        out, _ = self.mlff_block(embds)
        return  out, embds


class MLFFMobileNetV2S(MLFFBaseModel):
    def __init__(self, num_classes:int, token:str):
        """
        MobileNetV2S MLFF version. Backbone initialized with ImageNet1k weights provided by huggingface.

        Args:
            num_classes (int): number of classes 
            token (str): access token for huggingface 
        """
        super().__init__()
        self.model = MobileNetV2Model.from_pretrained(  # ImageNet1k pretrained
            "Matthijs/mobilenet_v2_1.0_224",
            token=token
        ) 
        for _, param in self.model.named_parameters():
            param.requires_grad = False

        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.flatten = torch.nn.Flatten()

        self.mlff_block = MLFFBlock(embedding_dims=[24, 32, 96, 1280],
                                   fc_dim=1280, num_classes=num_classes)

    def forward(self, x):
        out, _ = self.embedding_forward(x)
        return out
    
    def embedding_forward(self, x):
        """
        Args:
            x (torch.tensor): input tensor

        Returns:
            tuple(torch.tensor, torch.tensor): returns the output vector y and the fused embeddings 
        """
        outputs = self.model(pixel_values=x, output_hidden_states=True)
        sequence_output = outputs.last_hidden_state
        hidden_states = outputs.hidden_states

        embds = [hidden_states[1], hidden_states[4], hidden_states[11], sequence_output]
        embds = [self.avgpool(e) for e in embds]
        embds = [self.flatten(e) for e in embds]

        out, _ = self.mlff_block(embds)
        return  out, embds


class MLFFSwinV2B(MLFFBaseModel):
    def __init__(self, num_classes:int, token:str):
        """
        SwinV2B MLFF version. Backbone initialized with ImageNet1k weights provided by torchvision.

        Args:
            num_classes (int): number of classes 
            token (str): access token str. Not needed here, but args is added for consistency. 
        """
        super().__init__()
        self.model = torchvision.models.swin_v2_b(weights=torchvision.models.Swin_V2_B_Weights.IMAGENET1K_V1)
        del self.model.head
        for _, param in self.model.named_parameters():
            param.requires_grad = False

        self.mlff_block = MLFFBlock(embedding_dims=[128, 256, 512, 1024], 
                                    fc_dim=1024, 
                                    num_classes=num_classes) 

    def forward(self, x):
        out, _ = self.embedding_forward(x)
        return out

    def embedding_forward(self, x):
        """
        Args:
            x (torch.tensor): input tensor

        Returns:
            tuple(torch.tensor, torch.tensor): returns the output vector y and the fused embeddings 
        """
        embds = []
        for fb in self.model.features:
            x = fb(x)
            embds.append(x)

        embds = [embds[1], embds[3], embds[5], self.model.norm(embds[-1])]
        embds = [self.model.permute(e) for e in embds]
        embds = [self.model.avgpool(e) for e in embds]
        embds = [self.model.flatten(e) for e in embds]
        
        out, _ = self.mlff_block(embds)
        return out, embds
