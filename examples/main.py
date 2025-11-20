import torch
import torchvision
from mlff import models 
from PIL import Image

TOKEN = "your_huggingface_token"

NUM_CLASSES = 2
IMG_PATH = "examples/data/example_image.png"

if __name__ == "__main__":
    # define some models 
    res50 = models.MLFFResNet50(num_classes=NUM_CLASSES, token=None)
    dino = models.MLFFDINOv2RegBase(num_classes=NUM_CLASSES, token=TOKEN)
    mobile = models.MLFFMobileNetV2S(num_classes=NUM_CLASSES, token=TOKEN)
    swin = models.MLFFSwinV2B(num_classes=NUM_CLASSES, token=None)

    # define some inputs 
    x1 = torch.randn(32, 3, 128, 128) # first a tensor with noise from a random normal distribution and a batch size of 32
    
    x2 = Image.open(IMG_PATH) # second an example image 
    x2 = torchvision.transforms.v2.functional.pil_to_tensor(x2)
    if x2.shape[0] > 3:
        x2 = x2[:3, :, :]
    x2 = torchvision.transforms.v2.functional.to_dtype(x2, dtype=torch.float32, scale=True)
    x2 = torchvision.transforms.v2.functional.normalize(x2, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    x2 = torch.unsqueeze(x2, 0) # add a batch dimension in front
    
    print("="*100)
    print("Input size (x1 | x2)")
    print("-"*100)
    print(x1.shape, "|", x2.shape)

    # first print model sizes
    print("="*100)
    print("Model parameters (all | with grad)")
    print("-"*100)
    for model in [res50, dino, mobile, swin]:
        if torch.cuda.is_available():
            model.cuda()
            x1.cuda()
            x2.cuda()
        model.train()

        params = model.get_trainable_params()
        all_params = sum(p.numel() for p in model.parameters())
        with_grad = sum(p.numel() for p in params if p.requires_grad)
        print(f"{all_params:,} | {with_grad:,}")

    # check outputs 
    print("="*100)
    print("Model outputs (shape | type | device)")
    print("-"*100)
    for model in [res50, dino, mobile, swin]:
        model.eval()
        y = model(x1)
        print(f"{y.shape} | {y.dtype} | {y.device}")

    # check embeddings
    print("="*100)
    print("Extracted Embeddings (nr. | shapes)")
    print("-"*100)
    for model in [res50, dino, mobile, swin]:
        y, embds = model.embedding_forward(x2)
        print(f"{len(embds)} | {[e.shape for e in embds]}")
    print("-"*100)
