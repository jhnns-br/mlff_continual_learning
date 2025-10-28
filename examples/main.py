import torch
from mlff import models 


if __name__ == "__main__":
    n_c = 2
    res50 = models.MLFFResNet50(num_classes=n_c, token=None)
    dino = models.MLFFDINOv2RegBase(num_classes=n_c, token="your_huggingface_token")
    mobile = models.MLFFMobileNetV2S(num_classes=n_c, token="your_huggingface_token")
    swin = models.MLFFSwinV2B(num_classes=n_c, token=None)

    x = torch.randn(64, 3, 128, 128).cuda()

    for model in [res50, dino, mobile, swin]:
        model.cuda()
        model.train()

        params = model.get_trainable_params()
        all_params = sum(p.numel() for p in model.parameters())
        with_grad = sum(p.numel() for p in params if p.requires_grad)

        y = model(x)
        print(f"{y.shape} | {y.device} | {all_params:,} | {with_grad:,}")




