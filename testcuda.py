import torch
import gmmvaeBeta00 as Mb
from torchvision.models import resnet18

if __name__ == "__main__":
    print(
            "cuda available? ",
            torch.cuda.is_available(),
            )
    model = resnet18()
    model.cuda()
    model.cpu()
    model = Mb.VAE_Dirichlet_GMM_TypeB1602xz()
    model.cuda()
    model.cpu()
    print("done")

#torch.cuda.is_available(),
#model = resnet18()
#model.cuda()
#model.cpu()
#model = Mb.VAE_Dirichlet_GMM_TypeB1602xz()
#model.cuda()
#model.cpu()
