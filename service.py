import numpy as np
import bentoml
import numpy as np
from PIL.Image import Image as PILImage
from PIL import Image as ImageP
import cv2
from bentoml.io import Image
from bentoml.io import NumpyNdarray,File
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T

from numpy.typing import NDArray

deeplpf_runner = bentoml.pytorch.get("deeplpf_model:latest").to_runner()

svc = bentoml.Service("net", runners=[deeplpf_runner])

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

@svc.api(input=Image(), output=Image())
def predict(input_img: PILImage):
    transform = T.ToTensor()
    img = transform(input_img)
    
    img = img.unsqueeze(0)
    img = torch.clamp(img, 0, 1)
    img = img.cuda()

    #output
    outimg = deeplpf_runner.run(img)

    outimg = outimg.squeeze(0).permute(1,2,0)
    outimg = outimg.mul(255).byte().cpu().detach().numpy()
    
    #PIL为RGB格式，转为BGR格式显示
    # outimg = np.array(outimg)[:, :, ::-1]   
    image = ImageP.fromarray(np.uint8(outimg))
    return image