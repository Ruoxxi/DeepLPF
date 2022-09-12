import model
import bentoml
import torch

net = model.DeepLPFNet()
checkpoint_filepath = './pretrained_models/adobe_dpe/deeplpf_validpsnr_23.378_validloss_0.033_testpsnr_23.904_testloss_0.031_epoch_424_model.pt'
net.load_state_dict(torch.load(checkpoint_filepath))
net.eval()
net.cuda()

bentoml.pytorch.save_model(  
    "deeplpf_model",
    net,
    signatures={"predict": {"batchable": True, "batchdim": 0}}
)