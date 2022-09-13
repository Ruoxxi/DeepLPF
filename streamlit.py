import model
import streamlit as st
import torch
import numpy as np
import cv2
import io
from PIL import Image
import torchvision.transforms as T
import seaborn as sns
from pyheatmap.heatmap import HeatMap
import matplotlib.pyplot as plt 

#net
net = model.DeepLPFNet()
checkpoint_filepath = './pretrained_models/adobe_dpe/deeplpf_validpsnr_23.378_validloss_0.033_testpsnr_23.904_testloss_0.031_epoch_424_model.pt'
net.load_state_dict(torch.load(checkpoint_filepath))
net.eval()
net.cuda()

uploaded_file = st.file_uploader("", type=["jpg", "png", "bmp", "jpeg"])
col1, col2 = st.columns(2)
col3,col4,col5 = st.columns(3)
if uploaded_file is not None:
    img_PIL = Image.open(uploaded_file).convert('RGB')
    with col1:
        st.header("Input")
        st.image(img_PIL)

    transform = T.ToTensor()
    img = transform(img_PIL)
    img = img.unsqueeze(0)
    img = img.cuda()

    # backbone net
    x = net.backbonenet(img)
    feat = x[:, 3:64, :, :]
    img = x[:, 0:3, :, :]

    torch.cuda.empty_cache()
    shape = x.shape
    # get masks
    img_cubic,cubic_mask = net.deeplpfnet.cubic_filter.get_cubic_mask(feat, img)
    mask_scale_graduated = net.deeplpfnet.graduated_filter.get_graduated_mask(feat, img_cubic)
    mask_scale_elliptical = net.deeplpfnet.elliptical_filter.get_elliptical_mask(feat, img_cubic)
    
    mask_scale_fuse = torch.clamp(mask_scale_graduated+mask_scale_elliptical, 0, 2)
    img_fuse = torch.clamp(img_cubic*mask_scale_fuse, 0, 1)
    img = torch.clamp(img_fuse+img, 0, 1)

    mask_scale_elliptical = torch.clamp(mask_scale_elliptical,0,1)
    mask_scale_graduated = torch.clamp(mask_scale_graduated,0,1)
    cubic_mask = torch.clamp(cubic_mask,0,1)

    #result
    outimg = img.squeeze(0).permute(1,2,0)
    outimg = outimg.mul(255).byte().cpu().detach().numpy()
    with col2:
        st.header("result")
        st.image(outimg)

    #elliptical
    elliptical = mask_scale_elliptical.squeeze(0).permute(1,2,0)
    elliptical = elliptical.mul(255)
    print(elliptical)
    elliptical = elliptical.byte().cpu().detach().numpy()
    sns.heatmap(data=elliptical[:,:,1])
    plt.savefig('sns1.jpg', dpi = 600)
    image = Image.open('sns1.jpg')
    with col3:
        st.header("elliptical")
        st.markdown("green channel")
        st.image(image)
    plt.close('all')

    #graduated
    graduated = mask_scale_graduated.squeeze(0).permute(1,2,0)
    graduated = graduated.mul(255).byte().cpu().detach().numpy()
    print(graduated.shape)
    sns.heatmap(data=graduated[:,:,2])
    plt.savefig('sns2.jpg', dpi = 600)
    image = Image.open('sns2.jpg')
    with col4:
        st.header("graduated")
        st.markdown("blue channel")
        st.image(image)
    plt.close('all')

    #cubic
    cubic_mask = cubic_mask.squeeze(0).permute(1,2,0)
    cubic_mask = cubic_mask.mul(255).byte().cpu().detach().numpy()

    sns.heatmap(data=cubic_mask[:,:,0])
    plt.savefig('sns3.jpg', dpi = 600)
    image = Image.open('sns3.jpg')
    with col5:
        st.header("cubic")
        st.markdown("red channel")
        st.image(image)


    


