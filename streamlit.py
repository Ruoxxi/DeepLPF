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

#choose input image
img = st.columns(6)
input_image = None
gt_image = None
input_images = ['1.png','2.png','3.png','4.png','5.png','6.png']
j = 0
for i in input_images:
    with img[j]:
        if st.button(i):
            input_image = Image.open('./images/input/'+i)
            gt_image = Image.open('./images/groundtruth/'+i)
        st.image(Image.open('./images/input/'+i))
        j = j+1

#get threshold
st.header("Choose a threshold")
threshold = st.slider('Threshold:(%)',0,100,50)

if input_image is not None:
    #initiate net
    net = model.DeepLPFNet()
    checkpoint_filepath = './pretrained_models/adobe_upe/deeplpf_validpsnr_32.068971287609415_validloss_0.01073065772652626_testpsnr_25.13364554976333_testloss_0.0252277459949255_epoch_424_model.pt'
    net.load_state_dict(torch.load(checkpoint_filepath))
    net.eval()
    net.cuda()

    # #display label image
    in_image, label_image, out_image, diff_image = st.columns(4)
    gt_image =gt_image.convert('RGB')
    with label_image:
        st.header("Label")
        st.image(gt_image)

    #display input image
    input_image = input_image.convert('RGB')
    with in_image:
        st.header("Input")
        st.image(input_image)

    # backbone net
    transform = T.ToTensor()
    input_image = transform(input_image)
    input_image = input_image.unsqueeze(0)
    input_image = input_image.cuda()

    x = net.backbonenet(input_image)
    x.contiguous()  # remove memory holes
    x.cuda()
    feat = x[:, 3:64, :, :]
    img = x[:, 0:3, :, :]
    torch.cuda.empty_cache()
    shape = x.shape

    # get masks
    img_cubic,cubic_mask = net.deeplpfnet.cubic_filter.get_cubic_mask(feat, img)
    mask_scale_graduated = net.deeplpfnet.graduated_filter.get_graduated_mask(feat, img_cubic)
    mask_scale_elliptical = net.deeplpfnet.elliptical_filter.get_elliptical_mask(feat, img_cubic)
    mask_scale_fuse = torch.clamp(mask_scale_graduated+mask_scale_elliptical, 0, 2)
    img_fuse = torch.clamp(img_cubic*mask_scale_fuse, 0, 1)*threshold/100
    img = torch.clamp(img_fuse+img, 0, 1)

    mask_scale_elliptical = torch.clamp(mask_scale_elliptical,0,1)*threshold/100
    mask_scale_graduated = torch.clamp(mask_scale_graduated,0,1)*threshold/100
    
    cubic_mask = torch.clamp(cubic_mask,0,1)*threshold/100

    #result
    outimg = img.squeeze(0).permute(1,2,0)
    outimg = outimg.mul(255).byte().cpu().detach().numpy()
    with out_image:
        st.header("Result")
        st.image(outimg)

    #diff_image
    outimg = np.array(outimg).astype(np.float32)
    gt_image = np.array(gt_image).astype(np.float32)
    # diffimg = np.abs(outimg-gt_image)
    diff = cv2.absdiff(outimg,gt_image)
    diff[:,:,0],diff[:,:,2] = 0,0
    
    diff[diff > 10]= 255
    diff[diff <= 10] = 0
    with diff_image:
        st.header("Diff")
        st.image(diff,clamp=True)


    col3,col4,col5 = st.columns(3)
    #elliptical
    elliptical = mask_scale_elliptical.squeeze(0).permute(1,2,0)
    elliptical = elliptical.mul(255)
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


    


