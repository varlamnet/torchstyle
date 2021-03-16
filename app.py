import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import os
os.chdir(os.path.dirname(__file__))
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import copy
import time 
import webbrowser
################
### SIDEBARS ###
################
if st.sidebar.button('💻 Code'):
    webbrowser.open_new_tab('https://github.com/varlamnet/torchstyle')

option1 = st.sidebar.selectbox('Content image', ('Capitol', 'My image'))

if option1 == 'Capitol':
    option1 = Image.open(f"images/{option1}.jpg")
else:
    option1 = st.sidebar.file_uploader("Upload 800x800 image")
    if option1 is not None:
        option1 = Image.open(option1)
    else:
        option1 = Image.open("images/Capitol.jpg")

option2 = st.sidebar.selectbox('Style image', ('Scream', 'My image'))

if option2 == 'Scream':
    option2 = Image.open(f"images/{option2}.jpg")
else:
    option2 = st.sidebar.file_uploader("Upload Upload 800x800 image ")
    if option2 is not None:
        option2 = Image.open(option2)
    else:
        option2 = Image.open("images/Scream.jpg")

optm = st.sidebar.radio("Optimizer", ('RMSprop', 'LBFGS', 'Adam'))
Steps = st.sidebar.slider('Steps', 20, 600, 100, 20)
imsize = st.sidebar.slider('Output image quality', 50, 800, 250, 50)

############
### MAIN ###
############
st.markdown("# ✦ Neural Transfer ✦")
st.markdown('## PyTorch Edition')

st.write('Select images or upload your own (must be 800x800)')

col1, col2 = st.beta_columns(2)
with col1:
    st.image(option1, width = 250, caption = "Content image")
with col2:
    st.image(option2, width = 250, caption = "Style image")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

@st.cache(show_spinner=False)
def image_loader(image):
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

content_img = image_loader(option1)
style_img = image_loader(option2)
try:
    assert style_img.size() == content_img.size()
except AssertionError:
    st.markdown('# Image dimensionality problem!!!')
unloader = transforms.ToPILImage()  # reconvert into PIL image

class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute the gram product
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

cnn = models.alexnet(pretrained=True).features.to(device).eval()
# cnn = models.vgg19(pretrained=True).features.to(device).eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

input_img = content_img.clone()

def get_input_optimizer(input_img):
    if optm == "RMSprop":
        optimizer = optim.RMSprop([input_img.requires_grad_()])
    elif optm == "LBFGS":
        optimizer = optim.LBFGS([input_img.requires_grad_()])
    elif optm == "Adam":
        optimizer = optim.Adam([input_img.requires_grad_()])
    return optimizer

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=Steps,
                       style_weight=1000000, content_weight=1):
    st.write('Compiling the model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    st.write('Optimizing AlexNet..')
    run = [0]
    my_bar = st.progress(0)
    while run[0] < num_steps:

        def closure():
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()
            
            my_bar.progress(run[0]/(num_steps))
            
            run[0] += 1
            return style_score + content_score

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)

    return input_img

st.markdown('### See the result')
if st.button('Train 🎈'):
    st.write('*This will take a few minutes max.*')
    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img)

    st.markdown('### Result')
    output1 = transforms.ToPILImage()(output[0,:,:,:].cpu())
    st.image(output1, width=imsize, caption = "Stylized image")
