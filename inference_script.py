import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage.color import lab2rgb, rgb2gray
import numpy as np
import os
import cv2

def to_rgb(grayscale_input, ab_input):
  # Show/save rgb image from grayscale and ab channels
  plt.clf() # clear matplotlib 
  color_image = torch.cat((grayscale_input, ab_input), 0).numpy() # combine channels
  color_image = color_image.transpose((1, 2, 0))  # rescale for matplotlib
  color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
  color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128   
  color_image = lab2rgb(color_image.astype(np.float64))
  plt.imsave(arr=color_image, fname='inference/inference_output.jpg')

if __name__ == '__main__':
    os.makedirs('inference/', exist_ok=True)  
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', default = 'models/saved_model.pth', 
                        type = str, help = 'Path to the saved model')

    parser.add_argument('--image_path', default = 'inference/test_img.jpg', 
                        type = str, help = 'Path to the grayscale test image')

    args = parser.parse_args()
    print('Beginning Inference')
    model = torch.load(args.model_path)
    input_gray = cv2.imread(args.image_path)
    input_gray = cv2.resize(input_gray, (256,256))
    input_gray = rgb2gray(input_gray)
    input_gray = torch.from_numpy(input_gray).unsqueeze(0).float()
    input_gray = torch.unsqueeze(input_gray, dim=0).cuda()
    model.eval()
    output_ab = model(input_gray)
    to_rgb(input_gray[0].cpu(), output_ab[0].detach().cpu())
    print("Colorized image saved at 'inference/inference_output.jpg'")

    
