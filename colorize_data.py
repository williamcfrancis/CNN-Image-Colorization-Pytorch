import torchvision.transforms as T
import torch
import numpy as np
from skimage.color import rgb2lab, rgb2gray
from torchvision import datasets

class ColorizeData(datasets.ImageFolder):

  def __getitem__(self, index):
      path,_ = self.imgs[index] # Get the path to the image and its label based on the index
      input = self.loader(path) # Load the image data using the self.loader function
      input = self.transform(input) # Apply data augmentation/normalization using the self.transform function
      input = np.asarray(input) # Convert the input data to a numpy array
      img_lab = rgb2lab(input) # Convert the input image from RGB to LAB color space
      img_lab = (img_lab + 128) / 255 # Normalize the LAB values so that they are in the range of 0 to 1
      img_ab = img_lab[:, :, 1:3] # Get the "ab" channels from the LAB image
      img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float() # Convert the "ab" channels to a PyTorch tensor
      input = rgb2gray(input) # Convert the input image from RGB to grayscale
      input = torch.from_numpy(input).unsqueeze(0).float() # Convert the grayscale image to a PyTorch tensor and add a batch dimension
      return input, img_ab # Return the grayscale image and the "ab" channels as a tuple
