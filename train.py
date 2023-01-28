import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from colorize_data import ColorizeData
from skimage.color import lab2rgb
import time
from CNN_model import Net
import torch.nn as nn
import argparse
import torchvision.transforms as T
import cv2
import glob

class AverageMeter(object):
  # A handy class from the PyTorch ImageNet tutorial
  def __init__(self):
    self.reset()
  def reset(self):
    self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


class Trainer:
    def __init__(self):
        pass
        

    def to_rgb(self, grayscale_input, ab_input, save_path=None, save_name=None):
      # Show/save rgb image from grayscale and ab channels
      plt.clf() # clear matplotlib 
      color_image = torch.cat((grayscale_input, ab_input), 0).numpy() # combine channels
      color_image = color_image.transpose((1, 2, 0))  # rescale for matplotlib
      color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
      color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128   
      color_image = lab2rgb(color_image.astype(np.float64))

      grayscale_input = grayscale_input.squeeze().numpy()
      if save_path is not None and save_name is not None: 
        plt.imsave(arr=grayscale_input, fname='{}{}'.format(save_path['grayscale'], save_name), cmap='gray')
        plt.imsave(arr=color_image, fname='{}{}'.format(save_path['colorized'], save_name))


    def train(self, train_loader, epoch, model, criterion, optimizer, scheduler):
      print('Starting training epoch {}'.format(epoch+1))
      model.train()
      # Prepare value counters and timers
      batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()
      end = time.time()

      for i, (input_gray, input_ab) in enumerate(train_loader):

        input_gray, input_ab = input_gray.cuda(), input_ab.cuda()
        data_time.update(time.time() - end) # Record time to load data

        # Run forward pass
        output_ab = model(input_gray) 
        loss = criterion(output_ab, input_ab) 
        losses.update(loss.item(), input_gray.size(0))

        # Compute gradients and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record time to do forward and backward passes
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 2 == 0:
          print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.6f} ({loss.avg:.6f})\t'.format(
                  epoch+1, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses)) 

      print('Finished training epoch {}'.format(epoch+1))

    def validate(self, val_loader, epoch, save_images, model, criterion):
      model.eval()
      # Prepare value counters and timers
      batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()
      end = time.time()
      already_saved_images = False

      for i, (input_gray, input_ab) in enumerate(val_loader):
        data_time.update(time.time() - end)
        input_gray, input_ab = input_gray.cuda(), input_ab.cuda()

        # Run model and record loss
        output_ab = model(input_gray)
        loss = criterion(output_ab, input_ab)
        losses.update(loss.item(), input_gray.size(0))

        # Save images to file
        if save_images and not already_saved_images:
          already_saved_images = True
          for j in range(min(len(output_ab), 10)): # save 10 images each epoch
            save_path = {'grayscale': 'outputs/gray/', 'colorized': 'outputs/color/', 'ground_truth': 'outputs/ground_truth/'}
            save_name = 'img-{}-epoch-{}.jpg'.format(i * val_loader.batch_size + j, epoch+1)
            self.to_rgb(input_gray[j].cpu(), ab_input=output_ab[j].detach().cpu(), save_path=save_path, save_name=save_name)

        # Record time to do forward passes and save images
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 25 == 0:
          print('Validate: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.6f} ({loss.avg:.6f})\t'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses))

      print('Finished validation.')
      return losses.avg

def clean_train_imgs(folder_path): # remove images with no color

  for filename in os.listdir(folder_path):
      file_path = os.path.join(folder_path, filename)
      if os.path.isfile(file_path):
        image = cv2.imread(file_path)
        if image is not None:
          image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
          if(image_hsv[:,:,0].sum()==0 and image_hsv[:,:,1].sum()==0):
            os.remove(file_path)
            print('Removed image: {}'.format(file_path))

if __name__ == "__main__":
    #Parsing arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='landscape_images/',
                        help='Directory containing all images in the dataset')

    parser.add_argument('--n_val', type=int, default=100,
                        help='Number of images for validation')

    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')

    parser.add_argument('--save_images', type=bool, default=True,
                        help='Whether to save input and output images during validation')
    
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for training')

    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay value for Adam optimizer')
    
    parser.add_argument('--save_model', type=bool, default=True,
                        help='Whether to save the model after training')

    parser.add_argument('--loss', type=str, default='mse',
                        help='Choose between MAE or MSE Loss for training')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training and validation')

    args = parser.parse_args()

    #Splitting images into train and validation
    # os.makedirs('images/train/class', exist_ok=True) 
    # os.makedirs('images/val/class', exist_ok=True)   
    # for i, file in enumerate(os.listdir(args.image_dir)):
    #     if i < args.n_val: # first n_val images will be val
    #         os.rename(args.image_dir + file, 'images/val/class/' + file)
    #     else: # others will be train
    #         os.rename(args.image_dir + file, 'images/train/class/' + file)
    
    # # Make folders
    # os.makedirs('outputs/color', exist_ok=True)
    # os.makedirs('outputs/gray', exist_ok=True)
    # os.makedirs('models', exist_ok=True)
    files = glob.glob('outputs/color/*')
    for f in files:
        os.remove(f)
    files2 = glob.glob('outputs/gray/*')
    for f in files2:
        os.remove(f)

    # # Clean images
    # clean_train_imgs("./images/train/class/")
    # clean_train_imgs("./images/val/class/")

    model = Net().cuda()

    if args.loss=='mse': # Initialize loss according to choice
        criterion = nn.MSELoss().cuda()
    else:
        criterion = nn.L1Loss().cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5, verbose=False)
    # Training
    train_transforms = T.Compose([T.CenterCrop(224)])
    train_imagefolder = ColorizeData('images/train', train_transforms)
    train_loader = torch.utils.data.DataLoader(train_imagefolder, batch_size=args.batch_size, shuffle=True)

    # Validation 
    val_transforms = T.Compose([T.Resize(256), T.CenterCrop(224)])
    val_imagefolder = ColorizeData('images/val' , val_transforms)
    val_loader = torch.utils.data.DataLoader(val_imagefolder, batch_size=args.batch_size, shuffle=False)
    
    print("Image preprocessing completed!")
    # Train model
    for epoch in range(args.epochs):
        # Train for one epoch, then validate
        Trainer().train(train_loader, epoch, model, criterion, optimizer, scheduler)
        scheduler.step()
        with torch.no_grad():
            Trainer().validate(val_loader, epoch, args.save_images, model, criterion)

    if args.save_model==True: # Save the final model
        torch.save(model, 'models/saved_model.pth')

