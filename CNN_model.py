import torch.nn as nn
import torchvision.models as models


class Net(nn.Module):
  def __init__(self, input_size=128):
    super(Net, self).__init__()
    resnet = models.resnet34(num_classes=1000)
    resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1)) 
    self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0:6])
    RESNET_FEATURE_SIZE = 128
    # Upsampling Network
# self.upsample = nn.Sequential(     
#   nn.Conv2d(RESNET_FEATURE_SIZE, 128, kernel_size=3, stride=1, padding=1),
#   nn.BatchNorm2d(128),
#   nn.ReLU(),
#   nn.Upsample(scale_factor=2),
#   nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
#   nn.BatchNorm2d(64),
#   nn.ReLU(),
#   nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#   nn.BatchNorm2d(64),
#   nn.ReLU(),
#   nn.Upsample(scale_factor=2),
#   nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
#   nn.BatchNorm2d(32),
#   nn.ReLU(),
#   nn.Upsample(scale_factor=2),
#   nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
#   nn.BatchNorm2d(16),
#   nn.ReLU(),
#   nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1),
#   nn.Upsample(scale_factor=2)
# )

    self.upsample = nn.Sequential(     
      nn.Conv2d(RESNET_FEATURE_SIZE, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(16),
      nn.ReLU(),
      nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1), # Output has 2 channels instead of 3 because we are
                                                          # predicting a and b channels of Lab color space instead of RGB
      nn.Upsample(scale_factor=2)
    )

  def forward(self, input):
    midlevel_features = self.midlevel_resnet(input)
    output = self.upsample(midlevel_features)
    return output
