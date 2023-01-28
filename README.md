# Image Colorization
## Requirements

- PyTorch (>= 1.7.0)
- Numpy (>= 1.19.3)
- Matplotlib (>= 3.3.3)
- Scikit-Image (>= 0.18.3)
- TorchVision (>= 0.8.1)
- OpenCV-Python (>= 4.5.4)

It is recommended to use a virtual environment for the project to manage the dependencies. The dependencies can be installed by running the following command:
`pip install torch numpy matplotlib scikit-image torchvision opencv-python`

## How to Train the Model

1. Clone this repository using `git clone https://github.com/williamcfrancis/CNN-Image-Colorization-Pytorch.git`

Download the dataset zip file from https://drive.google.com/file/d/15jprd8VTdtIQeEtQj6wbRx6seM8j0Rx5/view?usp=sharing and extract it outside the current directory. The directory structure should look like:

```
│
└───Image_colorization_William
│      train.py
│      basic_model.py
|      colorize_data.py
|      inference_script.py
|      train_hue_control.py
|      colorize_data_hue_control.py
|      Report.pdf
|      README.md  <-- you are here
|
└───train_landscape_images
    │ 
    └─landscape_images
           1.jpg
           2.jpg
```

2. Run train.py with the following arguments:
3. 
`--image_dir`: Directory containing all images in the dataset\
`--n_val`: Number of images for validation\
`--epochs`: Number of training epochs\
`--save_images`: Whether to save input and output images during validation\
`--lr`: Learning rate for training\
`--weight_decay`: Weight decay value for Adam optimizer\
`--save_model`: Whether to save the model after training\
`--loss`: Choose between 'mae' or 'mse' Loss for training\
`--batch_size`: Batch size for training and validation

- The training creates a `/Outputs/` folder with subfolders `/Color/` and `/Gray/`. Validation results are saved in `/Color/` and inputs in `/Gray/`.

- The training also creates an `/Images/` folder with `train/val` images separated into different folders.

- If `save_model` is enabled, the final model is saved in a /Models/ folder as a .pth file.

## How to Train with Color Temperature Control

- Run train_hue_control.py and follow the training instructions above.
- The color temperature can be adjusted in colorize_data_hue_control.py by changing the hue value.

## Inference
To run inference on a grayscale image, the saved model can be used. The following steps outline how to perform inference:

1. Place the test image in the /inference/ folder and name it test_img.jpg (this can be changed by passing arguments).
2. Run the inference_script.py file.
3. Use arguments from the command line to set the parameters. The following arguments are available:\
--model_path: Path to the saved model.\
--image_path: Path to the grayscale test image.\
4. The inference output can be found in the /inference/ folder as inference_output.jpg.
