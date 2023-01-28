
# Image Colorization

## Requirements
To run the code, the environment requires all packages required for the starter code provided. 
In addition, please install skimage using the following command:

'$ conda install scikit-image'

## How to train the model
- Please download the dataset zip file from https://drive.google.com/file/d/15jprd8VTdtIQeEtQj6wbRx6seM8j0Rx5/view?usp=sharing and extract it outside the current directory. The directory tree should now look as follows:

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
           ...
```
Run train.py

Use arguments from the command line to set the parameters. The following arguments are available:
    --image_dir : Directory containing all images in the dataset
    --n_val : Number of images for validation
    --epochs : Number of training epochs
    --save_images : Whether to save input and output images during validation
    --lr : Learning rate for training
    --weight_decay : Weight decay value for Adam optimizer
    --save_model : Whether to save the model after training
    --loss : Choose between 'mae' or 'mse' Loss for training
    --batch_size : Batch size for training and validation

When train.py is run, a folder /Outputs/ is created with subfolders /Color/ and /Gray/. During training, after each epoch, validation results are saved in the /Color/ folder and their corresponding inputs are saved in /Gray/ folder.

Running train.py also creates an /Images/ folder that has training and validation images separated into different folders /train/class/ and /val/class/.

Once the training is completed, and if save_model is enabled, the final model is saved in a folder /Models/ as a .pth file.

## How to train the model with color temperature control

Run train_hue_control.py and follow the same instructions as above. 
The color temperature can be adjusted in colorize_data_hue_control.py in line 14 by changing the hue value.

## How to run inference on a grayscale image

After training, the model will be saved in /Models/ as a .pth file by default.

Place the test image in /inference/ folder and name it 'test_img.jpg'. (can be changed by passing arguments)

Run inference_script.py

Use arguments from the command line to set the parameters. The following arguments are available:
    --model_path : Path to the saved model
    --image_path : Path to the grayscale test image

The inference output can be found in the /inference/ folder as 'inference_output.jpg'

For any further questions, please email me at willcf@seas.upenn.edu
