from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from keras.preprocessing.image import load_img, img_to_array
from inpainter_utils.pconv2d_data import DataGenerator, torch_preprocessing, torch_postprocessing
from inpainter_utils.pconv2d_model import pconv_model
import matplotlib.pyplot as plt
import numpy as np

# SETTINGS:
IMG_DIR_TRAIN   = "data/images/train/"
IMG_DIR_VAL     = "data/images/validation/"
IMG_DIR_TEST    = "data/images/test/"
VGG16_WEIGHTS   = "data/vgg16_weights/vgg16_pytorch2keras.h5"
WEIGHTS_DIR     = "callbacks/weights/"
TB_DIR          = "callbacks/tensorboard/"
CSV_DIR         = "callbacks/csvlogger/"
BATCH_SIZE      = 5
STEPS_PER_EPOCH = 2500
EPOCHS_STAGE1   = 70
EPOCHS_STAGE2   = 50
LR_STAGE1       = 0.0002
LR_STAGE2       = 0.00005
STEPS_VAL       = 100
BATCH_SIZE_VAL  = 4
IMAGE_SIZE      = (512, 512)
LAST_CHECKPOINT =  WEIGHTS_DIR + "fine_tuning/weights.21-2.91-2.99.hdf5"

#model = pconv_model(lr=LR_STAGE1, image_size=IMAGE_SIZE, vgg16_weights=VGG16_WEIGHTS)

model = pconv_model(predict_only=True, image_size=IMAGE_SIZE)
model.load_weights(LAST_CHECKPOINT)
k = 1

img_fname  = "data/examples/own_image.jpg"
mask_fname = "data/examples/own_mask.png"
# Mask is assumed to have masked pixels in black and valid pixels in white

# Loading and pre-processing:
orig_img = img_to_array(load_img(img_fname, target_size=IMAGE_SIZE))
orig_img = orig_img[None,...] 

mask = load_img(mask_fname, target_size=IMAGE_SIZE)
mask = (img_to_array(mask) == 255).astype(np.float32)
mask = mask[None,...] 

# Prediction:
output_img = model.predict([torch_preprocessing(orig_img.copy()) * mask, mask])

# Post-processing:
output_img  = torch_postprocessing(output_img)
input_img   = orig_img * mask
output_comp = input_img.copy()
output_comp[mask == 0] = output_img[mask == 0]

# Plot:
fig, axes = plt.subplots(2, 2, figsize=(20,20))
axes[0,0].imshow(orig_img[0].astype('uint8'))
axes[0,0].set_title('Original image')
axes[0,1].imshow(mask[0])
axes[0,1].set_title('Mask')
axes[1,0].imshow(input_img[0].astype('uint8'))
axes[1,0].set_title('Masked image')
axes[1,1].imshow(output_img[0])
axes[1,1].set_title('Prediction')
for ax in axes.flatten():
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
plt.tight_layout()
plt.savefig("data/examples/own_image_result.png", bbox_inches='tight', pad_inches=0)