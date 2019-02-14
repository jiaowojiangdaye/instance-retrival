
# coding: utf-8

# Function: Visualization of data augmentation
# 
# Author: Jay Lee

# # 数据增强可视化

# In[3]:


from imgaug import augmenters as iaa
import numpy as np
#import matplotlib
import skimage.io
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')



imgname = "img_00000016.jpg"
image_test_path = "/home/mbzhao/Guests/Jay/Mask_RCNN/datasets/clothes/Img/img/Abstract_Mirrored_Print_Dress/"+imgname
image = skimage.io.imread(image_test_path)
images = np.random.randint(0, 255, (16, 300,201, 3), dtype=np.uint8)
# images = []
flipper = iaa.Fliplr(1) # always horizontally flip each input image
images[0] = flipper.augment_image(image) # horizontally flip image 0

vflipper = iaa.Flipud(1) # vertically flip each input rawimage with 90% probability
images[1] = vflipper.augment_image(image) # probably vertically flip image 1

rotat = iaa.Affine(rotate=45)
images[2] = rotat.augment_image(image) # blur image 2 by a sigma of 3.0
rotat = iaa.Affine(shear=135)
images[3] = rotat.augment_image(image) # blur image 3 by a sigma of 3.0 too
rotat = iaa.Affine(shear=45)
images[4] = rotat.augment_image(image) # blur image 3 by a sigma of 3.0 too

scaler = iaa.Affine(scale={"y": (0.7, 0.7)}) # scale each input image to 80-120% on the y axis
images[5] = scaler.augment_image(image) # scale image 5 by 80-120% on the y axis

scaler = iaa.Affine(scale={"y": (1.3, 1.3)}) # scale each input image to 80-120% on the y axis
images[6] = scaler.augment_image(image) # scale image 5 by 80-120% on the y axis

scaler = iaa.Affine(scale={"x": (0.7, 0.7)}) # scale each input image to 80-120% on the y axis
images[7] = scaler.augment_image(image) # scale image 5 by 80-120% on the y axis

scaler = iaa.Affine(scale={"x": (1.3, 1.3)}) # scale each input image to 80-120% on the y axis
images[8] = scaler.augment_image(image) # scale image 5 by 80-120% on the y axis

mul = iaa.Multiply((0.7, 0.7))
images[9] = mul.augment_image(image)
mul = iaa.Multiply((1.5, 1.5))
images[10] = mul.augment_image(image)

bl = iaa.GaussianBlur(sigma=(1.0, 5.0))
images[11] = bl.augment_image(image)  

translater = iaa.Affine(translate_px={"x": 64}) # move each input image by 16px to the left
images[12] = translater.augment_image(image) # move image 4 to the left
                      
# _, ax = plt.subplots(1, 4, figsize=(4*1, 4*4))
plt.subplot(2, 4, 1)
plt.imshow(image)
plt.subplot(2, 4, 2)
plt.imshow(images[0])
plt.subplot(2, 4, 3)
plt.imshow(images[1])
plt.subplot(2, 4, 4)
plt.imshow(images[2])
plt.subplot(2, 4, 5)
plt.imshow(images[3])
plt.subplot(2, 4, 6)
plt.imshow(images[4])
plt.subplot(2, 4, 7)
plt.imshow(images[5])
plt.subplot(2, 4, 8)
plt.imshow(images[12])

plt.imsave("img/imgaug/raw.jpg",image)
for i in range(13):
    plt.imsave("img/imgaug/{}.jpg".format(i),images[i])

