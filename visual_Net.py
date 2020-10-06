from keras.engine import Model
from vis.input_modifiers import Jitter
from vis.losses import ActivationMaximization
from vis.optimizer import Optimizer
from vis.callbacks import GifGenerator
from vis.regularizers import LPNorm, TotalVariation

# Build the VGG16 network with ImageNet weights
#coding=utf-8
from keras import optimizers
from keras.layers import Dense, Flatten, Dropout, Concatenate, Input, K, Lambda, Reshape
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.utils.np_utils import to_categorical
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from vis.utils import utils
from matplotlib import pyplot as plt
from vis.visualization import visualize_activation
import Image
import  os
# dimensions of our images.

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

img_width, img_height = 227, 227



def lrn(x, radius, alpha, beta, bias=1.0):
    """Create a local response normalization layer."""
    return K.tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias)

if __name__ == '__main__':

    print("Ready to Visualizing the different activation functions ......")

    inputs = Input(shape=(img_width,img_height,3))

    # CNNB alexnet
    x_B = Conv2D(96, (3, 3), strides=(2, 2), padding='valid', activation='relu', kernel_initializer='uniform')(inputs)
    x_B = Conv2D(96, (3, 3), strides=(1, 1), padding='valid', activation='relu', kernel_initializer='uniform')(x_B)
    x_B = Lambda(lrn, output_shape=[111, 111, 96], arguments={'radius': 2, 'alpha': 2e-05, 'beta': 0.75})(x_B)
    x_B = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x_B)
    # Layer2
    x_B = Conv2D(128, (3, 3), strides=(1, 1), padding='valid', activation='relu', kernel_initializer='uniform')(x_B)
    x_B = Conv2D(128, (3, 3), strides=(1, 1), padding='valid', activation='relu', kernel_initializer='uniform')(x_B)
    x_B = Lambda(lrn, output_shape=[51, 51, 128], arguments={'radius': 2, 'alpha': 2e-05, 'beta': 0.75})(x_B)
    x_B = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x_B)

    # Layer3
    x_B = Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x_B)
    x_B = Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x_B)
    x_B = Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x_B)
    x_B = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x_B)

    # layer4
    x_B = Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x_B)
    x_B = Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x_B)
    x_B = Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x_B)
    x_B = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x_B)

    # x_B = Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform')(x_B)
    # x_B = Lambda(lrn, output_shape=[27 ,27,256],arguments={'radius':2,'alpha':2e-05,'beta':0.75})(x_B)
    # x_B = MaxPooling2D(pool_size=(3,3),strides=(2,2))(x_B)
    # Layer5
    x_B = Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x_B)
    x_B = Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x_B)
    x_B = Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x_B)
    x_B = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x_B)

    x_B = Flatten()(x_B)
    x_F = Dropout(0.7)(x_B)
    x_F = Dense(10)(x_F)
    # x_F = Dense(10)(x_F)
    # create model
    model = Model(inputs, x_F, name='VGG16')
    model.load_weights("/raid/Wei/Codes/TestCode/PartitioningCNN/FeaturesAnalysis/FeaturesClustering_Visualization/ImageNet_Ours.h5")

    #visualize
    # vis_images = np.load("/raid/Wei/Codes/TestCode/VisualInteractionCNN/Visualization/ImageNet_Visualization/new_image1.npy")
    image = np.array(Image.open(r'/raid/Wei/Codes/TestCode/PartitioningCNN/FeaturesAnalysis/FeaturesClustering_Visualization/Selected_Images/5.jpg'))
    # ZoomImg = image.resize((227, 227), Image.ANTIALIAS)
    # img = Image.open()
    new_vis_images = []
    import scipy.misc
    layer_idx = utils.find_layer_idx(model, 'max_pooling2d_1')   # ATA: lambda_5;   A:flatten_1
    # layer_idx = utils.find_layer_idx(model, 'flatten_1')
    for i in range(0,96):
        img  = visualize_activation(model, layer_idx, seed_input=image,filter_indices=i, max_iter=1500,
                                    verbose = True,input_modifiers=[Jitter(8)], tv_weight=0.)
        # image_name = "/raid/Wei/Codes/TestCode/VisualInteractionCNN/Visualization/ImageNet_Visualization/Results/2/A_5.jpg"
        image_name = "/raid/Wei/Codes/TestCode/PartitioningCNN/FeaturesAnalysis/FeaturesClustering_Visualization/"+str(i)+".jpg"
        # image_name = "/raid/Wei/Codes/TestCode/VisualInteractionCNN/Visualization/ImageNet_Visualization/Results/1/A" + str(1) + ".jpg"
    # save image
        scipy.misc.imsave(image_name, img)
        new_vis_images.append(img)
    np.save("new_vis_images2",new_vis_images)