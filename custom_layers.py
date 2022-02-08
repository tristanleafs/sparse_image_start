import tensorflow as tf
import numpy as np
from tensorflow.keras import activations
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers


def call_splat_conv2d(input, kernel):
    placeholder = np.zeros(shape=input.shape[0])
    for image in input[0]:
        for ind in image.shape:
            placeholder[ind] = splat_conv2d(channel, kernel)
    return tf.constant(placeholder)

def splat_conv2d(input, kernel):
    #this function skips reversing the kernel
    #establish size of input image
    rows = input.shape[0]
    cols = input.shape[1]

    #establish weight kernel size
    weightSize = kernel.shape[0]

    #establish weight indexing
    weightIndex = int(weightSize/2) 

    #pad input matrix
    padInput = np.zeros((rows + weightIndex*2, cols + weightIndex*2))
    padInput[weightIndex:-weightIndex, weightIndex:-weightIndex] = input

    #create padded output image
    output = np.zeros_like(padInput)


    #iterate though input matrix
    for i in range(weightIndex, weightIndex+rows): #iterate through rows of input
        for j in range(weightIndex, weightIndex+cols): #iterate through columns of input
            
            #check if index is nonzero
            if(padInput[i][j] > 0 or padInput[i][j] <0):

                #update output using vector operations
                output[i-weightIndex:i+weightIndex +1,j-weightIndex:j+weightIndex+1] += padInput[i][j]*kernel


    #adjust final output size using splicing
    output = output[weightIndex:-weightIndex, weightIndex:-weightIndex]
    return tf.constant(output)
    #end of function


class splatter(tf.keras.layers.Conv2D):
    def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               groups=1,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
        super(splatter, self).__init__(
            # rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs
        )
        
    
    # def build(self, input_shape):
        
    def call(self, input):
        return call_splat_conv2d(input, self.kernel)


# layer = splatter(10 ,3, input_shape=(28,28,3,9))
# print(layer)
# pop = layer(tf.zeros([28,28,3,9]))
# print([var.name for var in layer.trainable_variables])
