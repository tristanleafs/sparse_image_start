import tensorflow as tf
import numpy as np
from tensorflow.keras import activations
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers


def call_splat_conv2d(input, kernel):
    print(input)
    print(input.shape)
    print(input[0])
    print(input[0][2])
    print(input[0,1])
    print(input[0][0,1])
    print(input[0][0])
    placeholder = np.zeros(shape=input[0].shape)
    print(placeholder)
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
    def convolution_op(self, inputs, kernel):
        print("hello")
        return call_splat_conv2d(inputs, kernel)
        # mean, var = tf.nn.moments(kernel, axes=[0, 1, 2], keepdims=True)
        # return tf.nn.conv2d(
        #     inputs,
        #     (kernel - mean) / tf.sqrt(var + 1e-10),
        #     padding="VALID",
        #     strides=list(self.strides),
        #     name=self.__class__.__name__,
        #     )


# layer = splatter(10 ,3, input_shape=(28,28,3,9))
# print(layer)
# pop = layer(tf.zeros([28,28,3,9]))
# print([var.name for var in layer.trainable_variables])
