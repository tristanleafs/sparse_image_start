import tensorflow as tf
import numpy as np


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


class splatter(tf.keras.layers.Layer):
    def __init__(self,filters,kernel_size,input_shape):
        self.num_outputs = 1
        self.filters = filters
        self.kernel_size = kernel_size
        self.input_shape = input_shape
    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",
         shape= [self.kernel_size,self.kernel_size])

    def call(self, input):
        return splat_conv2d(input, self.kernel)