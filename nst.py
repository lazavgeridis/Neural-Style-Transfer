'''
 - Implementation of the paper: Leon A. Gatys, Alexander S. Ecker, Matthias Bethge, (2015). A Neural Algorithm of Artistic Style
 - Code is heavily inspired by one of the deeplearning.ai' s deep learning specialization assignments (Course 4 - Week 4).

'''

import os
import sys
import matplotlib.pyplot as plt 
from matplotlib.pyplot import imshow 
from PIL import Image
from nst_utils import *
import numpy as np
import imageio

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
    
#%matplotlib inline


def compute_content_cost(a_C, a_G):
    """
    Computes the content cost

    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

    Returns:
    J_content -- scalar that you compute
    """

    # Retrieve dimensions from a_G
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape a_C and a_G 
    a_C_unrolled = tf.reshape(a_C, [n_H * n_W, n_C])
    a_G_unrolled = tf.reshape(a_G, [n_H * n_W, n_C])

    # compute the cost
    c1 = 1 / (4  * n_H * n_W * n_C)               # option a) the smallest coefficient
    c2 = 1 / (2 * n_H**0.5 * n_W**0.5 * n_C**0.5) # option b) c2 = sqrt(c1) => c2 > c1
    c3 = 0.5                                      # option c) what the authors of the paper used - it seems to preserve the content better

    c = c3
    J_content = c * tf.reduce_sum(tf.pow((a_C_unrolled - a_G_unrolled), 2))

    return J_content


def gram_matrix(A): 
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)
    
    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    
    GA = tf.matmul(A, tf.transpose(A))
    
    return GA


def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

    Returns:
    J_style_layer -- tensor representing a scalar value, style cost
    """

    # Retrieve dimensions from a_G
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape the images to have them of shape (n_C, n_H*n_W)
    #a_S = tf.reshape(a_S, [n_C, n_H * n_W])
    #a_G = tf.reshape(a_G, [n_C, n_H * n_W])
    a_S = tf.reshape(a_S, (n_H * n_W, n_C))
    a_G = tf.reshape(a_G, (n_H * n_W, n_C))

    # Computing gram_matrices for both images S and G
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss
    J_style_layer = (1. / (4 * n_H **2 * n_W**2 * n_C**2)) * tf.reduce_sum(tf.pow((GS - GG), 2))

    return J_style_layer


# We use the style layers and weights that are advised in the paper 
STYLE_LAYERS = [        
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]


def compute_style_cost(model, sess, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers

    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them

    Returns:
    J_style -- tensor representing a scalar value, style cost
    """

    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:

        # Select the output tensor of the currently selected layer
        out = model[layer_name]

        # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name]
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out

        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style


def total_cost(J_content, J_style, J_tv, alpha = 10, beta = 40, total_variation_weight = 1):
#def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function
    
    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost
    
    Returns:
    J -- total cost as defined by the formula
    """
    
    J = alpha * J_content + beta * J_style + total_variation_weight * J_tv
    
    return J


def model_nn(sess, model, train_step, J, J_content, J_style, input_image, num_iterations = 1000):
    
    # Initialize global variables
    sess.run(tf.global_variables_initializer())
    
    # Run the noisy input image (initial generated image) through the model.
    sess.run(model['input'].assign(input_image))
    
    for i in range(num_iterations):
    
        # Run the session on the train_step to minimize the total cost
        sess.run(train_step)
        
        # Compute the generated image by running the session on the current model['input']
        generated_image = sess.run(model['input'])

        # Print every x iteration.
        if i%100 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            
            # save current generated image in the "/output" directory
            save_image("output/" + str(i) + ".png", generated_image)
    
    # save last generated image
    #generated_image = generated_image.reshape((400, 300, 3))
    save_image('output/generated_image.jpg', generated_image)
    
    return generated_image


def main():
    # Reset the graph
    tf.reset_default_graph()
    
    # Start interactive session
    sess = tf.InteractiveSession()
    
    content_image = imageio.imread("images/dit_500x400.jpg")
    content_image = reshape_and_normalize_image(content_image)
    #imshow(content_image)

    style_image = imageio.imread("images/styles/scream_500x400.jpg")
    style_image = reshape_and_normalize_image(style_image)
    #imshow(style_image)

    generated_image = generate_noise_image(content_image)
    imshow(generated_image[0])
    model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
    
    # Assign the content image to be the input of the VGG model.
    sess.run(model['input'].assign(content_image))
    
    # Select the output tensor of layer conv4_2
    out = model['conv4_2']
    
    # Set a_C to be the hidden layer activation from the layer we have selected
    a_C = sess.run(out)
    
    # Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2']
    # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
    # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
    a_G = out
    
    # Compute the content cost
    J_content = compute_content_cost(a_C, a_G)
    
    # Assign the input of the model to be the "style" image
    sess.run(model['input'].assign(style_image))
    
    # Compute the style cost
    J_style = compute_style_cost(model, sess, STYLE_LAYERS)

    # Denoising loss
    J_tv = tf.image.total_variation(model['input'])

    a = 5
    b = 500000
    tv_weight = 1e4
    learn_rate = 10.0
    
    print("a = %d\nb = %d\ntotal variation weight = %f\nlearning_rate = %f" %(a, b, tv_weight, learn_rate))
    #print("a = %d\nb = %d\nlearning_rate = %f" %(a, b, learn_rate))
    
    # Compute the total cost
    #J = total_cost(J_content, J_style, alpha = a, beta = b)
    J = total_cost(J_content, J_style, J_tv, alpha = a, beta = b, total_variation_weight = tv_weight)
    
    # define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)
    #optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
    
    # define train_step
    train_step = optimizer.minimize(J)
    
    model_nn(sess, model, train_step, J, J_content, J_style, generated_image, num_iterations = 3000)
    
    sess.close()


if __name__ == '__main__':
  main()
