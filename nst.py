'''
 - Implementation of the paper: Leon A. Gatys, Alexander S. Ecker, Matthias Bethge, (2015). A Neural Algorithm of Artistic Style
 - Code is inspired by one of deeplearning.ai' s deep learning specialization assignments (Course 4 - Week 4).

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
    
# We use the style layers that are advised in the paper 
STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

# default arguments
ALPHA         = 5e0
BETA          = 5e4
LEARNING_RATE = 1e1


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
    _, n_H, n_W, n_C = a_G.get_shape().as_list()
    N_l = n_C
    M_l = n_H * n_W

    # Reshape a_C and a_G 
    a_C_unrolled = tf.reshape(a_C, (M_l, N_l))
    a_G_unrolled = tf.reshape(a_G, (M_l, N_l))

    # compute the cost - both options b and c are faster in terms of iterations when it comes to "painting" the content image
    c1 = 0.5                                      # option a - what the authors of the paper used 
    c2 = 1 / (4  * n_H * n_W * n_C)               # option b
    c3 = 1 / (2 * n_H**0.5 * n_W**0.5 * n_C**0.5) # option c

    C = c1
    J_content = C * tf.reduce_sum(tf.pow((a_C_unrolled - a_G_unrolled), 2))

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
    _, n_H, n_W, n_C = a_G.get_shape().as_list()
    N_l = n_C
    M_l = n_H * n_W

    # Reshape the images to have them of shape (n_C, n_H*n_W)
    a_S = tf.reshape(a_S, (M_l, N_l))
    a_G = tf.reshape(a_G, (M_l, N_l))

    # Computing gram_matrices for both images S and G
    GS = gram_matrix(tf.transpose(a_S))
    GG = gram_matrix(tf.transpose(a_G))

    # Computing the loss
    J_style_layer = (1. / (4 * n_H **2 * n_W**2 * n_C**2)) * tf.reduce_sum(tf.pow((GS - GG), 2))

    return J_style_layer


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

    for layer_name in STYLE_LAYERS:

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
        J_style += 1. / len(STYLE_LAYERS) * J_style_layer

    return J_style


def total_cost(J_content, J_style, alpha = 5, beta = 5000):
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
    
    J = alpha * J_content + beta * J_style    

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

        # Print every x iterations
        if i%100 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            
            # save current generated image in the "/output" directory
            save_image("output/" + str(i) + ".png", generated_image)
    
    # save last generated image
    save_image('output/generated_image.jpg', generated_image)
    
    return generated_image


def main():
    # Reset the graph
    tf.reset_default_graph()
    
    # Start interactive session
    sess = tf.InteractiveSession()
    
    content_image = imageio.imread("images/dit_500x400.jpg")
    content_image = reshape_and_normalize_image(content_image)

    style_image = imageio.imread("images/styles/scream_500x400.jpg")
    style_image = reshape_and_normalize_image(style_image)

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
    #J_tv = tf.image.total_variation(model['input'])

    a = ALPHA
    b = BETA
    #tv_weight = 1e4
    lr = LEARNING_RATE
    
    print("a = %d\nb = %d\nlearning_rate = %f" %(a, b, lr))
    
    # Compute the total cost
    J = total_cost(J_content, J_style, alpha = a, beta = b)
    #J = total_cost(J_content, J_style, J_tv, alpha = a, beta = b, total_variation_weight = tv_weight)
    
    # define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-08)
    
    # define train_step
    train_step = optimizer.minimize(J)
    
    model_nn(sess, model, train_step, J, J_content, J_style, generated_image, num_iterations = 1500)
    
    sess.close()


if __name__ == '__main__':
  main()
