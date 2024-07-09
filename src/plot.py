import numpy as np
import glob
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_with_colorbar(img,vmax=0):
    """
    args:
        vmax: The maximal value to be plotted
    """
    ax = plt.gca()
    if(vmax == 0):
        im = ax.imshow(img, cmap= 'gray')
    else:
        im = ax.imshow(img, cmap= 'gray',vmax=vmax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    

def plot_input_histogram(imgs,sensitivity):
    """
    
    The imgs variable consists of 1 image captured per different camera sensitivity (ISO) settings. plot_input_histogram
    visualize the histograms for each image in a subplot fashion

    
    args:
        imgs(np.ndarray): 3-dimensional array containing one image per intensity setting (not all the 200)
    
    """
    num_sensitivities = len(sensitivity)
    num_cols = (num_sensitivities + 1) // 2  # Calculate the number of columns for two rows
    fig, axes = plt.subplots(2, num_cols, sharey=True)
    
    for i in range(num_sensitivities):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col]
        ax.hist(imgs[:, :, i].ravel(), bins=100, alpha=0.5, range=(0, 254))
        ax.set_title(f'Sensitivity {sensitivity[i]}')
        ax.set_xlabel('Intensity')
        if col == 0:
            ax.set_ylabel('Count')
        ax.grid(True)
    
    # Hide any unused subplots
    for j in range(num_sensitivities, 2 * num_cols):
        fig.delaxes(axes[j // num_cols, j % num_cols])
    

def plot_histograms_channels(img,sensitivity):
    """
    
    Plots the histogram for each channel in a subplot (1 row, 3 cols)
    
    args:
        img(np.ndarray): The RGB image
        sensitivity(float): The gain settings of the img series
    
    """
    channel_titles = ['Red Channel', 'Green Channel', 'Blue Channel']
    channel_colors = ['red', 'green', 'blue']
    
    fig, axes = plt.subplots(1, 3, sharey=True)
    
    for i in range(3):
        ax = axes[i]
        ax.hist(img[:, :, i].ravel(), bins=100, color=channel_colors[i], alpha=0.5, range=(0, 255))
        ax.set_title(channel_titles[i])
        ax.set_xlabel('Intensity')
        if i == 0:
            ax.set_ylabel('Count')
        ax.grid(True)
    
    fig.suptitle(f'Sensitivity {sensitivity}', fontsize=16)
        
def plot_input_images(imgs,sensitivity):
    """
    
    The dataset consists of 1 image captured per different camera sensitivity (ISO) settings. Lets visualize a single image taken at each different sensitivity setting
    
    Hint: Use plot_with_colorbar. Use the vmax argument to have a scale to 255
    (if you don't use the vmax argument)
    
    args:
        imgs(np.ndarray): 3-dimensional array containing one image per intensity setting (not all the 200)
        sensitivity(np.ndarray): The sensitivy (gain) vector for the image database
    
    """
    num_sensitivities = len(sensitivity)
    num_cols = (num_sensitivities + 1) // 2  # Calculate the number of columns for two rows
    fig, axes = plt.subplots(2, num_cols, figsize=(15, 10))
    
    for i in range(num_sensitivities):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col]
        plt.sca(ax)  # Set the current axis
        # Select the first image for each sensitivity
        image_data = imgs[:, :, i]
        plot_with_colorbar(image_data, vmax=255)
        ax.set_title(f'Sensitivity {sensitivity[i]}')
        ax.axis('off')
    
    # Hide any unused subplots
    for j in range(num_sensitivities, 2 * num_cols):
        fig.delaxes(axes[j // num_cols, j % num_cols])
    

def plot_rgb_channel(img, sensitivity):
    fig, axes = plt.subplots(1, 3)
    channel_titles = ['Red Channel', 'Green Channel', 'Blue Channel']
    channel_colors = ['Reds', 'Greens', 'Blues']
    
    for i in range(3):
        ax = axes[i]
        ax.imshow(img[:, :, i], cmap=channel_colors[i])
        ax.set_title(channel_titles[i])
        ax.axis('off')
    
    fig.suptitle(f'Sensitivity {sensitivity}', fontsize=16)

def plot_images(data, sensitivity, statistic,color_channel):
    """
    this function should plot all 3 filters of your data, given a
    statistic (either mean or variance in this case!)

    args:

        data(np.ndarray): this should be the images, which are already
        filtered into a numpy array.

        statsistic(str): a string of either mean or variance (used for
        titling your graph mostly.)

    returns:

        void, but show the plots!

    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'{statistic.capitalize()} Images for {["Red", "Green", "Blue"][color_channel]} Channel')

    color_maps = ['Reds', 'Greens', 'Blues']
    
    for i, sens in enumerate(sensitivity):
        im = axes[i].imshow(data[:, :, i], cmap=color_maps[color_channel])
        axes[i].set_title(f'Sensitivity: {sens}')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
    
    
def plot_relations(means, variances, skip_pixel, sensitivity, color_idx):
    """
    this function plots the relationship between means and variance. 
    Because this data is so large, it is recommended that you skip
    some pixels to help see the pixels.

    args:
        means: contains the mean values with shape (200x300x3x6)
        variances: variance of the images (200x300x3x6)
        skip_pixel: amount of pixel skipped for visualization
        sensitivity: sensitivity array with 1x6
        color_idx: the color index (0 for red, 1 green, 2 for blue)

    returns:
        void, but show plots!
    """
    raise NotImplementedError
        
def plot_mean_variance_with_linear_fit(gain,delta,means,variances,skip_points=50,color_channel=0):
    """
        this function should plot the linear fit of mean vs. variance against a scatter plot of the data used for the fitting 
        
        args:
        gain (np.ndarray): the estimated slopes of the linear fits for each color channel and camera sensitivity

        delta (np.ndarray): the estimated bias/intercept of the linear fits for each color channel and camera sensitivity

        means (np.ndarray): the means of your data in the form of 
        a numpy array that has the means of each filter.

        variances (np.ndarray): the variances of your data in the form of 
        a numpy array that has the variances of each filter.
        
        skip_points: how many points to skip so the scatter plot isn't too dense
        
        color_channel: which color channel to plot

    returns:
        void, but show plots!
    """
    raise NotImplementedError
    
def plot_read_noise_fit(sigma_read, sigma_ADC, gain, delta, color_channel=0):
    """
        this function should plot the linear fit of read noise delta vs. gain plotted against the data used for the fitting 
        
        args:
        sigma_read (np.ndarray): the estimated gain-depdenent read noise for each color channel of the sensor 

        sigma_ADC (np.ndarray): the estimated gain-independent read noise for each color channel of the sensor

        gain (np.ndarray): the estimated slopes of the linear fits of mean vs. variance for each color channel and camera sensitivity

        delta (np.ndarray): the estimated bias/intercept of the linear fits of mean vs. variance for each color channel and camera sensitivity

        color_channel: which color channel to plot
        
    returns:
        void, but show plots!
    """
    
    raise NotImplementedError