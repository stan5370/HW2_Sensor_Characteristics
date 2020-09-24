import numpy as np

def simulate_noisy_images(images,gain,sigma_read,sigma_adc,fwc,num_ims=20):
    """
    simulate a set of images captured according to a camera noise model using the parameters estimated from the homework. 
    The input array should be signed integer, representing the number of photons detected at each pixel.  
        
    args:
        images(np.ndarray): (H, W, #colors) array of pixel intensities without noise
        gain(np.ndarray): (#sensitivity) array of estimated camera gains
        sigma2_read(np.ndarray): (#colors) the estimated read noise variance
        sigma2_adc(np.ndarray): (#colors) the estimated adc noise variance
        fwc: the estimated full well capactiy of the sensor
        num_ims: the number of noisy images to simulate 
    output:
        noisy_images(np.ndarray): (H, W, #colors, #images, #sensitivity)
    """
    
    raise NotImplementedError