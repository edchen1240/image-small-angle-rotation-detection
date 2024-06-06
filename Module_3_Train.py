"""
[ICM_0_module.py]
Purpose: Invert cone model
Author: Meng-Chi Ed Chen
Date: 2024-03-04
Reference:
    1.
    2.

Status: Complete.
"""
import os, sys, cv2, time, shutil, openpyxl, ast
import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from datetime import datetime
from scipy.stats import norm
from scipy.stats import linregress
import piexif
from PIL import Image
from scipy.spatial.distance import euclidean
import warnings
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator


from scipy.signal import savgol_filter
from sklearn.linear_model import RANSACRegressor





#[1] Image translation, rotation, and overlap - Start



def compress_two_arrimgs_to_same_size(arrimg1, arrimg2, ratio):
    if ratio > 3 or ratio < 0.02:
        raise ValueError('Please pick ratio between 0.02 to 3. ratio > 1  will enlarge the image.')
    #[1] Calculate new dimensions
    new_height1 = int(arrimg1.shape[0] * ratio)
    new_width1 = int(arrimg1.shape[1] * ratio)
    new_height2 = int(arrimg2.shape[0] * ratio)
    new_width2 = int(arrimg2.shape[1] * ratio)

    #[2] Resize images to the same dimensions based on the smallest of each dimension
    new_height = min(new_height1, new_height2)
    new_width = min(new_width1, new_width2)
    text = '\n[compress_two_arrimgs_to_same_size]\n'\
            f'--(new_height, new_width)= {new_height}, {new_width}\n'
    print(text)

    #[3] Resize both images
    arrimg1_resized = cv2.resize(arrimg1, (new_width, new_height), interpolation=cv2.INTER_AREA)
    arrimg2_resized = cv2.resize(arrimg2, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return arrimg1_resized, arrimg2_resized



def rotate_image(arrimg, shift_xyr, angle_positive_is_clockwise = True):   # rot_degree is defined as counter-clockwise
    #[1] Define the center of rotation
    ctr_rot_x, ctr_rot_y, rot_degree = shift_xyr
    center = (ctr_rot_x, ctr_rot_y)
    #[2] Calculate the rotation matrix
    if angle_positive_is_clockwise:
        M = cv2.getRotationMatrix2D(center, -rot_degree, 1.0)
    else:
        M = cv2.getRotationMatrix2D(center, rot_degree, 1.0)
    #[3] Rotate the image
    rotated_image = cv2.warpAffine(arrimg, M, (arrimg.shape[1], arrimg.shape[0]))
    return rotated_image


def overlap_two_images_together(arrimg_1, arrimg_2, color_1=[255, 128, 0], color_2=[0, 128, 255]):
    # [1] Convert two images to grayscale.
    arrimg_1_gs = cv2.cvtColor(arrimg_1, cv2.COLOR_BGR2GRAY)
    arrimg_2_gs = cv2.cvtColor(arrimg_2, cv2.COLOR_BGR2GRAY)
    
    # [2] Apply color_1 to arrimg_1_gs, and color_2 to arrimg_2_gs
    # First, convert grayscale images back to color space to apply colors
    arrimg_1_colored = cv2.cvtColor(arrimg_1_gs, cv2.COLOR_GRAY2BGR)
    arrimg_2_colored = cv2.cvtColor(arrimg_2_gs, cv2.COLOR_GRAY2BGR)

    # [3] Apply colors by multiplying the grayscale image with the color array
    # The color needs to be divided by 255 as OpenCV uses 0-1 floating point for operations
    arrimg_1_colored = arrimg_1_colored * (np.array(color_1) / 255.0)
    arrimg_2_colored = arrimg_2_colored * (np.array(color_2) / 255.0)
    text =  f'--Size check\n'\
            f'-- arrimg_1_colored: {arrimg_1_colored.shape}\n'\
            f'-- arrimg_2_colored: {arrimg_2_colored.shape}\n'
    print(text)

    # [4] Add two images arrays together, and normalize pixel value between 0-255.
    arrimg_overlap = cv2.addWeighted(arrimg_1_colored, 1, arrimg_2_colored, 1, 0)
    #arrimg_overlap = np.clip(arrimg_overlap, 0, 255).astype(np.uint8)
    
    return arrimg_overlap


#[1] Image translation, rotation, and overlap - End

















#[2] Loss - Start


def arrimg_subtraction_loss_selector(loss_cal_selection, 
                                        arrimg2, arrimg1_rot, 
                                        proposed_shift, power_of, scope_penalty):
    # loss_cal_selection == 'loss_mse_normal' or 'loss_mse_radial_weighted'
    if loss_cal_selection == 'loss_mse_normal':
        loss, pixel_eff, loss_eff, text = arrimg_subtraction_loss_mse_normal(arrimg2, arrimg1_rot, proposed_shift, power_of, scope_penalty)
    elif loss_cal_selection == 'loss_mse_radial_weighted':
        loss, pixel_eff, loss_eff, text = arrimg_subtraction_loss_mse_radial_weighted(arrimg2, arrimg1_rot, proposed_shift, power_of, scope_penalty)
    else:
        raise ValueError (f'Please choose valid selection. {loss_cal_selection} was given.')
    return loss, pixel_eff, loss_eff, text 
        


def arrimg_subtraction_loss_mse_normal(arrimg2, arrimg1_rot, proposed_shift, power_of, scope_penalty):
    
    #power_of = 2   # 2 = MSE, 4 = MQE.
    
    #[1] Size should already been checked and crop before this point. This is just for redundancy.
    img1_height, img1_width = arrimg2.shape[:2]
    img2_height, img2_width = arrimg1_rot.shape[:2]
    if not (img1_width == img2_width and img1_height == img2_height):
        raise ValueError('Both images must be the same size.')
    
    #[2] This line is used to show what proposed_shift contains.
    ctr_rot_x, ctr_rot_y, rot_degree = proposed_shift
    
    #[3] Create a white image of the same size as a reference of out-of-image area.
    white_image = np.full((img1_height, img1_width, 3), 255, dtype=np.uint8)
    white_image_rot = rotate_image(white_image, proposed_shift)
    
    #[4] Create a mask for where the white_image_rot is not white (meaning area is out-of-bounds)
    mask = np.any(white_image_rot != 255, axis=2) #True if not equal to 255.
    
    #[5] Apply the mask to calculate loss only on valid (in-bound) pixels [Important] "**2" is normal.
    valid_diff = np.where(mask, 0, np.mean((arrimg2.astype(int) - arrimg1_rot.astype(int))**power_of, axis=2))
    #valid_diff = np.where(mask, 0, np.mean((arrimg2.astype(int) - arrimg1_rot.astype(int))**2, axis=2))
    loss = np.sum(valid_diff)
    
    #[6] Calculate effective loss
    pixel_skip = np.sum(mask)
    pixel_total = img1_width * img1_height
    pixel_eff = pixel_total - pixel_skip
    if pixel_eff == 0:
        text_error = f'pixel_eff = {pixel_eff}. Angle of rotation was too larger that it shift the image away without overlap.\n'\
                    f'  proposed_shift = {proposed_shift}\n\n'
        raise ValueError(text_error)
    loss_eff = loss/pixel_eff
    
    #[7] Calculate loss with scope panalty
    loss_eff += pixel_skip * scope_penalty

    #[8] Prepare text report
    effective_pct = round(100 * pixel_eff / pixel_total, 2)
    text = '[arrimg_subtraction]\n'\
            f'-- proposed_shift (cr_x, cr_y, rot)= {proposed_shift}\n'\
            f'-- pixels (total, skipped, effective): \n'\
            f'    {pixel_total}, {pixel_skip}, {pixel_eff} ({effective_pct}%)\n'\
            f'-- scope_penalty: {scope_penalty}\n'\
            f'-- loss = {loss:,.0f}\n'\
            f'-- loss_eff = {loss_eff:.2f}'
    #print(text)
    return loss, pixel_eff, loss_eff, text


    #return white_image_rot, mask, valid_diff, loss, pixel_eff, loss_eff, text
    

def arrimg_subtraction_loss_mse_radial_weighted(arrimg2, arrimg1_rot, proposed_shift, power_of, scope_penalty):
    """
    Calculate the MSE loss weighted by a radial factor that increases from the center of the image.
    At the center of the image, the weight of the valid_diff is 1, increasing linearly outward.
    """
    radial_weight_center = 0.1
    radial_weight_slope = 0.5  # Weight increase by 0.1 for every pixel away from the center of the image.
    #power_of = 2   # 2 = MSE, 4 = MQE.

    #[1] Ensure both images are the same size
    img1_height, img1_width = arrimg2.shape[:2]
    img2_height, img2_width = arrimg1_rot.shape[:2]
    if not (img1_width == img2_width and img1_height == img2_height):
        raise ValueError('Both images must be the same size.')

    #[2] Extract shift components
    ctr_rot_x, ctr_rot_y, rot_degree = proposed_shift

    #[3] Create a white image of the same size as a reference for the out-of-image area
    white_image = np.full((img1_height, img1_width, 3), 255, dtype=np.uint8)
    white_image_rot = rotate_image(white_image, proposed_shift)

    #[4] Create a mask for where the white_image_rot is not white (indicating out-of-bounds area)
    mask = np.any(white_image_rot != 255, axis=2)  # True if not equal to 255.
    #mask = np.any(white_image_rot == 0, axis=2)  # True if equal to 0.

    #[5] Calculate radial weights
    x = np.arange(img1_width) - img1_width // 2
    y = np.arange(img1_height) - img1_height // 2
    x_grid, y_grid = np.meshgrid(x, y, indexing='xy')
    radial_weights = radial_weight_center + radial_weight_slope * np.sqrt(x_grid**2 + y_grid**2)
    #print(f'radial_weights.shape (before newaxis) = {radial_weights.shape}')    # (744, 991)
    radial_weights = radial_weights[:,:,np.newaxis]
    #print(f'radial_weights.shape (after newaxis) = {radial_weights.shape}')     # (744, 991, 1)
    #print(f'arrimg2.shape = {arrimg2.shape}')           # (744, 991, 3)
    #print(f'mask.shape = {mask.shape}')                 # (744, 991)
    

    #[6] Calculate valid difference using the mask and radial weights
    diff = (arrimg2.astype(int) - arrimg1_rot.astype(int))**power_of
    weighted_diff = radial_weights * diff
    valid_diff = np.where(mask, 0, np.mean(weighted_diff, axis=2))

    loss = np.sum(valid_diff)

    #[7] Calculate effective loss
    pixel_skip = np.sum(mask)
    pixel_total = img1_width * img1_height
    pixel_eff = pixel_total - pixel_skip
    if pixel_eff == 0:
        raise ValueError(f'pixel_eff = {pixel_eff}. Angle of rotation was too larger that it shift the image away without overlap.')
    loss_eff = loss/pixel_eff
    loss_eff = loss / pixel_eff if pixel_eff else float('inf')  # Handle division by zero

    #[8] Calculate loss with scope penalty
    loss_eff += pixel_skip * scope_penalty

    #[9] Prepare text report
    effective_pct = round(100 * pixel_eff / pixel_total, 2)
    text = '[arrimg_subtraction]\n'\
           f'-- proposed_shift (cr_x, cr_y, rot)= {proposed_shift}\n'\
           f'-- pixels (total, skipped, effective): \n'\
           f'   {pixel_total}, {pixel_skip}, {pixel_eff} ({effective_pct}%)\n'\
           f'-- scope_penalty: {scope_penalty}\n'\
           f'-- loss = {loss:,.0f}\n'\
           f'-- loss_eff = {loss_eff:.2f}'
    #print(text)
    return loss, pixel_eff, loss_eff, text



#[3] Loss - End















#[4] Gradient - Start


def compute_gradients_with_x_y_rot_fwd_diff(loss_cal_selection, 
                                    arrimg1, arrimg2, 
                                    loss, loss_eff, 
                                    current_proposed_shift, 
                                    power_of, 
                                    scope_penalty, 
                                    eps, rot_eps_constraint):
    
    #[3] Unpack shift info.
    ctr_rot_x, ctr_rot_y, rot_degree = current_proposed_shift
    
    #[5] Gradient with respect to ctr_rot_x
    shift_x_plus = [ctr_rot_x + eps, ctr_rot_y, rot_degree]
    arrimg1_rot_x_plus = rotate_image(arrimg1, shift_x_plus)
    loss_x_plus, _, loss_eff_x_plus, _ = arrimg_subtraction_loss_selector(loss_cal_selection, 
                                                                          arrimg2, arrimg1_rot_x_plus, 
                                                                          shift_x_plus, power_of, scope_penalty)
    grad_x = (loss_eff_x_plus - loss_eff) / eps
    
    #[6] Gradient with respect to ctr_rot_y
    shift_y_plus = [ctr_rot_x, ctr_rot_y + eps, rot_degree]
    arrimg1_rot_y_plus = rotate_image(arrimg1, shift_y_plus)
    loss_y_plus, _, loss_eff_y_plus, _ = arrimg_subtraction_loss_selector(loss_cal_selection, 
                                                                          arrimg2, arrimg1_rot_y_plus
                                                                          , shift_y_plus, power_of, scope_penalty)
    grad_y = (loss_eff_y_plus - loss_eff) / eps

    #[2] Gradient with respect to rot_degree
    angle_eps = eps  * rot_eps_constraint
    r_plus = rot_degree + angle_eps
    shift_r_plus = [ctr_rot_x, ctr_rot_y, r_plus]
    arrimg1_r_plus = rotate_image(arrimg1, shift_r_plus)
    loss_r_plus, _, loss_eff_r_plus, _ = arrimg_subtraction_loss_selector(loss_cal_selection, 
                                                                          arrimg2, arrimg1_r_plus
                                                                          , shift_r_plus, power_of, scope_penalty)
    grad_r = (loss_eff_r_plus - loss_eff) / angle_eps
    
    #[3] Print text for debugging. (Don't change the spaces here, it will affect the tab.)
    text =  f'eps, angle_eps: \t{eps},      \t{angle_eps}\n'\
            f'--shift_x_plus: \t{shift_x_plus}\n'\
            f'--shift_y_plus: \t{shift_x_plus}\n'\
            f'--shift_r_plus: \t{shift_r_plus}\n'\
            f'----grad_x, grad_y, grad_r:    \t{grad_x:.10f}, \t{grad_y:.10f}, \t{grad_r:.10f}\n'
    #print(text)
    return (grad_x, grad_y, grad_r, text )




def compute_gradients_with_x_y_rot_ctr_diff(loss_cal_selection, 
                                   arrimg1, arrimg2, 
                                   loss, loss_eff, 
                                   current_proposed_shift, 
                                   power_of, 
                                   scope_penalty, 
                                   eps, rot_eps_constraint):
    
    #[3] Unpack shift info.
    ctr_rot_x, ctr_rot_y, rot_degree = current_proposed_shift
    
    #[5] Gradient with respect to ctr_rot_x
    shift_x_plus = [ctr_rot_x + eps/2, ctr_rot_y, rot_degree]
    shift_x_minus = [ctr_rot_x - eps/2, ctr_rot_y, rot_degree]
    arrimg1_rot_x_plus = rotate_image(arrimg1, shift_x_plus)
    arrimg1_rot_x_minus = rotate_image(arrimg1, shift_x_minus)
    loss_x_plus, _, loss_eff_x_plus, _ = arrimg_subtraction_loss_selector(loss_cal_selection, 
                                                                          arrimg2, arrimg1_rot_x_plus, 
                                                                          shift_x_plus, power_of, scope_penalty)
    loss_x_minus, _, loss_eff_x_minus, _ = arrimg_subtraction_loss_selector(loss_cal_selection, 
                                                                            arrimg2, arrimg1_rot_x_minus, 
                                                                            shift_x_minus, power_of, scope_penalty)
    grad_x = (loss_eff_x_plus - loss_eff_x_minus) / eps
    
    #[6] Gradient with respect to ctr_rot_y
    shift_y_plus = [ctr_rot_x, ctr_rot_y + eps/2, rot_degree]
    shift_y_minus = [ctr_rot_x, ctr_rot_y - eps/2, rot_degree]
    arrimg1_rot_y_plus = rotate_image(arrimg1, shift_y_plus)
    arrimg1_rot_y_minus = rotate_image(arrimg1, shift_y_minus)
    loss_y_plus, _, loss_eff_y_plus, _ = arrimg_subtraction_loss_selector(loss_cal_selection, 
                                                                          arrimg2, arrimg1_rot_y_plus,
                                                                          shift_y_plus, power_of, scope_penalty)
    loss_y_minus, _, loss_eff_y_minus, _ = arrimg_subtraction_loss_selector(loss_cal_selection, 
                                                                            arrimg2, arrimg1_rot_y_minus,
                                                                            shift_y_minus, power_of, scope_penalty)
    grad_y = (loss_eff_y_plus - loss_eff_y_minus) / eps

    #[2] Gradient with respect to rot_degree
    angle_eps = eps * rot_eps_constraint
    r_plus = rot_degree + angle_eps/2
    r_minus = rot_degree - angle_eps/2
    shift_r_plus = [ctr_rot_x, ctr_rot_y, r_plus]
    shift_r_minus = [ctr_rot_x, ctr_rot_y, r_minus]
    arrimg1_r_plus = rotate_image(arrimg1, shift_r_plus)
    arrimg1_r_minus = rotate_image(arrimg1, shift_r_minus)
    loss_r_plus, _, loss_eff_r_plus, _ = arrimg_subtraction_loss_selector(loss_cal_selection, 
                                                                          arrimg2, arrimg1_r_plus,
                                                                          shift_r_plus, power_of, scope_penalty)
    loss_r_minus, _, loss_eff_r_minus, _ = arrimg_subtraction_loss_selector(loss_cal_selection, 
                                                                            arrimg2, arrimg1_r_minus,
                                                                            shift_r_minus, power_of, scope_penalty)
    grad_r = (loss_eff_r_plus - loss_eff_r_minus) / angle_eps
    
    #[3] Print text for debugging. (Don't change the spaces here, it will affect the tab.)
    text =  f'eps, angle_eps: \t{eps},      \t{angle_eps}\n'\
            f'--shift_x_plus: \t{shift_x_plus}\n'\
            f'--shift_y_plus: \t{shift_x_plus}\n'\
            f'--shift_r_plus: \t{shift_r_plus}\n'\
            f'----grad_x, grad_y, grad_r:    \t{grad_x:.10f}, \t{grad_y:.10f}, \t{grad_r:.10f}\n'
    #print(text)
    return (grad_x, grad_y, grad_r, text)



#[4] Gradient - End














#[5] Optimizer - Start


class AdamOptimizer_normal:
    """
    lr: initial learning rate.
    beta1: Controls the exponential decay of the first moment (mean) of the gradients.
    beta2: Controls the exponential decay of the second moment (variance) of the gradients.
    epsilon: Affect the step size in small gradient situation. 
                Increase to avoid small gradient makes larger step.
    --
    step_size: a step size calculated from gradient and learning rate.
    --
    Case 1: Increase beta1 to build more momentum when the gradients show a strong directional trend 
            and are not noisy, but the optimizer oscillates back and forth. This helps to smooth 
            the updates and keep the optimizer on a steady path toward the minimum.
    Case 2: Increase beta2 to stabilize the updates in a noisy dataset where gradients vary widely 
            between iterations. This makes the optimizer more conservative, reducing its sensitivity 
            to outliers in the gradients, which can be due to label noise or irrelevant information.
    """
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0
        self.step_size_hist = []  # Call "list_ss_x = x_optimizer.step_size_hist" to get step size history.

    def update(self, grad):
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad**2)
        m_hat = self.m / (1 - self.beta1)
        v_hat = self.v / (1 - self.beta2)
        step_size = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        self.step_size_hist.append(step_size)  # Track the effective update
        return step_size



class AdamOptimizer_track_lr:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0  # Time step
        self.lr_history = []  # List to store learning rates over updates
        self.loss_history = []  # List to store loss values to monitor progress
        self.lr_increase_factor = 1.1  # Factor by which the learning rate is increased
        self.threshold = 0.01  # Threshold for the loss change rate below which we increase the learning rate

    def update(self, grad, loss=None):
        if self.m is None:
            self.m = np.zeros_like(grad)
        if self.v is None:
            self.v = np.zeros_like(grad)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad**2)

        #[2] Bias correction
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        #[3] Compute the adjusted learning rate
        lr_adjustment = self.lr / (np.sqrt(v_hat) + self.epsilon)
        adjusted_lr = lr_adjustment * m_hat  # Using m_hat (corrected first moment)

        self.lr_history.append(lr_adjustment)  # Track adjusted learning rate
        
        #[4] Optionally update loss history and adjust learning rate
        if loss is not None:
            self.loss_history.append(loss)
            self.adjust_learning_rate()

        return adjusted_lr

    def adjust_learning_rate(self):
        if len(self.loss_history) > 5:  # Ensure there are enough points to compare
            recent_losses = self.loss_history[-5:]  # Last 5 losses
            #[8] Check if there is insufficient decrease
            if recent_losses[-1] > recent_losses[0] - self.threshold:
                self.lr *= self.lr_increase_factor
                print(f"Learning rate increased to {self.lr}")

    def get_lr_history(self):
        return self.lr_history


#[5] Optimizer - End












#[6] Training Loop - Start


def train_x_y_rot_to_minimize_substraction_loss_eff(arrimg2, arrimg1, 
                                                    init_proposed_shift, 
                                                    dic_hyperp,
                                                    text_log, name_tag):
    
    """
    [Goal]
        arrimg2, arrimg1 are two images but shifted (translation + rotation) a little bit.
        Optimizes the parameters ctr_rot_x, ctr_rot_y, rot_degree to minimize the loss using linear regression.
        
    [Args]
        arrimg2: image after rotated (rotation with x, y, c)
        arrimg1: image before rotated.
        proposed_shift: ctr_rot_x, ctr_rot_y, rot_degree
        init_lr: the learning rate for gradient descent
        num_epochs: the number of epochs to train the model
        text_log: log file
    [Note]
        1. Define the origin as the upper left corner of the image as the raster coordinate do.
        2. Use raster coordinate system where x-axis increases to the right and the y-axis increases downward.
    """
    print('\n[train_to_fit_the_invert_cone]')
    
    #[1] Initialize parameters
    ctr_rot_x, ctr_rot_y, rot_degree = init_proposed_shift
    
    #[2] Hyperparameter settings
    """
    Value (x, y, r) oscillate means learning rate too large. Loss oscillate means eps too large.
    """
    
    #[3] Unpacking the dictionary into individual variables
    init_lr = dic_hyperp['init_lr']
    scope_penalty = dic_hyperp['scope_penalty']  
    eps = dic_hyperp['eps']
    num_epochs = dic_hyperp['num_epochs']
    min_epoch_ratio = dic_hyperp['min_epoch_ratio']
    loss_cal_selection = dic_hyperp['loss_cal_selection']
    power_of = dic_hyperp['power_of']
    grad_cal_selection = dic_hyperp['grad_cal_selection']
    rot_eps_constraint = dic_hyperp['rot_eps_constraint']
    rot_lr_constraint = dic_hyperp['rot_lr_constraint']
    rot_update_constraint = dic_hyperp['rot_update_constraint']
    AO_optimizer = dic_hyperp['AO_optimizer']
    AO_beta1 = dic_hyperp['AO_beta1']
    AO_beta2 = dic_hyperp['AO_beta2']
    AO_eps = dic_hyperp['AO_eps']
    
    min_epoch = num_epochs * min_epoch_ratio

    #[4] Initiate AdamOptimizer_track_lr
    if  AO_optimizer == 'AdamOptimizer_normal':
        x_optimizer = AdamOptimizer_normal(init_lr, AO_beta1, AO_beta2, AO_eps)
        y_optimizer = AdamOptimizer_normal(init_lr, AO_beta1, AO_beta2, AO_eps)
        r_optimizer = AdamOptimizer_normal(init_lr * rot_lr_constraint, AO_beta1, AO_beta2, AO_eps)
    elif AO_optimizer == 'AdamOptimizer_track_lr':
        x_optimizer = AdamOptimizer_track_lr(init_lr, AO_beta1, AO_beta2, AO_eps)
        y_optimizer = AdamOptimizer_track_lr(init_lr, AO_beta1, AO_beta2, AO_eps)
        r_optimizer = AdamOptimizer_track_lr(init_lr * rot_lr_constraint, AO_beta1, AO_beta2, AO_eps)
    
    #[5] Define numbers and objects.
    list_x, list_y, list_r = [ctr_rot_x], [ctr_rot_y], [rot_degree]
    list_loss, list_loss_eff, list_pixel_eff = [], [], []
    list_grad_x, list_grad_y, list_grad_r = [], [], []
    text_in_for_loop = ''
    
    #[6] Print initial valuse
    text = f'\n\n[Initial Value]\t(cx, cy, rot) = {ctr_rot_x:.10f}, {ctr_rot_y:.10f}, {rot_degree:.10f}\n'
    print(text)
    text_log += text

    #[8] Train the model for the specified number of epochs
    for epoch in range(num_epochs):
        #[10] Start appending from the second iteration to keep list in the same length during dataframe construction.
        if epoch >= 1:
            list_x.append(ctr_rot_x)
            list_y.append(ctr_rot_y)
            list_r.append(rot_degree)
        
        #[11] Rotate arrimg1 with current_proposed_shift.
        current_proposed_shift = [ctr_rot_x, ctr_rot_y, rot_degree]
        arrimg1_rot = rotate_image(arrimg1, current_proposed_shift)
        
        #[12] Calculate loss
        loss, pixel_eff, loss_eff, text = arrimg_subtraction_loss_selector(loss_cal_selection, 
                                                                          arrimg2, arrimg1_rot, 
                                                                          current_proposed_shift, 
                                                                          power_of, 
                                                                          scope_penalty)
        list_loss.append(loss)
        list_pixel_eff.append(pixel_eff)
        list_loss_eff.append(loss_eff)
        
        #[15] Backward pass (gradient calculation)
        if grad_cal_selection == 'grad_fwd_diff':
            (grad_x, grad_y, grad_r, text) = compute_gradients_with_x_y_rot_fwd_diff(loss_cal_selection, 
                                                                                    arrimg1, arrimg2, 
                                                                                    loss, loss_eff, 
                                                                                    current_proposed_shift,
                                                                                    power_of, 
                                                                                    scope_penalty, 
                                                                                    eps, rot_eps_constraint)
        elif grad_cal_selection == 'grad_ctr_diff':
            (grad_x, grad_y, grad_r, text) = compute_gradients_with_x_y_rot_ctr_diff(loss_cal_selection, 
                                                                                    arrimg1, arrimg2, 
                                                                                    loss, loss_eff, 
                                                                                    current_proposed_shift,
                                                                                    power_of, 
                                                                                    scope_penalty, 
                                                                                    eps, rot_eps_constraint)
        else:
            raise ValueError (f'Please choose valid selection. {grad_cal_selection} was given.')
        
        #[16] Record gradients
        text_in_for_loop += f'\nEpoch [{epoch+1}/{num_epochs}] eps: {eps} \n{text}'
        list_grad_x.append(grad_x)
        list_grad_y.append(grad_y)
        list_grad_r.append(grad_r)
        
        #[18] Update primary parameters using Adam optimizer
        ss_x = x_optimizer.update(grad_x)
        ss_y = y_optimizer.update(grad_y)
        ss_r = r_optimizer.update(grad_r)
        ctr_rot_x -= ss_x
        ctr_rot_y -= ss_y
        rot_degree -= ss_r * rot_update_constraint

        
        #[19] Update secondary parameters
        current_proposed_shift = ctr_rot_x, ctr_rot_y, rot_degree
        
        #[20] Find lowest list_loss_eff
        if (epoch >= min_epoch):
            lowest_loss_eff = np.min(list_loss_eff[5:]) # Except for first 5 elements.
            last_five_loss_eff = list_loss_eff[-5:]
            
            #[21] Check if the lowest loss efficiency is NOT in the last five record
            if (lowest_loss_eff not in last_five_loss_eff):
                text =  f'\nEpoch [{epoch+1}/{num_epochs}] \t({name_tag}) \tloss_eff: \t{loss_eff:.0f}.\n'\
                        f'--[Good enough] lowest_loss_eff ({lowest_loss_eff:2f}) is not in the last_five_loss_eff.\n'\
                        f'--  last_five_loss_eff: {last_five_loss_eff}'
                text_in_for_loop += text
                print(text)
                break
            
        #[26] Print progress
        text =  f'\nEpoch [{epoch+1}/{num_epochs}] \t({name_tag}) \tloss_eff: \t{loss_eff:.0f}.\n'\
                f'    (grad_x, grad_y, grad_r) =            \t{grad_x:.8f}, \t{grad_y:.8f}, \t{grad_r:.8f}\n'\
                f'    (ss_x, ss_y, ss_r) =                  \t{ss_x:.8f}, \t{ss_y:.8f}, \t{ss_r:.8f}\n'\
                f'    (ctr_rot_x, ctr_rot_y, rot_degree) =  \t{ctr_rot_x:.4f}, \t{ctr_rot_y:.4f}, \t{rot_degree:.4f}'
        text_in_for_loop += text
        print(text)
        
    #[30] Gather learning rate from Adam optimizer after training
    list_ss_x = x_optimizer.step_size_hist  # Access as attribute (with out "()"), not method (with "()").
    list_ss_y = y_optimizer.step_size_hist
    list_ss_r = r_optimizer.step_size_hist

    #[33] Dataframe length check.
    df_length_check = False
    if df_length_check:
        print(f'[list_ss_x] {len(list_ss_x)}')
        print(f'[list_ss_y] {len(list_ss_y)}')
        print(f'[list_ss_r] {len(list_ss_r)}')
        print(f'[list_grad_x] {len(list_grad_x)}')
        print(f'[list_grad_y] {len(list_grad_y)}')
        print(f'[list_grad_r] {len(list_grad_r)}')
        print(f'[list_x] {len(list_x)}')
        print(f'[list_y] {len(list_y)}')
        print(f'[list_r] {len(list_r)}')
        print(f'[list_loss] {len(list_loss)}')
        print(f'[list_loss_eff] {len(list_loss_eff)}')
        print(f'[list_pixel_eff] {len(list_pixel_eff)}')
    
    #[34] Create dataframe for the training process.
    df_train_recrod_xyrot = pd.DataFrame({
                        'list_ss_x': list_ss_x,
                        'list_ss_y': list_ss_y,
                        'list_ss_r': list_ss_r,
                        'list_grad_x': list_grad_x,
                        'list_grad_y': list_grad_y,
                        'list_grad_r': list_grad_r,
                        'list_x': list_x,
                        'list_y': list_y,
                        'list_r': list_r,
                        'list_loss': list_loss,
                        'list_loss_eff': list_loss_eff,
                        'list_pixel_eff': list_pixel_eff,
                    })

    df_train_recrod_xyrot.index.name = 'epoch'
    print()
    
    #[35] Return optimized parameters and final loss
    return (df_train_recrod_xyrot, text_log)


#[6] Training Loop - End
























#[7] Training dataframe - Start




def recover_lists_from_df_train_recrod_xyrot(df_train_recrod_xyrot):
    """
    This function reads the following dataframe into lists
    df_train_recrod_xyrot = pd.DataFrame({
                        'list_ss_x': list_ss_x,
                        'list_ss_y': list_ss_y,
                        'list_ss_r': list_ss_r,
                        'list_grad_x': list_grad_x,
                        'list_grad_y': list_grad_y,
                        'list_grad_r': list_grad_r,
                        'list_x': list_x,
                        'list_y': list_y,
                        'list_r': list_r,
                        'list_loss': list_loss,
                        'list_loss_eff': list_loss_eff,
                        'list_pixel_eff': list_pixel_eff,
                    })
    """
    #[1] Extracting each list from the dataframe columns
    list_ss_x = df_train_recrod_xyrot['list_ss_x'].tolist()
    list_ss_y = df_train_recrod_xyrot['list_ss_y'].tolist()
    list_ss_r = df_train_recrod_xyrot['list_ss_r'].tolist()
    list_grad_x = df_train_recrod_xyrot['list_grad_x'].tolist()
    list_grad_y = df_train_recrod_xyrot['list_grad_y'].tolist()
    list_grad_r = df_train_recrod_xyrot['list_grad_r'].tolist()
    list_x = df_train_recrod_xyrot['list_x'].tolist()
    list_y = df_train_recrod_xyrot['list_y'].tolist()
    list_r = df_train_recrod_xyrot['list_r'].tolist()
    list_loss = df_train_recrod_xyrot['list_loss'].tolist()
    list_loss_eff = df_train_recrod_xyrot['list_loss_eff'].tolist()
    list_pixel_eff = df_train_recrod_xyrot['list_pixel_eff'].tolist()
    
    #[2] Return the index of the row with lowest 
    idx_lowest_loss_eff = df_train_recrod_xyrot['list_loss_eff'].idxmin()
    
    return (list_ss_x, list_ss_y, list_ss_r,
            list_grad_x, list_grad_y, list_grad_r,
            list_x, list_y, list_r,
            list_loss, list_loss_eff, list_pixel_eff), idx_lowest_loss_eff



def return_opt_from_df_train_recrod_xyrot(df_train_recrod_xyrot):
    #[35] Recover df_train_recrod
    (list_ss_x, list_ss_y, list_ss_r,
        list_grad_x, list_grad_y, list_grad_r,
        list_x, list_y, list_r,
        list_loss, list_loss_eff, list_pixel_eff),\
    idx_lowest_loss_eff = recover_lists_from_df_train_recrod_xyrot(df_train_recrod_xyrot)
    #[37] Extract optimized values of 0th train
    i_opt = idx_lowest_loss_eff
    x_opt, y_opt, r_opt = list_x[i_opt], list_y[i_opt], list_r[i_opt]
    le_opt = list_loss_eff[i_opt]
    return (i_opt, x_opt, y_opt, r_opt, le_opt)
    


#[7] Training dataframe - End










#[8] Minimum search - Start


def cubic_window_sliced_mapping(arrimg2, 
                                arrimg1, 
                                list_x, list_y, list_z,
                                hp_min, hp_max,
                                dic_hyperp, 
                                dir_save, 
                                text_log):

    #[1] Ensure directory exists
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
    
    #[2] Set fontsize
    fs_title = 20
    fs_axis = 16
    
    #[3] Unpacking the dictionary into individual variables
    scope_penalty = dic_hyperp['scope_penalty']  
    loss_cal_selection = dic_hyperp['loss_cal_selection']
    power_of = dic_hyperp['power_of']
    
    #[5] Create meshgrid
    X_mesh, Y_mesh, Z_mesh = np.meshgrid(list_x, list_y, list_z, indexing='ij')
    
    #[7] Initialize list for loss_eff
    list_loss_eff = []
    
    #[8] Iterate through each layer of z
    for z_idx, z in enumerate(list_z):
        print(f'\n\n[Current z layer] {z:.4f}')
        #[10] Iterate through the XY mesh in a specific z layer
        for cnt, (x, y) in enumerate(zip(np.ravel(X_mesh[:,:,z_idx]), np.ravel(Y_mesh[:,:,z_idx]))):
            iter_shift = [x, y, z]
            arrimg1_rot = rotate_image(arrimg1, iter_shift)
            
            #[12] Calculate loss
            loss, pixel_eff, loss_eff, text = arrimg_subtraction_loss_selector(loss_cal_selection, 
                                                                              arrimg2, arrimg1_rot, 
                                                                              iter_shift, 
                                                                              power_of, 
                                                                              scope_penalty)
            list_loss_eff.append(loss_eff)
        
        #[13] Reshape list_loss_eff into 2D array for current z layer
        loss_eff_layer = np.array(list_loss_eff[-(len(list_x) * len(list_y)):]).reshape(len(list_x), len(list_y))
        
        #[14] Find the location with max and min loss_eff in that position
        idx_max_le = np.unravel_index(np.argmax(loss_eff_layer, axis=None), loss_eff_layer.shape)
        idx_min_le = np.unravel_index(np.argmin(loss_eff_layer, axis=None), loss_eff_layer.shape)
        text =  f'\n\n[Current z layer] {z:.4f}'\
                f'  max = ([{list_x[idx_max_le[0]]}, {list_y[idx_max_le[1]]}]), loss_eff = {loss_eff_layer[idx_max_le]}\n'\
                f'  min = ([{list_x[idx_min_le[0]]}, {list_y[idx_min_le[1]]}]), loss_eff = {loss_eff_layer[idx_min_le]}\n'
        text_log += text
        
        #[16] Plot and save the heat map of that layer of z
        plt.figure(figsize=(6, 6))
        plt.subplots_adjust(left=0.15, bottom=0.05, right=0.95, top=0.90)
        plt.imshow(loss_eff_layer, 
                   extent=[list_x[0], list_x[-1], list_y[0], list_y[-1]], 
                   origin='upper', 
                   aspect='equal', 
                   vmin=hp_min, 
                   vmax=hp_max)
        plt.colorbar(label='Loss')
        text_title = f'Heat Map of Loss for z={z:.4f}\n'\
                        f' (max, min) = {loss_eff_layer[idx_max_le]:.0f}, {loss_eff_layer[idx_min_le]:.0f}'
                        
        plt.title(text_title, fontsize=fs_title, pad=20)
        plt.xlabel('x', fontsize=fs_axis)
        plt.ylabel('y', fontsize=fs_axis)
        plt.gca().invert_yaxis()
        
        #[18] Save the heat map of that layer of z
        path_save = os.path.join(dir_save, f'01_Heatmap_z{z:.4f}.png')
        plt.savefig(path_save)
        plt.close()
    
    #[22] Reshape arr_loss_eff to 3D shape
    arr_loss_eff = np.array(list_loss_eff).reshape(len(list_x), len(list_y), len(list_z))
    min_loss_idx = np.unravel_index(np.argmin(arr_loss_eff, axis=None), arr_loss_eff.shape)
    min_loss_value = arr_loss_eff[min_loss_idx]
    
    #[25] Create DataFrame columns for x and y coordinates
    cnt_total_points = len(list_x) * len(list_y) * len(list_z)
    cnt_points_per_layer = len(list_x) * len(list_y)
    
    #[27] Create x_coor and y_coor
    x_coor, y_coor = [], []
    for cnt, (x, y) in enumerate(zip(np.ravel(X_mesh[:,:,0]), np.ravel(Y_mesh[:,:,0]))):
        x_coor.append(x)
        y_coor.append(y)

    
    #[30] Create the DataFrame
    df_loss_eff = pd.DataFrame({'x_coor': x_coor, 'y_coor': y_coor})
    
    for i, iter_z in enumerate(list_z):
        start_z = cnt_points_per_layer * i
        end_z = cnt_points_per_layer * (i+1)
        df_loss_eff[f'z={iter_z:.4f}'] = list_loss_eff[start_z:end_z]
    
    #[31] Save the DataFrame to an Excel file
    timestamp = datetime.now().strftime("%Y-%m%d-%H%M")
    path_xlsx = os.path.join(dir_save, f'02_Table of loss_eff_{timestamp}.xlsx')
    df_loss_eff.to_excel(path_xlsx, index=False)
    
    #[34] Return the index of the point with minimum loss
    text =  f'\n[sliced mapping final report]\n'\
            f'--  min_loss_idx, min_loss_value: {min_loss_idx}, {min_loss_value}\n'\
            f'--  x, y, z, loss_eff: {X_mesh[min_loss_idx]}, {Y_mesh[min_loss_idx]}, {Z_mesh[min_loss_idx]}, {arr_loss_eff[min_loss_idx]}\n'
    text_log += text
    
    return X_mesh, Y_mesh, Z_mesh, arr_loss_eff, min_loss_idx, text_log


#[8] Minimum search - End



