"""
[Module_2_Image.py]
Purpose: Invert cone model
Author: Meng-Chi Ed Chen
Date: 2024-03-04
Reference:
    1.
    2.

Status: Complete.
"""
import os, sys, cv2
import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import piexif
from PIL import Image
from scipy.signal import savgol_filter
from sklearn.linear_model import RANSACRegressor


import Module_1_Processing as M1PCS #type: ignore        PCS = Processing
import Module_3_Train as M3TRA #type: ignore        TRA = Train





"""
#[1] Image - Start
"""

def crop_to_same_size(path_img_t1, path_img_t2):
    print('\n[crop_to_same_size] sometime, size missed by a few pixel, and it makes ')
    global img1_cropped
    global img2_cropped
    img1 = cv2.imread(path_img_t1)
    img2 = cv2.imread(path_img_t2)
    img1_height, img1_width = img1.shape[:2]
    img2_height, img2_width = img2.shape[:2]
    if img1_width == img2_width and img1_height == img2_height:
        print('--Two images have the same size.', img1_width, ' x ', img1_height)
        path_img_t1_cropped = path_img_t1
        path_img_t2_cropped = path_img_t2
        
    else:
        print('--Two images have different size. Image_1:', img1_width, ' x ', img1_height, '; Image_2:', img2_width, 'x', img2_height)
        width_min =  min(img1_width, img2_width)
        height_min = min(img1_height, img2_height)
        img1_cropped = img1[0:width_min, 0:height_min]
        img2_cropped = img2[0:width_min, 0:height_min]
        img1_width, img1_height = img1_cropped.shape[:2]
        img2_width, img2_height = img2_cropped.shape[:2]
        print('--After crop. Image_1:', img1_width, ' x ', img1_height, '; Image_2:', img2_width, 'x', img2_height)
        path_img_t1_cropped = path_img_t1.replace('.png', '_1-crop.png')
        path_img_t2_cropped = path_img_t2.replace('.png', '_1-crop.png')
        cv2.imwrite(path_img_t1_cropped, img1_cropped) 
        cv2.imwrite(path_img_t2_cropped, img2_cropped) 
        
    return path_img_t1_cropped, path_img_t2_cropped



def blur_image(path_img, k, bsn='1-blurred'):
    print(f'\n[blur_image] {os.path.basename(path_img)}')
    #[1] Check image
    img_BGRarray = cv2.imread(path_img, 1)  # 1 for color image
    if img_BGRarray is None:
        print(f"Error: Image at {path_img} could not be loaded.")
        return None

    #[2] Apply Gaussian blur to the image
    blurred_img = cv2.GaussianBlur(img_BGRarray, (k, k), 0)  # k must be positive and odd

    #[3] Save blurred image
    img_ext = os.path.splitext(path_img)[1]  # Find file extension
    path_img_blurred = path_img.replace(img_ext, f'_{bsn}{img_ext}')
    cv2.imwrite(path_img_blurred, blurred_img)
    #print(f'--Blurred Image saved: {os.path.basename(path_img_blurred)}')
    
    return path_img_blurred


def enhance_image(path_img, clipLimit, tileGridSize, bsn='2-enh'):#F09_OMcam
    print(f'\n[enhance_image] {os.path.basename(path_img)}')
    #[1] Check image
    img_BGRarray = cv2.imread(path_img, 1)
    if img_BGRarray is None:
        print(f"Error: Image at {path_img} could not be loaded.")
    #[2] Enhance image
    lab = cv2.cvtColor(img_BGRarray, cv2.COLOR_BGR2LAB) # Converting to LAB color space
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(tileGridSize, tileGridSize)) # clipLimit = 2, tileGridSize = 8
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b)) # Merge the CLAHE enhanced L-channel with the a and b channel
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR) # Converting from LAB Color model to BGR color space
    
    #[3] Save enhanced image.
    img_ext = os.path.splitext(path_img)[1] # Find file extension so that it works with different image files.
    path_img_enhanced = path_img.replace(img_ext, f'_{bsn}{img_ext}')
    cv2.imwrite(path_img_enhanced, enhanced_img)
    #print('--Image enhanced:', os.path.basename(path_img_enhanced))
    return path_img_enhanced


def sharpen_image(path_img, sharp_k, ch, sigma_high, cl, sigma_low, bsn='3-sharpened'):
    print(f'\n[sharpen_image] {os.path.basename(path_img)}')
    img_BGRarray = cv2.imread(path_img, 1)
    if img_BGRarray is None:
        print(f"Error: Image at {path_img} could not be loaded.")
        return None

    kernel_high = ch * gauss_kernel(sharp_k, sigma_high) - cl * gauss_kernel(sharp_k, sigma_low)
    #print("Kernel high shape:", kernel_high.shape, "Min:", np.min(kernel_high), "Max:", np.max(kernel_high))
    text_k = f'sk{sharp_k}_ch{ch}, sh{sigma_high}_cl{cl}, sl{sigma_low}'
    path_kernel = os.path.join(os.path.dirname(path_img), f'kernel_{text_k}.png')
    visualize_matrix_save(kernel_high, bsn, path_kernel, dpi=300)
    #[3] Filter image
    sharpened_img = cv2.filter2D(img_BGRarray, -1, kernel_high)
    #print("Before normalization - Min:", np.min(sharpened_img), "Max:", np.max(sharpened_img))
    #[4] Normalize image
    sharpened_img = cv2.normalize(sharpened_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #print("After normalization - Min:", np.min(sharpened_img), "Max:", np.max(sharpened_img))
    #[5] Save image
    img_ext = os.path.splitext(path_img)[1]
    path_img_sharpened = path_img.replace(img_ext, f'_{bsn}{img_ext}')
    cv2.imwrite(path_img_sharpened, sharpened_img)
    
    return path_img_sharpened


def gauss_kernel(k_size, sigma):
    #[1] Check kernel size
    if k_size % 2 == 0 or k_size < 1:
        raise ValueError("Kernel size must be a positive odd integer.")
    ax = np.arange(-k_size // 2 + 1., k_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel



def visualize_matrix_save(matrix, text, path, dpi=300):
    #[1] Find shape and define figure size.
    nrows, ncols = matrix.shape
    fig_size = (max(nrows, 5) * 0.5, max(ncols, 5) * 0.5)
    plt.figure(figsize=fig_size, dpi=dpi)
    max_abs_val = np.max(np.abs(matrix))  # Find maximum of absolute value.
    plt.imshow(matrix, cmap='seismic', vmin=-max_abs_val, vmax=max_abs_val)
    #[3] Optionally add gridlines for very small matrices
    if nrows <= 5 and ncols <= 5:
        plt.grid(which='major', color='black', linestyle='-', linewidth=2)
    #[4] Matrix statistics
    stat_max, stat_min, stat_mean, stat_med = round(np.max(matrix),3), round(np.min(matrix),3), round(np.mean(matrix),3), round(np.median(matrix),3)
    stats_text = f'Shape: {nrows} × {ncols}\n'\
                    f'Max: {stat_max},  Min: {stat_min}\n'\
                    f'Mean: {stat_mean},  Median: {stat_med}'\
                    f'{text}'
    plt.text(0, -ncols//10, stats_text, fontsize=8, color='white', bbox=dict(facecolor='black', alpha=0.5))
    plt.axis('off')
    #[5] Check if directory exists and create if not
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close()




def add_random_noise(image, std=10):
    """
    Generate an arr_noise that has the same shape as the image.
    The values in arr_noise follow a Gaussian distribution centered at 0 with a standard deviation of "sigma".
    Add these two arrays together to create the noisy image.
    """   
    #[2] Generate noise array with the same shape as the image
    noise = np.random.normal(0, std, image.shape)
    
    #[3] Add the noise array to the original image
    noisy_image = image + noise
    
    #[4] Clip the values to ensure they are within the valid range [0, 255] and convert to integer
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    #[5] Return the noisy image
    return noisy_image

def batch_images_add_noise(list_path_img, std, overwrite=True):
    """
    Add salt-and-pepper noise to a batch of images and overwrite the original images.
    :param list_path_img: List of image file paths.
    :return: List of paths to the noisy images.
    """
    for iter_path in list_path_img:
        #[1] Read the image
        arrimg = cv2.imread(iter_path)
        if arrimg is not None:
            #[2] Print info
            bsn = os.path.basename(iter_path)
            print(f'--Adding noise for {bsn}.')
            #[3] Add noise
            noisy_img = add_random_noise(arrimg, std)
            #[4] Overwrite the original image or not?
            if not overwrite:
                iter_path = iter_path.replace('.jpg', '_n.jpg')
            cv2.imwrite(iter_path, noisy_img) 
        else:
            raise ValueError('Check if arrimg is none.')

    return list_path_img


def batch_image_apply_blur(list_path_img, k=9, overwrite=True):
    """
    Add salt-and-pepper noise to a batch of images and overwrite the original images.
    :param list_path_img: List of image file paths.
    :return: List of paths to the noisy images.
    """
    for iter_path in list_path_img:
        #[1] Read the image
        arrimg = cv2.imread(iter_path)
        if arrimg is not None:
            #[2] Print info
            bsn = os.path.basename(iter_path)
            print(f'--Apply blur for {bsn}.')
            #[3] Add noise
            blurred_img = cv2.GaussianBlur(arrimg, (k, k), 0) 
            #[4] Overwrite the original image or not?
            if not overwrite:
                iter_path = iter_path.replace('.jpg', '_n.jpg')
            cv2.imwrite(iter_path, blurred_img) 
        else:
            raise ValueError('Check if arrimg is none.')

    return list_path_img



def print_arrimg_stat(path_or_arr_img, kstd=3, name='Color Statistics', print_status=False):
    #[1] Check if input is a path or an array
    if isinstance(path_or_arr_img, str):
        arr_img = cv2.imread(path_or_arr_img)
    else:
        arr_img = path_or_arr_img

    #[2] Check dimension
    if arr_img.ndim != 3 or arr_img.shape[2] != 3:
        raise ValueError("Input array must be a 3-channel image array (BGR).")

    #[3] Calculate statistics for each channel
    b_channel = arr_img[:,:,0]
    g_channel = arr_img[:,:,1]
    r_channel = arr_img[:,:,2]

    b_max, b_mean, b_min, b_std = np.max(b_channel), np.mean(b_channel), np.min(b_channel), np.std(b_channel)
    g_max, g_mean, g_min, g_std = np.max(g_channel), np.mean(g_channel), np.min(g_channel), np.std(g_channel)
    r_max, r_mean, r_min, r_std = np.max(r_channel), np.mean(r_channel), np.min(r_channel), np.std(r_channel)
    b_pks, b_nks = b_mean + kstd*b_std, b_mean - kstd*b_std
    g_pks, g_nks = g_mean + kstd*g_std, g_mean - kstd*g_std
    r_pks, r_nks = r_mean + kstd*r_std, r_mean - kstd*r_std

    #[4] Create a DataFrame to organize the statistics
    data = {
        'Channel': ['B', 'G', 'R'],
        'Max': [b_max, g_max, r_max],
        'Mean': [b_mean, g_mean, r_mean],
        'Min': [b_min, g_min, r_min],
        'Std': [b_std, g_std, r_std],
        f'Mean + {kstd} Std': [b_pks, g_pks, r_pks],
        f'Mean - {kstd} Std': [b_nks, g_nks, r_nks]
    }

    df_stat = pd.DataFrame(data)
    df_stat.set_index('Channel', inplace=True)
    
    #[5] Print the DataFrame
    if print_status:
        print(f'\n[{name}]')
        print(df_stat.to_string(float_format='%.2f'))
    
    #[6] Round the stds we need.
    b_pks, b_nks = int(b_pks), int(b_nks)
    g_pks, g_nks = int(g_pks), int(g_nks)
    r_pks, r_nks = int(r_pks), int(r_nks)
    
    return df_stat, b_pks, b_nks, g_pks, g_nks, r_pks, r_nks


def generate_strongly_blurred_image_as_blank(path_img, r = 10, blur_times=3, bsn='1-strongBlur'):
    # Increase r to have larger k. If min(h, w) = 512 and r = 50, k = 11; r = 10, k = 103.
    #[1] Check input
    if not isinstance(blur_times, int) or blur_times < 1 or blur_times > 10:
        raise ValueError(f'blur_times must be an integer between 1 and 10. Received: {blur_times}')
    
    #[2] Get size
    arrimg = cv2.imread(path_img)
    ext = os.path.splitext(path_img)[1]
    h, w, c = arrimg.shape
    
    #[3] Blur image
    k = int(min(h, w)/r)*2 + 1
    print(f'-- Gaussian k = {k}')
    for i in range(blur_times):
        arrimg = cv2.GaussianBlur(arrimg, (k, k), 0)
    
    #[10] Save the corrected image.
    path_img_blank = os.path.join(os.path.dirname(path_img), f'{bsn}{ext}')
    cv2.imwrite(path_img_blank, arrimg)
    
    return path_img_blank

def vignetting_correction(path_img, bsn='4-vignetCorr'):
    print(f'\n[img_vignetting_correction] {os.path.basename(path_img)}')
    #[1] Set values.
    ratio_extend = 0.05
    """
    ratio_extend: Ratio of averging extention.
    If ratio_extend = 0.05, it means averaging pixels from 0.45 to 0.55 along width and leangth.
    """
    
    #[3] Read blank image.
    path_img_blank = generate_strongly_blurred_image_as_blank(path_img, r = 10, blur_times=10, bsn='2-strongBlur')
    arrimg_blank = cv2.imread(path_img_blank)
    ext = os.path.splitext(path_img_blank)[1]

    #[5] Calculate image dimension and average center area.
    h, w, c = arrimg_blank.shape
    h_center = round(h/2)
    h_upper = round(h_center + h*ratio_extend)
    h_lower = round(h_center - h*ratio_extend)
    
    w_center = round(w/2)
    w_upper = round(w_center + w*ratio_extend)
    w_lower = round(w_center - w*ratio_extend)
    
    arr1D_center_color_mean = (arrimg_blank[h_lower:h_upper,w_lower:w_upper].mean(axis=0).mean(axis=0)).astype(int)
    arr1D_center_color_mean = np.around(arr1D_center_color_mean)
    
    #[7] Use sampled color to tile a ndarray.
    arr3D_center_averaged = np.tile(arr1D_center_color_mean, (h, w, 1))

    #[8] Generate image modifier for color correction.
    arr_modifier = arrimg_blank - arr3D_center_averaged
    
    #[9] Generate corrected image.
    arrimg = cv2.imread(path_img)
    arrimg_corrected = arrimg  - arr_modifier
    
    #[10] Save the corrected image.
    path_img_corrected = os.path.join(os.path.dirname(path_img), f'{bsn}{ext}')
    cv2.imwrite(path_img_corrected, arrimg_corrected)
    
    return path_img_corrected


def adjust_gamma(path_img, gamma=1.0, bsn='6-gamma'):
    #[1] Read image
    arrimg = cv2.imread(path_img)
    ext = os.path.splitext(path_img)[1]
    
    #[2] Gamma correction
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    arrimg_gamma = cv2.LUT(arrimg, table)

    #[10] Save the corrected image.
    path_img_corrected = os.path.join(os.path.dirname(path_img), f'{bsn}{ext}')
    cv2.imwrite(path_img_corrected, arrimg_gamma)
    
    return path_img_corrected


def color_balance(path_img, adjust_percent=1, bsn='7-color-balance'):
    #[1] Read image
    arrimg = cv2.imread(path_img)
    ext = os.path.splitext(path_img)[1]
    
    #[2] Color balance
    for channel in range(arrimg.shape[2]):
        # Get the current channel
        c = arrimg[:, :, channel]
        
        # Find the low and high percentiles
        total = c.shape[0] * c.shape[1]
        lower_bound = np.percentile(c, adjust_percent)
        upper_bound = np.percentile(c, 100 - adjust_percent)
        
        # Clip and rescale the intensities
        c = np.clip(c, lower_bound, upper_bound)
        c = ((c - lower_bound) / (upper_bound - lower_bound) * 255).astype(np.uint8)
        arrimg[:, :, channel] = c

    #[10] Save the corrected image.
    path_img_corrected = os.path.join(os.path.dirname(path_img), f'{bsn}{ext}')
    cv2.imwrite(path_img_corrected, arrimg)
    
    return path_img_corrected



def normalize_single_channel(path_img, channel, lower_bound, upper_bound, bsn='7-C'):
    #[1] Read image
    arrimg = cv2.imread(path_img)
    ext = os.path.splitext(path_img)[1]
    
    #[2] Check if lower_bound and upper_bound are within the valid range
    if (not (0 <= lower_bound <= 255)) or (not (0 <= upper_bound <= 255)):
        print(f'lower_bound({lower_bound}) and upper_bound({upper_bound}) must be between 0 and 255.')
    
    #[3] Color balance
    channel_index = {'B': 0, 'G': 1, 'R': 2}
    if channel in channel_index:
        #[4] Get the channel data
        c = arrimg[:, :, channel_index[channel]]
        
        #[5] Normalize the channel data
        mask = (c >= lower_bound) & (c <= upper_bound)
        c_normalized = np.zeros_like(c)
        c_normalized[mask] = ((c[mask] - lower_bound) / (upper_bound - lower_bound) * 255).astype(np.uint8)
     
        #[6] Update the image channel
        arrimg[:, :, channel_index[channel]] = c_normalized
    else:
        raise ValueError(f'Invalid channel name: {channel}')

    #[7] Save the corrected image
    bsn = bsn.replace('C', channel)
    path_img_enh = os.path.join(os.path.dirname(path_img), f'{bsn}_{lower_bound}~{upper_bound}{ext}')
    cv2.imwrite(path_img_enh, arrimg)
    
    return path_img_enh




def enhance_single_channel(path_img, channel, clip_point, move_up=True, bsn='7-C'):
    #[1] Read image
    arrimg = cv2.imread(path_img)
    ext = os.path.splitext(path_img)[1]
    
    #[2] Check if clip_point is within the valid range
    if not (0 <= clip_point <= 255):
        raise ValueError('clip_point must be between 0 and 255')
    
    #[3] Color balance
    channel_index = {'B': 0, 'G': 1, 'R': 2}
    if channel in channel_index:
        #[4] Get the channel data
        c = arrimg[:, :, channel_index[channel]]
        
        #[5] Normalize the channel data
        c = np.clip(c, 0, 255)  # Ensure values are within 0-255
        if move_up:      # Normalize between clip_point-255
            c = ((c / 255) * (255 - clip_point) + clip_point).astype(np.uint8)
        else:           # Normalize between 0-clip_point
            c = ((c / 255) * clip_point).astype(np.uint8)
        
        #[6] Update the image channel
        arrimg[:, :, channel_index[channel]] = c
    else:
        raise ValueError(f'Invalid channel name: {channel}')

    #[7] Save the corrected image
    bsn = bsn.replace('C', channel)
    path_img_enh = os.path.join(os.path.dirname(path_img), f'{bsn}{clip_point}{move_up}{ext}')
    cv2.imwrite(path_img_enh, arrimg)
    
    return path_img_enh


def enhance_contrast_of_single_channel(path_img, channel, clip_point, push_range, bsn='7-C'):
    #[1] Read image
    arrimg = cv2.imread(path_img)
    ext = os.path.splitext(path_img)[1]
    
    #[2] Validate inputs
    if not (32 <= clip_point <= 224):
        raise ValueError('clip_point must be between 32 and 224')
    if push_range > 64:
        raise ValueError('push_range must be 64 or less')
    
    #[3] Color balance
    # Map the channel to the corresponding index
    channel_index = {'B': 0, 'G': 1, 'R': 2}
    if channel in channel_index:
        #[4] Get the channel data
        c = arrimg[:, :, channel_index[channel]]
    else:
        raise ValueError(f'Invalid channel name: {channel}')
    
    #[5] Define the new ranges
    lower_bound = clip_point - push_range // 2
    upper_bound = clip_point + push_range // 2
    
    #[6] Normalize values between 0 and clip_point to 0 and lower_bound
    mask1 = c <= clip_point
    c[mask1] = np.interp(c[mask1], [0, clip_point], [0, lower_bound])
    
    #[7] Normalize values between clip_point and 255 to upper_bound and 255
    mask2 = c > clip_point
    c[mask2] = np.interp(c[mask2], [clip_point, 255], [upper_bound, 255])
    
    #[8] Replace the channel data in the original image
    arrimg[:, :, channel_index[channel]] = c
    
    #[9] Modify the base name
    bsn = bsn.replace('C', channel)
    
    #[10] Save the corrected image
    path_img_enh = os.path.join(os.path.dirname(path_img), f'{bsn}{ext}')
    cv2.imwrite(path_img_enh, arrimg)
    
    #[11] Return the path to the enhanced image
    return path_img_enh



#[1] Image - End










"""
#[2] Match and Displacement - Start
"""
def find_good_matches_SIFT(img1, img2, LR = 0.7):  # scale invariant feature transform
    print('\n[find_good_matches]')
    SIFT = cv2.SIFT_create()       # Initialize SIFT detector
    
    #[1] Detect SIFT features and compute descriptors.
    key_point_info = []
    keypoints1, descriptors1 = SIFT.detectAndCompute(img1, None)
    keypoints2, descriptors2 = SIFT.detectAndCompute(img2, None)
    descriptors1 = descriptors1.astype(np.float32)
    descriptors2 = descriptors2.astype(np.float32)
    num_kpt_1 = len(keypoints1)
    num_kpt_2 = len(keypoints2)
    key_point_info.append(num_kpt_1)
    key_point_info.append(num_kpt_2)
    print(f'--Number of keypoints in (img1, img2): ({num_kpt_1}, {num_kpt_2})')

    matcher = 1

    # [2] Initialize and use FLANN matcher
    """
    [FLANN_INDEX_KDTREE]
    This is set to 1. It instructs the FLANN matcher to use the kd-tree indexing algorithm, 
    which is efficient for searching in low-dimensional feature spaces.
    
    [FLANN_INDEX_KMEANS]
    This is set to 2. It directs the FLANN matcher to use the hierarchical k-means tree algorithm, 
    which can be more suitable for higher-dimensional data.
    """
    if matcher == 1:
        # FLANN_INDEX_KDTREE
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    elif matcher == 2:
        # FLANN_INDEX_KMEANS
        FLANN_INDEX_KMEANS = 2
        index_params = dict(algorithm=FLANN_INDEX_KMEANS, trees=5)
        search_params = dict(checks=50)  # Adjust search parameters as needed
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    else:
        raise ValueError('Choose valid matcher.')
    
    #[3] Filter matches using the Lowe's ratio test
    list_good_matches = []
    for best_match, second_match in matches:
        if best_match.distance < LR * second_match.distance:
            list_good_matches.append(best_match)
            
    return list_good_matches, keypoints1, keypoints2



def cvt_to_opponent_color(arrimg):
    print('\n[cvt_to_opponent_color]')
    #[1] Ensure the image is in floating point before processing
    if arrimg.dtype != np.float32:
        arrimg = np.float32(arrimg)
    
    #[2] Calculate opponent channels
    b, g, r = cv2.split(arrimg)
    O1 = (r - g) / np.sqrt(2)
    O2 = (r + g - 2 * b) / np.sqrt(6)
    O3 = (r + g + b) / np.sqrt(3)
    
    #[3] Normalize and convert to uint8
    O1 = cv2.normalize(O1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    O2 = cv2.normalize(O2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    O3 = cv2.normalize(O3, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return cv2.merge([O1, O2, O3])


def find_good_matches_OpponentSIFT(img1, img2, LR=0.7):
    print('\n[find_good_matches_OpponentSIFT]')
    #[1] Convert images to opponent color space
    img1_opponent = cvt_to_opponent_color(img1)
    img2_opponent = cvt_to_opponent_color(img2)
    SIFT = cv2.SIFT_create()  # Initialize SIFT detector

    #[2] Detect SIFT features and compute descriptors in each color channel
    keypoints1, keypoints2 = [], []
    descriptors1, descriptors2 = np.array([]), np.array([])
    for i in range(3):  # Process each channel separately
        kp1, des1 = SIFT.detectAndCompute\
            (img1_opponent[:, :, i], None)
        kp2, des2 = SIFT.detectAndCompute\
            (img2_opponent[:, :, i], None)
        keypoints1.extend(kp1)
        keypoints2.extend(kp2)
        if des1 is not None and des2 is not None:
            if descriptors1.size == 0:
                descriptors1 = des1
                descriptors2 = des2
            else:
                descriptors1 = \
                np.vstack((descriptors1, des1))
                descriptors2 = \
                np.vstack((descriptors2, des2))    
    descriptors1 = descriptors1\
        .astype(np.float32)
    descriptors2 = descriptors2\
        .astype(np.float32)
    
    #[3] Printing the number of keypoints
    num_kpt_1, num_kpt_2 = len(keypoints1), len(keypoints2)
    print(f'--Number of keypoints in (img1, img2): ({num_kpt_1}, {num_kpt_2})')
    
    #[4] Initialize and use FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    #[5] Filter matches using the Lowe's ratio test
    list_good_matches = []
    for best_match, second_match in matches:
        if best_match.distance < LR * second_match.distance:
            list_good_matches.append(best_match)

    return list_good_matches, keypoints1, keypoints2


def convert_match_to_lists(list_good_matches, keypoints1, keypoints2):
    # [4] Calculate displacement distances
    list_coor2_x, list_coor2_y = [], []
    list_coor1_x, list_coor1_y = [], []
    list_d_x, list_d_y, list_d_l = [], [], []
    for cnt, iter_match in enumerate(list_good_matches):
        # Ensure match.trainIdx is within the range of keypoints1 and keypoints2
        if 0 <= iter_match.trainIdx < len(keypoints1) and 0 <= iter_match.queryIdx < len(keypoints2):
            coor2_x, coor2_y = keypoints2[iter_match.trainIdx].pt
            coor1_x, coor1_y = keypoints1[iter_match.queryIdx].pt
            d_x, d_y = (coor2_x - coor1_x) , (coor2_y - coor1_y)
            d_l = np.sqrt(d_x ** 2 + d_y ** 2)
            list_coor1_x.append(coor1_x)
            list_coor1_y.append(coor1_y)
            list_coor2_x.append(coor2_x)
            list_coor2_y.append(coor2_y)
            list_d_x.append(d_x)
            list_d_y.append(d_y)
            list_d_l.append(d_l)
            
    return (
        list_coor1_x, list_coor1_y
        , list_coor2_x, list_coor2_y
        , list_d_x, list_d_y, list_d_l)




def RANSAC_filtering_xyz(arr_x, arr_y, arr_z):
    #[1] Extract the first three columns into numpy arrays
    arr_x = np.array(arr_x)
    arr_y = np.array(arr_y)
    arr_z = np.array(arr_z)
    assert arr_x.shape[0] == arr_y.shape[0] == arr_z.shape[0], "Input arrays must have the same length"
    print(f'[RANSAC_filtering_xyz] {arr_x.shape[0]}')
    # [2] Prepare the features matrix 'X' and target vector 'z'
    X = np.column_stack((arr_x, arr_y))
    z = arr_z
    # [3] Fit the RANSAC regressor
    ransac = RANSACRegressor()
    ransac.fit(X, z)
    # [4] Identify inliers and outliers
    inlier_mask = ransac.inlier_mask_
    arr_x = arr_x[inlier_mask]
    arr_y = arr_y[inlier_mask]
    arr_z = arr_z[inlier_mask]
    return arr_x, arr_y, arr_z, inlier_mask



def refining_match_quality(list_coor1_x, list_coor1_y
                    , list_coor2_x, list_coor2_y
                    , list_d_x, list_d_y, list_d_l
                    , f1_dl_upper_limit, f2_dl_std_limit, dir, cnt):
    """
    [d_std_limit] is important. 2 is too lose, 1 is not tight enough. 0.5 is better. 
    """
    record_df_record =True
    df_stat=None
    #[1] Remove outliers
    if f1_dl_upper_limit is not None and f1_dl_upper_limit > 0 :
        
        #[2] Record dataframe df_record_f0
        if record_df_record:
            df_record_f0 = pd.DataFrame({'coor2_x': list_coor2_x, 
                                        'coor2_y': list_coor2_y,
                                        'd_l': list_d_l})
            M1PCS.save_df_as_excel_add(df_record_f0, dir, 'Filtering Outliser.xlsx', f'{cnt}_f0')
        
        #[First Filtering] (list_input, title=None, df_stat=None)
        #[5] Before first filtering, initiate empty lists and gather information. 
        _, f0_mean_d_l, _, _, _, _, df_stat = match_statistic_report_Memsrc(list_d_l, 'list_f0_d_l')
        list_f1_coor2_x, list_f1_coor2_y = [], []
        list_f1_coor1_x, list_f1_coor1_y = [], []
        list_f1_d_x, list_f1_d_y, list_f1_d_l = [], [], []
        
        #[6] First filtering
        f1_d_l_lower_lmt, f1_d_l_upper_lmt = 5, f1_dl_upper_limit
        print(f'\n--[First filtering]\n--Removing displacement outliers exceeding {f1_dl_upper_limit} pixel.')
        print(f'--Retaining displacements within the range of {f1_d_l_lower_lmt:.2f} to {f1_d_l_upper_lmt:.2f} pixels.')
        for i, iter_d_l in enumerate(list_d_l):
            if f1_d_l_lower_lmt <= iter_d_l and iter_d_l <= f1_d_l_upper_lmt:
                #[6] Collect lists
                list_f1_coor1_x.append(list_coor1_x[i])
                list_f1_coor1_y.append(list_coor1_y[i])
                list_f1_coor2_x.append(list_coor2_x[i])
                list_f1_coor2_y.append(list_coor2_y[i])
                list_f1_d_x.append(list_d_x[i])
                list_f1_d_y.append(list_d_y[i])
                list_f1_d_l.append(list_d_l[i])
        
        #[Second Filtering]
        #[10] Before second filtering, initiate empty lists and gather information.
        _, f1_mean_d_l, _, f1_std_d_l, _, _, df_stat \
            = match_statistic_report_Memsrc(list_f1_d_l, 'list_f1_d_l', df_stat) 
        list_f2_coor2_x, list_f2_coor2_y = [], []
        list_f2_coor1_x, list_f2_coor1_y = [], []
        list_f2_d_x, list_f2_d_y, list_f2_d_l = [], [], []

        #[12] second filtering
        d_upper_lmt, d_lower_lmt \
            = f1_mean_d_l + f2_dl_std_limit * f1_std_d_l\
                , f1_mean_d_l - f2_dl_std_limit * f1_std_d_l
        print(f'\n--[Second filtering]\n--Removing displacement outside the range of {f2_dl_std_limit} standard deviations.')
        for i, iter_d_l in enumerate(list_f1_d_l):
            if d_lower_lmt <= iter_d_l and iter_d_l <= d_upper_lmt:
                #[15] Collect lists
                list_f2_coor1_x.append(list_f1_coor1_x[i])
                list_f2_coor1_y.append(list_f1_coor1_y[i])
                list_f2_coor2_x.append(list_f1_coor2_x[i])
                list_f2_coor2_y.append(list_f1_coor2_y[i])
                list_f2_d_x.append(list_f1_d_x[i])
                list_f2_d_y.append(list_f1_d_y[i])
                list_f2_d_l.append(list_f1_d_l[i])
                
        #[2] Record dataframe df_record_f2
        if record_df_record:
            df_record_f2 = pd.DataFrame({'coor2_x': list_f2_coor2_x, 
                                        'coor2_y': list_f2_coor2_y,
                                        'd_l': list_f2_d_l})
            M1PCS.save_df_as_excel_add(df_record_f2, dir, 'Filtering Outliser.xlsx', f'{cnt}_f2')
        
        #[13] Match statistics after second filtering.
        _, _, _, _, _, _, df_stat = match_statistic_report_Memsrc(list_f2_d_l, 'list_f2_d_l', df_stat) 

        #[Third Filtering]
        third_filtering_with_RANSAC = True
        if third_filtering_with_RANSAC:
            #[15] Third filtering RANSAN
            list_f3_coor2_x, list_f3_coor2_y, list_f3_d_l, inlier_mask = RANSAC_filtering_xyz(list_f2_coor2_x, list_f2_coor2_y, list_f2_d_l)
            list_f3_coor1_x = np.array(list_f2_coor1_x)[inlier_mask]
            list_f3_coor1_y = np.array(list_f2_coor1_y)[inlier_mask]
            list_f3_d_x = np.array(list_f2_d_x)[inlier_mask]
            list_f3_d_y = np.array(list_f2_d_y)[inlier_mask]
            
            #[16] Convert everything to list.
            list_f9_coor1_x, list_f9_coor1_y, list_f9_coor2_x, list_f9_coor2_y, list_f9_d_x, list_f9_d_y, list_f9_d_l = (
                list_f3_coor1_x.tolist(), list_f3_coor1_y.tolist(), list_f3_coor2_x.tolist(), list_f3_coor2_y.tolist(),
                list_f3_d_x.tolist(), list_f3_d_y.tolist(), list_f3_d_l.tolist())
            
            #[17] Match statistics after third filtering.
            _, _, _, _, _, _, df_stat = match_statistic_report_Memsrc(list_f3_d_l, 'list_f3_d_l', df_stat)
        else:
            #[18] If not doing third filtering, put f2 to last f9 for return.
            list_f9_coor1_x, list_f9_coor1_y, list_f9_coor2_x, list_f9_coor2_y, list_f9_d_x, list_f9_d_y, list_f9_d_l = (
            list_f2_coor1_x, list_f2_coor1_y, list_f2_coor2_x, list_f2_coor2_y, list_f2_d_x, list_f2_d_y, list_f2_d_l)
            
        #[19] Check if we accidentlly swipe away all data.
        if len(list_f9_d_l) < 3:
            text = '\n--Few or non data points satisfy the condition of such STD. System exit.\n'\
                    f'--Match count: {len(list_f9_d_l)}\n'
            print(text)
            sys.exit()   
            
        #[2] Record dataframe df_record_f3
        if record_df_record:
            df_record_f3 = pd.DataFrame({'coor2_x': list_f9_coor2_x, 
                                        'coor2_y': list_f9_coor2_y,
                                        'd_l': list_f9_d_l})
            M1PCS.save_df_as_excel_add(df_record_f3, dir, 'Filtering Outliser.xlsx', f'{cnt}_f3')
            
        #[20] Update the original lists with the filtered ones
        before_filtering, after_filtering = len(list_d_l), len(list_f2_d_l)
        
        print(f'--Outliers (before, after, removed) = ({before_filtering}, {after_filtering}, {before_filtering - after_filtering})')

    return (list_f9_coor1_x, list_f9_coor1_y
            , list_f9_coor2_x, list_f9_coor2_y
            , list_f9_d_x, list_f9_d_y, list_f9_d_l
            , df_stat)




def match_statistic_report_Memsrc(list_input, title=None, df_stat=None, print=None):
    """
    Passes and append statistics to a dataframe.
    """
    #[1] Calculate statistics
    max_val, mean_val, min_val = np.max(list_input), np.mean(list_input), np.min(list_input)
    std_val, range_val, count_val = np.std(list_input), max_val - min_val, len(list_input)
    #[2] Prepare the statistics dictionary
    stats_dict = {
        'title': title,
        'max': max_val,
        'mean': mean_val,
        'min': min_val,
        'std': std_val,
        'range': range_val,
        'count': count_val}
    
    #[3] Check if df_stat is provided, if not create a new DataFrame
    if df_stat is None:
        df_stat = pd.DataFrame([stats_dict])
    else:
        df_stat = pd.concat([df_stat, pd.DataFrame([stats_dict])], ignore_index=True)
    #[4] Print if title is given.
    if print:
        print(f'  --[Stat.] {title}')
        print(f'  -- (max, mean, min) = ({round(max_val, 3)}, {round(mean_val, 3)}, {round(min_val, 3)})')
        print(f'  -- (std, range, count) = ({round(std_val, 3)}, {round(range_val, 3)}, {round(count_val, 3)})')
    
    return max_val, mean_val, min_val, std_val, range_val, count_val, df_stat



def mark_good_matches_on_img(list_d_l, list_coor2_x, list_coor2_y, large_font, path_image, bsn = '4-marked'):
    print('\n[mark_good_matches_on_img]')
    print(f'[list_d_l] {list_d_l[:5]}')
    print(f'[list_coor2_x] {list_coor2_x[:5]}')
    print(f'[list_coor2_y] {list_coor2_y[:5]}')
    
    #[1] Load the image from the given path
    img2 = cv2.imread(path_image)
    if img2 is None:
        raise FileNotFoundError(f"No image found at {path_image}")
    
    #[10] Set font size.
    if large_font:
        mmfs, mmft = 1.8, 5     # fs = font size (float); ft = font thickness (int).
        esfs, esft = 1.2, 4     # mm = Max and min; es = else
        s_cir = 15
    else:
        mmfs, mmft = 1, 3
        esfs, esft = 0.3, 1
        s_cir = 7

    #[2] Calculate max, min, and range.
    max_displs = np.round(np.max(list_d_l), 4)
    min_displs = np.round(np.min(list_d_l), 4)
    rangeMm = np.round(max_displs - min_displs, 4)   # Range = Max - min
    print(f'(max_displs, min_displs, rangeMm) = ({max_displs}, {min_displs}, {rangeMm})')
    if rangeMm == 0:
        rangeMm = 1  # Prevent division by zero if all displacements are the same

    #[3] Iterate over all good matches and their corresponding displacements
    list_location = []
    for i, displacement in enumerate(list_d_l):
        if i not in [np.argmin(list_d_l), np.argmax(list_d_l)]:
            #[5] Create a tuple of (x, y) coordinates from separate lists
            location = (int(list_coor2_x[i]), int(list_coor2_y[i]))
            
            #[6] Draw a circle at the keypoint location
            scale = (displacement - min_displs) / rangeMm
            #scale_10 = max(int(10 * scale) + 1, 1)      # scale_10 ranges from 1-11. This line will go wrong if there is only one match and rangeMm ==0.
            scale_10 = s_cir
            scale_288 = int(288 * scale)                # Deliberatly exceed 255 to have better visual effect.
            cv2.circle(img2, location, scale_10, (255, 271 - scale_288, scale_288 - 16), -1) # From (255, 271, -16) to (255, -16, 271).
            
            #[8] Decide on the displacement text format based on min_displs
            text_shift = 20
            text_format = "%.3f" if displacement < 10 else "%.2f" if displacement < 100 else "%.1f"
            text = text_format % displacement
            text_location = (location[0] + text_shift, location[1] + text_shift)
            list_location.append(location)
            
            #[10] Make extrema later, so that they are on top of other points.
            if i != np.argmin(list_d_l) or i != np.argmax(list_d_l):
                cv2.putText(img2, text, text_location, cv2.FONT_HERSHEY_SIMPLEX, esfs, (40, 40, 40), esft) 

    #[10] Draw extrema points
    for i in [np.argmin(list_d_l), np.argmax(list_d_l)]:
        displacement = list_d_l[i]
        #[11] Create a tuple of (x, y) coordinates from separate lists
        location = (int(list_coor2_x[i]), int(list_coor2_y[i]))
        
        #[12] Draw a circle at the keypoint location
        scale = (displacement - min_displs) / rangeMm
        #scale_10 = max(int(10 * scale) + 1, 1)      # scale_10 ranges from 1-11. This line will go wrong if there is only one match and rangeMm ==0.
        scale_10 = s_cir
        scale_288 = int(288 * scale)                # Deliberatly exceed 255 to have better visual effect.
        cv2.circle(img2, location, scale_10, (255, 271 - scale_288, scale_288 - 16), -1) # From (255, 271, -16) to (255, -16, 271).
        
        #[13] Decide on the displacement text format based on min_displs
        text_shift = 20
        text_format = "%.3f" if displacement < 10 else "%.2f" if displacement < 100 else "%.1f"
        text = text_format % displacement
        text_location = (location[0] + text_shift, location[1] + text_shift)
        list_location.append(location)
        
        #[14] Put text for exmtrema.
        if i == np.argmin(list_d_l) or i == np.argmax(list_d_l):
            cv2.putText(img2, text, text_location, cv2.FONT_HERSHEY_SIMPLEX, mmfs, (240, 240, 240), mmft)
    
    
    #[12] Save the result with a modified file name to indicate marking
    img_ext = os.path.splitext(path_image)[1]
    path_marked_image = path_image.replace(img_ext, f'_{bsn}{img_ext}')
    cv2.imwrite(path_marked_image, img2)

    return (list_d_l, list_coor2_x, list_coor2_y, path_marked_image)





def generate_match_statistics_and_df_one_pair(list_coor1_x, list_coor1_y\
                                            , list_coor2_x, list_coor2_y\
                                            , list_d_x, list_d_y, list_d_l):
    print('\n[match_statistics]')
    #[1] Calculate statistics and round to 3 decimal places
    """stat_max = round(np.max(list_d_l), 2)
    stat_min = round(np.min(list_d_l), 2)
    stat_mean = round(np.mean(list_d_l), 2)
    stat_med = round(np.median(list_d_l), 2)
    stat_std = round(np.std(list_d_l), 2)
    stats_text = f'--Max: {stat_max}, Mean: {stat_mean}, Min: {stat_min}\n--Median: {stat_med}, STD: {stat_std}'
    print('--[Match Statistics]\n', stats_text)"""
    
    #[2] Create a DataFrame from the provided lists
    df_one_pair = pd.DataFrame({
        'coor1_x': list_coor1_x,        # list of x coordinate on the first image.
        'coor1_y': list_coor1_y,        # list of y coordinate on the first image.
        'coor2_x': list_coor2_x,        # list of x coordinate on the second image.
        'coor2_y': list_coor2_y,        # list of y coordinate on the second image.
        'd_x': list_d_x,        # list of displacement in x direction.
        'd_y': list_d_y,        # list of displacement in y direction.
        'd_l': list_d_l})       # list of displacement length.
    
    #[5] Print the DataFrame using tabulate for a nice table format.
    #df_first_10 = df_one_pair.head(10)
    #print(f'--[df_first_10]\n{tabulate(df_first_10, headers="keys", tablefmt="orgtbl")}\n\n')
    return df_one_pair

#[3] Match and Displacement - End












"""
#[4] Fit the invert cone - Start
"""


def perform_linear_regression_traditional(list_x, list_y):
    list_x = np.array(list_x)
    list_y = np.array(list_y)
    #[1] Adding a column of ones to list_x to account for the intercept (b)
    A = np.vstack([list_x, np.ones(len(list_x))]).T
    #[2] Perform linear regression
    slope, y_intc = np.linalg.lstsq(A, list_y, rcond=None)[0]
    x_intc = -y_intc/slope
    return slope, y_intc, x_intc  # Slope (slope), y-intercept (y_intc), x-intercept (x_intc)


def perform_linear_regression_RANSAC(list_x, list_y):
    list_x = np.array(list_x)
    list_y = np.array(list_y)
    #[1] Reshape list_x to be a 2D array for sklearn compatibility
    list_x_reshaped = list_x.reshape(-1, 1)
    #[2] Create the RANSAC regressor and fit the model
    ransac = RANSACRegressor()
    ransac.fit(list_x_reshaped, list_y)
    #[3] Retrieve the coefficient (slope) and the intercept from the RANSAC model
    slope = ransac.estimator_.coef_[0]
    y_intc = ransac.estimator_.intercept_
    x_intc = -y_intc / slope if slope != 0 else np.inf  # Check for division by zero
    return slope, y_intc, x_intc  # Slope, y-intercept, x-intercept


def find_gradient_and_distance_to_z0_from_image_center(list_x, list_y, list_z):
    print('\n[find_gradient_and_distance_to_z0_from_image_center]')
    #[1] compute mean gradients (using zero if there are no valid gradients)
    mean_grad_x, _, _ = perform_linear_regression_traditional(list_x, list_z)
    mean_grad_y, _, _ = perform_linear_regression_traditional(list_y, list_z)
    print(f'[mean_grad_x] {mean_grad_x}')
    print(f'[mean_grad_y] {mean_grad_y}')
    gradient_length = np.linalg.norm([mean_grad_x, mean_grad_y])

    #[7] Compute the distance to the ground from each point
    avg_z = np.mean(list_z)
    dist_to_gnd = avg_z / gradient_length
    text = f'\n--[gradient] (x, y) = {mean_grad_x}, {mean_grad_y}\n'\
            f'--[gradient_length] {gradient_length}\n'\
            f'--[avg_z] {avg_z}\n'\
            f'--[dist_to_gnd] {dist_to_gnd}\n\n'
    print(text)
    return (mean_grad_x, mean_grad_y, dist_to_gnd, text)



#[4] Fit the invert cone - End














"""
#[5] Generate Rotate Example - Start
"""

def check_image_ang_if_total_rotation_distance_too_long(path_img, x_ctr_rot, y_ctr_rot, rot_degree, steps):
    #[1] Check image and get dimension. Able to handle both grayscale or color images.
    arrimg = cv2.imread(path_img)
    if arrimg is not None:
        dimensions = arrimg.shape
        if len(dimensions) == 2:
            h, w = dimensions
        elif len(dimensions) == 3:
            h, w, _ = dimensions
    else:
        raise ValueError("Image not loaded.")
    
    #[2] Assume the image is a circle by using the smallest side as diameter.
    D = min(h, w)
    L = np.linalg.norm(np.array([x_ctr_rot, y_ctr_rot]))
    rot_dist = 2 * L * np.tan(np.radians(rot_degree*steps/2))
    if rot_dist >= D * 0.8:
        text =  f'\n  [Rotated distance too long]\n'\
                f'    Current rotated distance is {rot_dist:.2f} and the shorter length of image is {D:.2f}\n'
        raise ValueError(text)
    
    return h, w
    


def update_image_exif_datetime_yesterday(list_path_img):
    # [15] Write every file's DateTimeOriginal as yesterday's current time with +1 min interval according to the rotation sequence.
    current_datetime = datetime.now() - timedelta(days=1)  # Yesterday's date and time
    for idx, iter_path in enumerate(list_path_img):
        # [16] Check if the current file is an image
        if os.path.isfile(iter_path) and iter_path.lower().endswith(('.jpg', '.jpeg', '.tiff')):
            # [17] Increment the time by 1 minute for each file
            new_datetime = current_datetime + timedelta(minutes=idx)
            exif_date = new_datetime.strftime("%Y:%m:%d %H:%M:%S")

            # [18] Check if image has EXIF data, if not, create an empty dict
            img = Image.open(iter_path)
            exif_dict = piexif.load(img.info['exif']) if 'exif' in img.info else {"Exif": {}}
            # [19] Update the EXIF data with the new timestamp
            exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = exif_date
            exif_bytes = piexif.dump(exif_dict)

            # [20] Save the image with updated EXIF data
            img.save(iter_path, "jpeg", exif=exif_bytes)
            img.close()
            bsn = os.path.basename(iter_path)
            print(f'--{bsn}, {exif_date}')

    return list_path_img




def find_largest_inscribe_rectangle_2(arrimg, square=True):
    #[1] Check image and get dimension. Able to handle both grayscale or color images.
    if arrimg is not None:
        dimensions = arrimg.shape
        if len(dimensions) == 2:
            h, w = dimensions
            white_pixels = np.where(arrimg == 255)
        elif len(dimensions) == 3:
            h, w, _ = dimensions
            white_pixels = np.where(np.all(arrimg == [255, 255, 255], axis=-1))
    else:
        raise ValueError("Image not loaded.")
    
    # [2] Calculate the average coordinates of white pixels if they exist
    if white_pixels[0].size > 0 and white_pixels[1].size > 0:
        average_y = int(np.mean(white_pixels[0]))
        average_x = int(np.mean(white_pixels[1]))
        hc,  wc = average_y, average_x
    else:
        #[3] Or use center as starting point.
        hc,  wc = h//2, w//2
    
    
    #[4] Set initial
    min_x, max_x = wc-1, wc+1
    min_y, max_y = hc-1, hc+1
    limit_l = 1000
    limit_s = 500
    
    #[5] Crop
    arrimg_crop = arrimg[min_y:max_y, min_x:max_x]
    if not np.all(arrimg_crop == [255, 255, 255]):
        raise Exception('Image might be too small.')
    
    for i_all4 in range(limit_l):
        min_x -= 1
        max_x += 1
        min_y -= 1
        max_y += 1
        arrimg_crop = arrimg[min_y:max_y, min_x:max_x]
        if not np.all(arrimg_crop == [255, 255, 255]):
            min_x += 1
            max_x -= 1
            min_y += 1
            max_y -= 1
            print(f'--[{i_all4}] Expanding on all 4 sides stops. \t\t\tLTRB({min_x}, {min_y}, {max_x}, {max_y})')
            
            #[7] Break the loop if user want square only.
            if square:
                
                #[7-1] Expand to the left and top.
                for i_1 in range(limit_s):
                    min_x -= 1
                    min_y -= 1
                    arrimg_crop = arrimg[min_y:max_y, min_x:max_x]
                    if min_x <= 0 or min_y <= 0 or (not np.all(arrimg_crop == [255, 255, 255])):
                        min_x += 1
                        min_y += 1
                        print(f'----[{i_1}] Expand to the left and top stops. \tLTRB({min_x}, {min_y}, {max_x}, {max_y})')
                        break
                        
                #[7-2] Expand to the top and right.
                for i_1 in range(limit_s):
                    min_y -= 1
                    max_x += 1
                    arrimg_crop = arrimg[min_y:max_y, min_x:max_x]
                    if min_y <= 0 or max_x >= w or (not np.all(arrimg_crop == [255, 255, 255])):
                        min_y += 1
                        max_x -= 1
                        print(f'----[{i_1}] Expand to the top and right stops. \tLTRB({min_x}, {min_y}, {max_x}, {max_y})')
                        break
                        
                #[7-3] Expand to the right and bottom.
                for i_1 in range(limit_s):
                    max_y += 1
                    max_x += 1
                    arrimg_crop = arrimg[min_y:max_y, min_x:max_x]
                    if max_y >= h or max_x >= w or (not np.all(arrimg_crop == [255, 255, 255])):
                        max_y -= 1
                        max_x -= 1
                        print(f'----[{i_1}] Expand to the right and bottom stops. \tLTRB({min_x}, {min_y}, {max_x}, {max_y})')
                        break
                        
                #[7-4] Expand to the bottom and left.
                for i_1 in range(limit_s):
                    min_x -= 1
                    max_y += 1
                    arrimg_crop = arrimg[min_y:max_y, min_x:max_x]
                    if min_x <= 0 or max_y >= h or (not np.all(arrimg_crop == [255, 255, 255])):
                        min_x += 1
                        max_y -= 1
                        print(f'----[{i_1}] Expand to the bottom and left stops. \tLTRB({min_x}, {min_y}, {max_x}, {max_y})')
                        break       
                break
            else:
                #[8-1] Expand to the left by decreasing min_x.
                for i_1 in range(limit_s):
                    min_x -= 1
                    #print(f'----({i_1} Decreasing min_x to {min_x}.')
                    arrimg_crop = arrimg[min_y:max_y, min_x:max_x]
                    if min_x <= 0 or (not np.all(arrimg_crop == [255, 255, 255])):
                        min_x += 1
                        print(f'----[{i_1}] Expand to the left by decreasing min_x stops. \tLTRB({min_x}, {min_y}, {max_x}, {max_y})')
                        break
                        
                #[8-2] Expand to the right by increasing max_x.
                for i_1 in range(limit_s):
                    max_x += 1
                    #print(f'----({i_1} Increasing max_x to {max_x}.')
                    arrimg_crop = arrimg[min_y:max_y, min_x:max_x]
                    if max_x >= w or (not np.all(arrimg_crop == [255, 255, 255])):
                        max_x -= 1
                        print(f'----[{i_1}] Expand to the right by increasing max_x stops. \tLTRB({min_x}, {min_y}, {max_x}, {max_y})')
                        break
                        
                #[8-3] Expand to the top by decreasing min_y.
                for i_1 in range(limit_s):
                    min_y -= 1
                    #print(f'----({i_1} Decreasing min_y to {min_y}.')
                    arrimg_crop = arrimg[min_y:max_y, min_x:max_x]
                    if min_y <= 0 or (not np.all(arrimg_crop == [255, 255, 255])):
                        min_y += 1
                        print(f'----[{i_1}] Expand to the top by decreasing min_y stops.    \tLTRB({min_x}, {min_y}, {max_x}, {max_y})')
                        break
                        
                #[8-4] Expand to the bottom by increasing max_y.
                for i_1 in range(limit_s):
                    max_y += 1
                    #print(f'----({i_1} Increasing max_y to {max_y}.')
                    arrimg_crop = arrimg[min_y:max_y, min_x:max_x]
                    if max_y >= h or (not np.all(arrimg_crop == [255, 255, 255])):
                        max_y -= 1
                        print(f'----[{i_1}] Expand to the bottom by increasing max_y stops. \tLTRB({min_x}, {min_y}, {max_x}, {max_y})')
                        break
                break
            
    #[18] Find side
    box = min_x, min_y, max_x, max_y
    return min_x, min_y, max_x, max_y




def generate_rotated_example_images_3(path_img, x_ctr_rot, y_ctr_rot, rot_degree, steps):
    """
    If steps = 3, 
    we should have step-00, step-01, step-02, and step-03 image.
    This function use original image as last image (step-03 image in this case).
    It then rotates the original image "backward" by 1 step to have step-02 image, 
    and rotates step-02 by 1 step to have step-01, 
    and rotates step-01 by 1 step to have step-00.
    Then, crops every image together.
    """
    #[2] Read the base image.
    h, w = check_image_ang_if_total_rotation_distance_too_long(path_img, x_ctr_rot, y_ctr_rot, rot_degree, steps)
    arrimg = cv2.imread(path_img)

    #[4] Initialize lists and dictionary
    shift_xyr = [x_ctr_rot, y_ctr_rot, rot_degree]
    list_path_img = []
    dic_rotated_images = {}

    #[5] Create a white array that rotates with the arrimg and use it as dummy later.
    arr_white = np.ones((h, w, 3), dtype=np.uint8) * 255
    arr_white_rot = arr_white.copy()
    
    #[6] Iterate through steps.
    for iter_steps in range(steps):
        #[8] Rotate the image
        if iter_steps == 0:     # No need to rotate the first image.
            arrimg_rot = arrimg
        else:                   # Rotate last image.
            arrimg_rot = M3TRA.rotate_image(last_arrimg_rot, shift_xyr)
            arr_white_rot = M3TRA.rotate_image(arr_white_rot, shift_xyr)
        arr_white_rot[(arr_white_rot[:, :, :] == [0, 0, 0]).all(axis=-1)] = [128, 128, 128] # Keep this line. It's for visual aid later.

        #[10] Add image to dictionary.
        iter_steps_reverse = steps - iter_steps
        dic_rotated_images[f'rotated_arrimg_{iter_steps_reverse:02d}'] = arrimg_rot
        last_arrimg_rot = arrimg_rot
                
        #[12] Update mask intersection for cropping
    
    #[10] Check crop box.
    min_x, min_y, max_x, max_y = find_largest_inscribe_rectangle_2(arr_white_rot, square=True)
    print(f'\n[Original Image] {h}, {w}')
    print(f'[Crop box] X {min_x}, {max_x}')
    print(f'[Crop box] Y {min_y}, {max_y}\n')

    #[14] Save all the images in the directory.
    dir_img_folder = os.path.dirname(path_img)
    for idx, (key, rotated_img) in enumerate(dic_rotated_images.items()):
        #[15] Crop image
        cropped_img = rotated_img[min_y:max_y, min_x:max_x] 
        #[16] Save image
        bsn_original = os.path.splitext(os.path.basename(path_img))[0]
        filename = f"{bsn_original}_step-{idx:02d}.jpg"
        path_save = os.path.join(dir_img_folder, filename)
        cv2.imwrite(path_save, cropped_img)
        list_path_img.append(path_save)
    
    #[18] Save white image.
    cv2.rectangle(arr_white_rot, (min_x, min_y), (max_x, max_y), (220,180,30), 20)
    box = min_x, min_y, max_x, max_y
    path_save = os.path.join(dir_img_folder, 'arr_white_rot.jpg')
    cv2.imwrite(path_save, arr_white_rot)
        
    # [20] Update with datetime.
    list_path_img = update_image_exif_datetime_yesterday(list_path_img)

    return list_path_img, box







#[5] Generate Rotate Example - End







