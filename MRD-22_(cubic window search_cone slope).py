"""
[MRD-1_rotation detection.py]
Purpose: Calculate image shift due to rotation.
Author: Meng-Chi Ed Chen
Date: 2024-03-04
Reference:
    1.
    2.

Status: Complete.
"""
import os, sys, cv2, time, shutil, openpyxl
import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from datetime import datetime
from scipy.stats import norm
from scipy.stats import linregress
import ast

import MRD_0_module as MRD #type: ignore        MRD = Material Rotation Detection
import ICM_0_module as ICM #type: ignore        ICM = Invert Cone Model

"""
1. Define the origin as the upper left corner of the image.
2. Use raster coordinate system where x-axis increases to the right and the y-axis increases downward.
"""



dir_img_folder = r'D:\01_Floor\a_Ed\09_EECS\10_Python\04_OngoingTools\2023-0919_Image Mismatch Detection\04-IMD_Rotation Pictures\R8'

#[2] Determine path
bsn_folder = os.path.basename(dir_img_folder)
path_excel = os.path.join(dir_img_folder, f'{bsn_folder}.xlsx')

#[3] Read datasheets
list_of_dfs, list_of_sheetnames = MRD.read_excel_with_multiple_sheets_into_list_of_dfs(path_excel)
df_batch_pairs = list_of_dfs[-1]  # Get the last report dataframe.
list_of_dfs.pop(-1)  # Remove the last df in the list of dfs.
text = f'\n\n--[df_batch_matches]\n{tabulate(df_batch_pairs, headers="keys", tablefmt="orgtbl")}\n\n'
#print(text) # Print the last report datasheet.

#[4] Recover lists from the last report datasheet of df_batch_pairs
(list_bsn, list_total_matches, 
    list_mean_d_x, list_mean_d_y, list_mean_d_l, list_d_dir_degree, 
    list_mean_grad_x, list_mean_grad_y, list_mean_grad_l,
    list_grad_dir_degree, list_dist_to_gnd, 
    list_ctr_rot_x, list_ctr_rot_y, list_rot_degree, list_rot_ang_dir,
    list_dir2ctr_degree, list_dir2ctr_x, list_dir2ctr_y) = MRD.recover_lists_from_df_batch_pairs(df_batch_pairs)

print(f'[list_ctr_rot_x]: {list_ctr_rot_x}')
print(f'[list_ctr_rot_y]: {list_ctr_rot_y}')
print(f'[list_rot_degree]: {list_rot_degree}')


generate_gif = True
#generate_gif = False

text_log = f'[dir_img_folder]\n{dir_img_folder}\n{bsn_folder}\n\n\n'\


#...previous code

#[3] Initialize folder-wise lists
list_processed_sheetnames = []
list_opt_rot_degree = []
list_last_loss = []
list_avg_x_vector, list_avg_y_vector = [], []
list_opt_ctr_rot_x, list_opt_ctr_rot_y, list_opt_cone_slope = [], [], []

#[10] Take data from one dataframe, which is the matching result from 2 images.
for cnt, (iter_df, iter_text_title) in enumerate(zip(list_of_dfs, list_of_sheetnames)):
    list_processed_sheetnames.append(list_of_sheetnames[cnt])
    text = f'--------------------\n[{cnt+1}] {iter_text_title}\n'
    text_log += text

    #[11] Check the data and convert to raster coordinate.
    (list_coor1_x, list_coor1_y
    , list_coor2_x, list_coor2_y
    , list_d_x, list_d_y, list_d_l) = MRD.recover_lists_from_df_one_pair(iter_df)
    
    
    #[12] Set initial values
    ratio_to_ctr_rot = 1
    initial_ctr_rot_x = -141554    # Original: -140050.29506317805     Record: -139844, -140099, -140542, -140758, -140934, -141162
    initial_ctr_rot_y = -42393    # Original: -49280.323992299025     Record: -47766, -47012, -45662, -44984, -44425, -43672
    initial_rot_degree = list_rot_degree[cnt]
    initial_cone_slope = 0.000298   # Original: 0.0002936708030538492   Record: 0.000298, 0.000298, 0.000298, 0.000298, 0.000298, 0.000298
    
    #[12] Set hyperparameters
    #initial_learning_rate = 1
    lr_decay = 0.8
    num_epochs = 2000
    #learning_rate = initial_learning_rate
    lr_x, lr_y, lr_z = 1, 1, 1
    bias_x, bias_y, bias_z = 0, 0, 0
    
    
    #[13] Create empty lists
    list_bias_x, list_bias_y, list_bias_z = [], [], []
    #list_learning_rate = []
    list_lr_x, list_lr_y, list_lr_z = [], [], []
    list_idx_x, list_idx_y, list_idx_z = [], [], []
    
    #[20] Set mapping space length.
    space_length_xy = 1e-1
    space_length_z = 1e-5
    
    #[21] Set mapping density.
    data_per_axis = 10
    
    #[24] Set data density
    data_count_1D_xy = int(data_per_axis) + 1
    data_count_1D_z = int(data_per_axis) + 1
    data_count_all = data_count_1D_xy**2 * data_count_1D_z
    if data_count_all > 2000:
        raise ValueError(f'Data count is currently {data_count_all}. Consider keep it less than 2000 before making png.')


    for epoch in range(num_epochs):   

        #[25] Calculate value
        val_x = initial_ctr_rot_x + bias_x
        val_y = initial_ctr_rot_y + bias_y
        val_z = initial_cone_slope + bias_z
        
        #[26] Plot line spacing
        list_ctr_rot_x = np.linspace(val_x - space_length_xy / 2
                                    , val_x + space_length_xy / 2, data_count_1D_xy)
        list_ctr_rot_y = np.linspace(val_y - space_length_xy / 2
                                    , val_y + space_length_xy / 2, data_count_1D_xy)
        list_cone_slope = np.linspace(val_z - space_length_z / 2
                                    , val_z + space_length_z / 2, data_count_1D_z)
        
        #[28] Start grid search to find min loss.
        arr_loss, X_mesh, Y_mesh, Z_mesh, min_loss_idx, min_loss_value = ICM.grid_search_to_find_min_loss(list_coor2_x, list_coor2_y, list_d_l
                                                                , list_ctr_rot_x, list_ctr_rot_y, list_cone_slope)
        
        min_idx_x, min_idx_y, min_idx_z  = min_loss_idx
        
        #[29] Collect list
        list_bias_x.append(bias_x)
        list_bias_y.append(bias_y)
        list_bias_z.append(bias_z)
        list_lr_x.append(lr_x)
        list_lr_y.append(lr_y)
        list_lr_z.append(lr_z)
        list_idx_x.append(min_idx_x)
        list_idx_y.append(min_idx_y)
        list_idx_z.append(min_idx_z)
        
        #[31] Print progress
        text = f'  Epoch [{epoch+1}/{num_epochs}]\t min_loss (val, idx): {min_loss_value}, {min_loss_idx}.\t  lr_(x, y, z): {lr_x:.2e}, {lr_y:.2e}, {lr_z:.2e}.\t '\
                f' bias_(x, y, z) = {bias_x}, {bias_y}, {bias_z}\n'
        print(text)
        text_log += text 
        
        #[32] Update range to make min_loss_idx = [5, 5, 5] (suppose data_per_axis = 10)
        mid_point = data_per_axis // 2
        step_x = lr_x * 10e0
        step_y = lr_y * 10e0
        step_z = lr_z * 10e0
        good_x, good_y, good_z = 0, 0, 0
        
        #[34] Update range for x
        if min_idx_x < mid_point - 1:       # If the minimum idx of x is less than 5-1=4,
            bias_x -= step_x                # shift the window down so that we can enclose the minimum value.
        elif mid_point + 1 < min_idx_x:
            bias_x += step_x
        else:
            #print('--initial_ctr_rot_x is good enough.')
            good_x = 1
            
        #[35] Update range for y
        if min_idx_y < mid_point - 1:
            bias_y += step_y
        elif mid_point + 1 < min_idx_y:
            bias_y -= step_y
        else:
            #print('--initial_ctr_rot_y is good enough.')
            good_y = 1
            
        #[37] Update range for z    
        if min_idx_z < mid_point - 1:
            bias_z -= step_z
        elif mid_point + 1 < min_idx_z:
            bias_z += step_z    
        else:
            #print('--initial_cone_slope is good enough.')
            good_z = 1
            if epoch >= 2 and list_idx_x[-1] == list_idx_x[-2]\
                and list_idx_y[-1] == list_idx_y[-2]:
                lr_x /= lr_decay
                lr_y /= lr_decay
            
        #[38] Stop condition
        if good_x * good_y * good_z == 1:
            print('--initial_cone_slope is good enough.')
            break
        
        #[39] Decay the learning rate after each epoch
        if epoch >= 2:
            if list_idx_x[-1] != list_idx_x[-2] and good_x != 1:
                lr_x *= lr_decay
            if list_idx_y[-1] != list_idx_y[-2] and good_y != 1:
                lr_y *= lr_decay
            if list_idx_z[-1] != list_idx_z[-2] and good_z != 1:
                lr_z *= lr_decay   

    #[40] Convert list to scientific notation
    list_bias_x = ['{:.10f}'.format(x) for x in list_bias_x]
    list_bias_y = ['{:.10f}'.format(y) for y in list_bias_y]
    list_bias_z = ['{:.10f}'.format(z) for z in list_bias_z]
    list_lr_x = ['{:.10e}'.format(x) for x in list_lr_x]
    list_lr_y = ['{:.10e}'.format(y) for y in list_lr_y]
    list_lr_z = ['{:.10e}'.format(z) for z in list_lr_z]
    
    #[41] Collect lists as dataframe 
    df_grid_fit = pd.DataFrame({
        'bias_x': list_bias_x, 
        'bias_y': list_bias_y,
        'bias_z': list_bias_z,  
        'lr_x': list_lr_x, 
        'lr_y': list_lr_y, 
        'lr_z': list_lr_z, 
        'idx_x': list_idx_x,   
        'idx_y': list_idx_y, 
        'idx_z': list_idx_z})

    
    text = f'\n\n--[df_grid_fit]\n{tabulate(df_grid_fit, headers="keys", tablefmt="orgtbl")}\n\n'
    text_log += text

    #[52] Save log report
    timestamp = datetime.now().strftime("%Y-%m%d_%H%M%S")
    path_txt = os.path.join(dir_img_folder, f'{timestamp}_{bsn_folder}_3-grid search log.txt')
    MRD.create_txt_and_save_text(path_txt, text_log)
        
    #[53] Set viewing angle
    elev, azim = 80, -45
    
    #[54] Set mapping text and title.
    timestamp = datetime.now().strftime("%Y-%m%d_%H%M%S")
    text_note = ''
    text_title = 'Universal Mapping'
    path_save = os.path.join(dir_img_folder, f'{timestamp}_{iter_text_title}_3D_4-univ mapping.png')
    
    #[55] Generate plot.
    fig, ax = ICM.plot_3D_universal_mapping(arr_loss, X_mesh, Y_mesh, Z_mesh, min_loss_idx
                                            ,text_title, text_note
                                            , 30, -60, path_save)


    #[56] Generate more expensive 3D view of the optimization process.
    if generate_gif and cnt == 0:
        
        #[60] Generate even more expensive 3D gif.
        elve_azim_center = (45, -50) # Center elevation and azimuth
        elve_azim_start = (20, -20) # Start elevation and azimuth
        cnt_point = 50
        list_view_init = ICM.generate_circular_view_init(elve_azim_center, elve_azim_start, cnt_point, clockwise=True)
        path_save = os.path.join(dir_img_folder, f'{timestamp}_{iter_text_title}_3D_5-univ mapping.gif')
        ICM.convert_plot_to_gif(fig, ax, list_view_init, path_save, 100)

    break  # Do just one pair during testing.

sys.exit()





sys.exit()



