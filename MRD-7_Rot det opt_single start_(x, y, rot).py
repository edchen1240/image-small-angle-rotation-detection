"""
[MRD-3_Rotation detection optimization.py]
Purpose: Calculate image shift due to rotation.
Author: Meng-Chi Ed Chen
Date: 2024-03-04
Reference:
    1.
    2.

Status: Complete.
"""
import os, sys, cv2, time, shutil, openpyxl, json
import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from datetime import datetime

sys.path.append(r'D:\01_Floor\a_Ed\09_EECS\10_Python\00_Classes and Functions')
import F04_EXLDF as F04_EXLDF       #type: ignore

import Module_1_Processing as M1PCS #type: ignore        PCS = Processing
import Module_2_Image as M2IMG #type: ignore        PCS = Process
import Module_3_Train as M3TRA #type: ignore        TRA = Train
import Module_4_Visualization as M4VIS #type: ignore        VIS = Visualization
import Module_5_Image_File_Info as M5IFI #type: ignore        IFI = Image File Info


"""
1. Define the origin as the upper left corner of the image.
2. Use raster coordinate system where x-axis increases to the right and the y-axis increases downward.
"""



dir_img_folder = r'D:\01_Floor\a_Ed\09_EECS\10_Python\04_OngoingTools\2023-0919_Image Mismatch Detection\04-IMD_Rotation Pictures\I1'

#[1] Clean old files
#list_keywords_select_to_dump = ['crop', 'enh', 'arrow', 'optimiz', 'map', '.txt']
#MRD.batch_file_relocate(dir_img_folder, list_keywords_select=list_keywords_select_to_dump, clean_dump_folder = False)

#[2] Determine path
bsn_folder = os.path.basename(dir_img_folder)
path_excel = os.path.join(dir_img_folder, f'{bsn_folder}.xlsx')

#[3] Read datasheets
list_of_dfs, list_of_sheetnames = F04_EXLDF.read_excel_with_multiple_sheets_into_list_of_dfs(path_excel)

#[4] Remove sheets that contain 'opt'.
for cnt, iter_sheetname in enumerate(list_of_sheetnames):
    print(iter_sheetname)
    if 'opt' in iter_sheetname:
        print(f'Removing sheet: {iter_sheetname}')
        list_of_dfs.pop(cnt)
        list_of_sheetnames.pop(cnt)
    #[5] Get the last report dataframe.
    elif 'report' in iter_sheetname:
        df_batch_pairs = list_of_dfs[cnt] 
        list_of_dfs.pop(cnt)
        list_of_sheetnames.pop(cnt)

text = f'\n\n--[df_batch_matches]\n{tabulate(df_batch_pairs, headers="keys", tablefmt="orgtbl")}\n\n'
#print(text) # Print the last report datasheet.

#[6] Recover lists from the last report datasheet of df_batch_pairs
(list_bsn, list_total_matches, 
    list_mean_d_x, list_mean_d_y, list_mean_d_l, list_d_dir_degree, 
    list_mean_grad_x, list_mean_grad_y, list_mean_grad_l,
    list_grad_dir_degree, list_dist_to_gnd, 
    list_ctr_rot_x, list_ctr_rot_y, list_rot_degree, list_rot_ang_dir,
    list_dir2ctr_degree, list_dir2ctr_x, list_dir2ctr_y) = M1PCS.recover_lists_from_df_batch_pairs(df_batch_pairs, pop_none=False)

print(f'[list_ctr_rot_x]: {list_ctr_rot_x}')
print(f'[list_ctr_rot_y]: {list_ctr_rot_y}')
print(f'[list_rot_degree]: {list_rot_degree}')

#[7] Expensive plot setting and start text log. 
text_log = f'[dir_img_folder]\n{dir_img_folder}\n{bsn_folder}\n\n\n'
text_hyper = f'[Hyperparameters]\n'

#[8] Initialize folder-wise lists
list_x_opt, list_y_opt, list_r_opt = [], [], []
list_processed_sheetnames = []
list_len_epoch = []
list_last_loss, list_last_loss_eff = [], []


#[18] Take data from one dataframe, which is the matching result from 2 images.
for cnt, iter_bsn in enumerate(list_bsn):
    #[19] Record 0th image, and skip to 1st image
    if cnt == 0:
        #[20] The 0th image.
        image_t1 = iter_bsn
        continue 
    else:
        image_t2 = iter_bsn
    
    #[22] Text log for every pari of batch
    iter_bsn_noExt = os.path.splitext(iter_bsn)[0]
    timestamp = datetime.now().strftime("%Y-%m%d-%H%M%S")
    list_processed_sheetnames.append(list_of_sheetnames[cnt-1])  # The first one is skipped, so "cnt-1" here.
    text = f'--------------------\n[{cnt+1}] {list_bsn[cnt]}\n'
    text_log += text
    
    
    
    #[23] Set initial values
    init_ctr_rot_x = list_ctr_rot_x[cnt]
    init_ctr_rot_y = list_ctr_rot_y[cnt]
    init_rot_degree = list_rot_degree[cnt] * list_rot_ang_dir[cnt]

    #[24] Acquire image pathes and read images
    path_img_t1 = os.path.join(dir_img_folder, image_t1)
    path_img_t2 = os.path.join(dir_img_folder, image_t2)
    arrimg_t1, arrimg_t2 = cv2.imread(path_img_t1, 1), cv2.imread(path_img_t2, 1)
    
    #[25] Set initial values
    init_shift_0 = [init_ctr_rot_x, init_ctr_rot_y, init_rot_degree] # x_ctr_rot, y_ctr_rot, rot_theta
    
    #[26] Initiate list for special marker
    list_marker_idx = []
    GT_xyrl = None  # Initiate as None.
    
    
    #[30] Set hyperparameters
    """
    [Experiences about setting hyperparameters]
    1. If eps decreases, increase learning rate. Minimum eps is 1e-3. 5e-4 is too small.
        (init_lr, esp, rot_eps_constraint) = (1e0, 1e-3, 1e-2)
        (init_lr, esp, rot_eps_constraint) = (1e3, 5e-4, 1e-2)
    2. When eps is too large, loss goes to minimum and come back because the approximation of the gradient becomes less accurate.
    3. When init_lr is to larger, (x, y, r) oscillates.
    4. If only r is oscillating and (x,y) are not, reduce rot_lr_constraint to make the learning rate of r smaller.
    5. If you see a straight line in 2D or 3D opt plot, that means eps is too small for x and y to find gradient, so it's not updating.
    """
    dic_hyperp = {
        'time_stamp': timestamp,
        'init_lr': 1e2,                                         # 1e3 is too big. 1e-2 too slow. Currently 1e2 is good. (1e2)
        'scope_penalty': 0,                                     # This doesn't help with the optimization. Set to 0, not even 1. (0)
        'eps': 1e-2,                                            # Gradiant increase if this value reduces. # eps < 1e-3 grad = 0. (1e-3)
        'num_epochs': 200,                                      # Usually 100 epoch. (100)
        'min_epoch_ratio': 0.5,                                 # The training can only stop after {min_epoch_ratio*num_epochs} epoches. (0.5)
        'loss_cal_selection': 'loss_mse_normal',                # loss_cal_selection == 'loss_mse_normal' or 'loss_mse_radial_weighted'
        'power_of': 2,                                          # 2 = MSE, 4 = MQE. (2)
        'grad_cal_selection': 'grad_ctr_diff',                  # grad_fwd_diff = forward difference, grad_ctr_diff = center difference.
        'rot_eps_constraint': 1e-4,                             # rot is more sensitive, so it might need smaller eps. (1e-3)
        'rot_lr_constraint': 5e-4,                              # 1 Means no constraint, typically use 1e-4 ~ 1e-5 if necessary. (5e-6)
        'rot_update_constraint': 1,                             # 1 Means no constraint, use 1e-1 or 1e-2 if necessary. 
        'AO_optimizer': 'AdamOptimizer_normal',                 # AdamOptimizer_normal, AdamOptimizer_track_lr
        'AO_beta1': 0.995,                                        # Decrease AO_beta1 if (x,y) oscillates. (0.7)
        'AO_beta2': 0.9999, 
        'AO_eps': 1e-6,                                         # To avoid dividied by zero. Not important acutally.                             # Algorithm tend to find futher COR with smaller AOR. This force it to try closer COR with larger AOR
        'generate_3D_optimization_process': True,
        'generate_initial_ovlp': False,
        'show_GT': True
    }
    init_lr = dic_hyperp['init_lr']
    scope_penalty = dic_hyperp['scope_penalty']  # Notice the typo in 'scope_panalty' should be 'scope_penalty' if corrected
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
    generate_3D_optimization_process = dic_hyperp['generate_3D_optimization_process']
    generate_initial_ovlp = dic_hyperp['generate_initial_ovlp']
    show_GT = dic_hyperp['show_GT']

    
    #[31] Compare with ground truth # Do you want to insert GT as 0th value to see how far it is from the opt result?
    if show_GT:
        latest_path_txt, x_corr, y_corr, r_real = M1PCS.read_GT_from_gen_setting_txt_file(dir_img_folder)
        GT_shift = [x_corr, y_corr, r_real] 
        text = f'\n[GT] {GT_shift[0]}, {GT_shift[1]}, {GT_shift[2]}\n'
        text_log += text
        print(GT_shift)
    #GT_shift = [4572, 4999, 3]         # Do you want to start with GT?
    #init_shift_0 = GT_shift  # Do you want to start optimization from GT rather than coarse (x, y, r) calculated from the feature matching result.?
    

    #[32] Add hyperparameter info into text_hyper
    if cnt <= 1:
        text_hyper += json.dumps(dic_hyperp, indent=4) + '\n\n----------\n'
    
    
    #[32] Before optimization start, use initial valuse to perform overlap check.
    if generate_initial_ovlp:
        arrimg_t1_rt = M3TRA.rotate_image(arrimg_t1, init_shift_0)
        arrimg_overlap = M3TRA.overlap_two_images_together(arrimg_t1_rt, arrimg_t2, color_1=[0, 128, 255], color_2=[255, 128, 0])
        init_shift_0_text = f'{int(init_shift_0[0])}, {int(init_shift_0[1])}, {round(init_shift_0[0], 2)}'
        path_arrimg_overlap = os.path.join(dir_img_folder, f'{timestamp}_{iter_bsn_noExt}_00_initial ovlp({init_shift_0_text}).jpg')
        cv2.imwrite(path_arrimg_overlap, arrimg_overlap) 
    
    #[34] Train to minimize substraction loss
    name_tag = 'p0 (1/1)'
    (df_train_recrod_xyrot_1, text_log) \
          = M3TRA.train_x_y_rot_to_minimize_substraction_loss_eff(arrimg_t2, arrimg_t1,
                                                                init_shift_0,
                                                                dic_hyperp,
                                                                text_log, name_tag)
    
    
    
    """
    Training finished, collect training record.   
    """
    #[35] Recover df_train_recrod
    (list_ss_x, list_ss_y, list_ss_r,
            list_grad_x, list_grad_y, list_grad_r,
            list_x, list_y, list_r,
            list_loss, list_loss_eff, list_eff_pixel), idx_lowest_loss_eff= M3TRA.recover_lists_from_df_train_recrod_xyrot(df_train_recrod_xyrot_1)
    text_log += f'\n{tabulate(df_train_recrod_xyrot_1, headers="keys", tablefmt="orgtbl")}\n\n'
    print(f'\n\n[list_loss_eff]_1\n{[int(x) for x in list_loss_eff]}\n\n')
    


    #[47] Extract optimized values
    i_opt = idx_lowest_loss_eff
    x_opt, y_opt, r_opt = list_x[i_opt], list_y[i_opt], list_r[i_opt]
    opt_shift_1 = np.array([x_opt, y_opt, r_opt])
    len_epoch = len(list_loss)
    loss_opt = list_loss[i_opt]
    loss_eff_opt = list_loss_eff[i_opt]
    text =  f'\n\n[Best Iteration]\n  idx_lowest_loss_eff: {idx_lowest_loss_eff}\n'\
            f'  len_epoch: {len_epoch}\n'\
            f'  loss_opt, loss_eff_opt: {loss_opt}, {loss_eff_opt}\n'\
            f'  init_shift_0: {init_shift_0[0]:.2f}, {init_shift_0[1]:.2f}, {init_shift_0[2]:.2f}\n'\
            f'  opt_shift_1:  {opt_shift_1[0]:.2f}, {opt_shift_1[1]:.2f}, {opt_shift_1[2]:.2f}'
    text_log += text
    print(text, end='')
    

        
        
    #[66] Insert ground truth as the 0th value. This must be placed after the last generation of "recover_lists_from_df_train_recrod_xyrot".
    if show_GT:
        arrimg1_rot = M3TRA.rotate_image(arrimg_t1, GT_shift)
        loss, pixel_eff, loss_eff, text = M3TRA.arrimg_subtraction_loss_selector(loss_cal_selection, 
                                                                        arrimg_t2, arrimg1_rot, 
                                                                        GT_shift, 
                                                                        power_of, 
                                                                        scope_penalty)
        text_GT = f'\n\n[GT] {GT_shift}\n'\
                    f'--loss_eff: {loss_eff:.4f}\n\n'
        text_log += text_GT
        print(text_GT)
        print(f'\n\n[list_loss_eff]_2\n{[int(x) for x in list_loss_eff]}\n\n')
        GT_xyrl = [GT_shift[0], GT_shift[1], GT_shift[2], loss_eff]
        

    
    #[63] Collect optimized values
    list_x_opt.append(x_opt)
    list_y_opt.append(y_opt)
    list_r_opt.append(r_opt)
    list_len_epoch.append(len_epoch)
    list_last_loss.append(loss_opt)
    list_last_loss_eff.append(loss_eff_opt)
    
    
    #[51] Check two opt lengthes
    list_marker_idx.append(0)     # Started value
    list_marker_idx.append(i_opt) # Opt
    text =  f'\n\n[list_marker_idx] {list_marker_idx}\n\n'
    print(text)
    text_log += text
    
    
        
    """
    Prepare to record training data.
    """
    #[71] Record text data
    text_note = f'[{timestamp}]_{iter_bsn}\n'\
                f'[Initial] (x, y, r):      {init_ctr_rot_x:.4f},  {init_ctr_rot_y:.4f}, {init_rot_degree:.4f}\n'\
                f'init_lr, scope_penalty, epsilon:  {init_lr},      {scope_penalty},      {eps}\n'\
                f'Epoch (set, ratio, 1st length):     {num_epochs},   {min_epoch_ratio}, {len_epoch}\n'\
                f'loss_cal, power_of(MSE/MQE), grad_cal: {loss_cal_selection}, {power_of}, {grad_cal_selection}\n'\
                f'rot_constraints (_eps, _lr, _update):     {rot_eps_constraint}, {rot_lr_constraint}, {rot_update_constraint}\n'\
                f'{AO_optimizer}:           {AO_beta1}, {AO_eps}\n'\
                f'loss_opt, loss_eff_opt:     {loss_opt:.4f},    {loss_eff_opt:.4f}\n'\
                f'[Optimized] (x, y, r, idx):      {x_opt:.4f},  {y_opt:.4f},  {r_opt:.4f},  {int(idx_lowest_loss_eff)}\n'
    text_log += f'{text_note}\n\n\n\n'
    print(f'\n\n{text_note}\n\n')


    #[75] Rotate image
    true_rot_degree = r_opt # No need to multiply by list_rot_ang_dir[cnt] here.
    final_opt_shift = [x_opt, y_opt, true_rot_degree]
    print(f'\n\n[final_opt_shift]\n{final_opt_shift}\n\n\n')
    arrimg_t1_rt = M3TRA.rotate_image(arrimg_t1, final_opt_shift)
    
    #[76] Overlap images to check
    arrimg_overlap = M3TRA.overlap_two_images_together(arrimg_t1_rt, arrimg_t2, color_1=[0, 128, 255], color_2=[255, 128, 0])
    final_opt_shift_text = f'{round(final_opt_shift[0],0)}, {round(final_opt_shift[1],0)}, {round(final_opt_shift[0],2)}'
    path_arrimg_overlap = os.path.join(dir_img_folder, f'{timestamp}_{iter_bsn_noExt}_01_final ovlp({final_opt_shift_text}).jpg')
    cv2.imwrite(path_arrimg_overlap, arrimg_overlap) 
    

    """
    Plot training record and optimization process.
    """
    #[81] Plot training record
    path_save = os.path.join(dir_img_folder, f'{timestamp}_{iter_bsn_noExt}_02_learning record plot.png')
    M4VIS.plot_2loss_and_3learning_rate_record(list_loss, list_loss_eff, 
                                             list_ss_x, list_ss_y, list_ss_r, rot_lr_constraint, 
                                             iter_bsn, text_note, path_save)
    
    #[83] Plot and save 2D opt process
    path_save = os.path.join(dir_img_folder, f'{timestamp}_{iter_bsn_noExt}_03_2D opt process sep.png')
    M4VIS.plot_2D_optimization_process_xyLoss_separate( list_x, 'Center of Rotation (x)', 
                                                        list_y, 'Center of Rotation (y)', 
                                                        list_r, 'Angel of Rotation (degree)', 
                                                        list_loss_eff, 'Effective Loss',
                                                        GT_xyrl, list_marker_idx, f'2D Optimization Record of {iter_bsn}', text_note, path_save)
    
    

    #[84] Generate more expensive 3D view of the optimization process.
    path_save = os.path.join(dir_img_folder, f'{timestamp}_{iter_bsn_noExt}_04_3D opt process.png')
    fig, ax =    M4VIS.plot_3D_optimization_process(list_x, 'Center of Rotation (x)', 
                                                    list_y, 'Center of Rotation (y)', 
                                                    list_r, 'Angel of Rotation (degree)', 
                                                    list_loss_eff, 'Effective Loss',
                                                    GT_xyrl, list_marker_idx, f'3D Optimization Record of {iter_bsn}', text_note, path_save)
    
    #[85] Generate even more expensive 3D gif.
    if generate_3D_optimization_process and cnt <= 1:
        elve_azim_center = (50, -50) # Center elevation and azimuth
        elve_azim_start = (30, -20) # Start elevation and azimuth
        cnt_point = 50
        list_view_init = M4VIS.generate_circular_view_init(elve_azim_center, elve_azim_start, cnt_point, clockwise=True)
        path_save = os.path.join(dir_img_folder, f'{timestamp}_{iter_bsn_noExt}_05_3D opt process anime.gif')
        M4VIS.convert_plot_to_gif(fig, ax, list_view_init, path_save, 100)
        
    #[86] Update image_t1. [Important]!
    image_t1 = iter_bsn
    #break  # Do just one pair during testing.


"""
Folder batch finished, start collecting data for report.
"""
#[91] Check the length.
list_bsn.pop(0)     # The first image wasn't used.
df_length_check = False
if df_length_check:
    print(f'[iter_bsn] {len(list_processed_sheetnames)}')
    print(f'[list_x_opt] {len(list_x_opt)}')
    print(f'[list_y_opt] {len(list_y_opt)}')
    print(f'[list_r_opt] {len(list_r_opt)}')
    print(f'[list_len_epoch] {len(list_len_epoch)}')
    print(f'[list_last_loss] {len(list_last_loss)}')
    print(f'[list_last_loss_eff] {len(list_last_loss_eff)}')

#[92] Create dataframe for optimized result.
df_opt = pd.DataFrame({'Image Name': list_processed_sheetnames,
                        'list_x_opt': list_x_opt,
                        'list_y_opt': list_y_opt,
                        'list_r_opt': list_r_opt,
                        'list_len_epoch': list_len_epoch,
                        'list_last_loss': list_last_loss,
                        'list_last_loss_eff': list_last_loss_eff})

print(f'\n\n{tabulate(df_opt, headers="keys", tablefmt="orgtbl")}')
text_df_opt = f'\n{tabulate(df_opt, headers="keys", tablefmt="orgtbl")}\n\n----------\n\n'


#[93] Save dataframe to excel
sheet_name = 'opt'
bsn_xlsx = os.path.basename(path_excel)
real_sheet_name = M1PCS.save_df_as_excel_add(df_opt, dir_img_folder, bsn_xlsx, sheet_name)
M1PCS.adjust_multiple_column_widths(os.path.join(dir_img_folder, bsn_xlsx)
                                  , real_sheet_name
                                  , ['B', 'C', 'D', 'E', 'F', 'G', 'H']
                                  , [ 20,  20,  20,  20,  20,  20,  20])

#[95] Save log report
last_sheet_name = M1PCS.return_name_of_last_excel_sheet(path_excel)
M1PCS.add_dictionary_to_excel(dic_hyperp, path_excel, last_sheet_name, cell_location='B20')
path_txt = os.path.join(dir_img_folder, f'{timestamp}_{bsn_folder}_3-subtraction fit log_{last_sheet_name}.txt')
M1PCS.create_txt_and_save_text(path_txt, text_hyper + text_df_opt + text_log)

print(f'Complete')