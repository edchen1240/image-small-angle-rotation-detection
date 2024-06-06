"""
[MRD-2_Find Matches and Calculate Displacements.py]
Purpose: Calculate image shift due to rotation.
Author: Meng-Chi Ed Chen
Date: 2024-03-04
Reference:
    1.
    2.

Status: Complete.
"""
import os, sys, cv2
import pandas as pd
import numpy as np
from tabulate import tabulate
from datetime import datetime


import Module_1_Processing as M1PCS #type: ignore        PCS = Processing
import Module_2_Image as M2IMG #type: ignore        PCS = Process
import Module_3_Train as M3TRA #type: ignore        TRA = Train
import Module_4_Visualization as M4VIS #type: ignore        VIS = Visualization
import Module_5_Image_File_Info as M5IFI #type: ignore        IFI = Image File Info




#[1] Ask directory
dir_img_folder = r'D:\01_Floor\a_Ed\09_EECS\10_Python\04_OngoingTools\2023-0919_Image Mismatch Detection\04-IMD_Rotation Pictures\I1'
dir_bsn = os.path.basename(dir_img_folder)  #The name of this folder.

#[3] Create text_log
text_log = f'[dir_img_folder]\n{dir_img_folder}\n{dir_bsn}\n\n'\


#[4] Match Settings
LR = 0.9  # LR = 0.7    # Decrease to have less point; increase to have more point.
f1_dl_upper_limit_ratio = 2     #1.5
f2_dl_std_limit = 2             # Usually 1, adjust to 2 for larger imageor or gradient is in the diagonal direction.
text_setting =  f'[Match Settings]\n'\
                f'LR: {LR}\n'\
                f'f1_dl_upper_limit_ratio: {f1_dl_upper_limit_ratio}\n'\
                f'f2_dl_std_limit: {f2_dl_std_limit}\n\n'
text_log += text_setting



#[6] Remove old files
list_keywords_select_to_dump = ['blurred', 'blur', 'blr', 'sharpen', 'shp', 'enh', 'marked', 'mk', 'crop', 'overlap', 'ovlp', 'arrow', 
                                '.xlsx', 'learning', 'optimiz', 'map', 'log', 'kernel', 'filter', 'opt', 'keep']
M1PCS.batch_file_relocate(dir_img_folder, 
                        list_keywords_select = list_keywords_select_to_dump,
                        dump_folder_name = '00_dump')

#[8] Select files
list_keywords_select = ['step']  # [Important] Name every file as "step" in order to be pick up.
list_keywords_discard = ['crop', 'blur', 'enh', '.xlsx', '.txt', 'keep']
df_image = M5IFI.tabulate_creation_time_of_files_in_dir(dir_img_folder, list_keywords_select, list_keywords_discard)
list_bsn = df_image['File Name'].tolist()

#[10] Initialize list
list_total_matches = []
list_mean_d_x, list_mean_d_y, list_mean_d_l, list_d_dir_degree = [], [], [], []
list_mean_grad_x, list_mean_grad_y, list_mean_grad_l = [], [], []
list_grad_dir_degree, list_dist_to_gnd = [], []
list_ctr_rot_x, list_ctr_rot_y, list_dist_ctr_rot = [], [], []
list_rot_degree, list_rot_ang_dir = [], []
list_dir2ctr_degree, list_dir2ctr_x, list_dir2ctr_y = [], [], []

#[20] interate through all the images in the folder
for cnt, iter_bsn in enumerate(list_bsn):
    print(f'[Iteration {cnt}]')
    #[22] Record 0th image, and skip to 1st image
    if cnt == 0:
        #[24] The 0th image.
        image_t1 = iter_bsn
        #[26] Append others with None.
        list_total_matches.append(None)
        # mean_d
        list_mean_d_x.append(None)
        list_mean_d_y.append(None)
        list_mean_d_l.append(None)
        list_d_dir_degree.append(None) 
        list_mean_grad_x.append(None)
        list_mean_grad_y.append(None) 
        list_mean_grad_l.append(None) 
        list_grad_dir_degree.append(None) 
        list_dist_to_gnd.append(None) 
        list_ctr_rot_x.append(None)
        list_ctr_rot_y.append(None)
        list_dist_ctr_rot.append(None) 
        list_rot_degree.append(None) 
        list_rot_ang_dir.append(None) 
        # dir2ctr
        list_dir2ctr_degree.append(None)
        list_dir2ctr_x.append(None)
        list_dir2ctr_y.append(None)
        continue 
    else:
        image_t2 = iter_bsn
    
    #[30] Acquire image pathes
    path_img_t1 = os.path.join(dir_img_folder, image_t1)
    path_img_t2 = os.path.join(dir_img_folder, image_t2)
    
    #[32] Create dataframe
    df_ImgTable = pd.DataFrame()
    df_ImgTable['Image Name'] = [image_t1, image_t2]
    df_ImgTable['Image Number'] = ('t1', 't2')
    
    #[34] Prepare image (size, blur, and enhancement)
    path_img_t1, path_img_t2 = M2IMG.crop_to_same_size(path_img_t1, path_img_t2)
    blur_k = 9  #17
    path_img_t1, path_img_t2 =  M2IMG.blur_image(path_img_t1, blur_k, '1blr'), \
                                M2IMG.blur_image(path_img_t2, blur_k, '1blr')
    sharp_k = 21
    ch, sigma_high, cl, sigma_low = 11, 3, 10, 5
    path_img_t1, path_img_t2 =  M2IMG.sharpen_image(path_img_t1, sharp_k, ch, sigma_high, cl, sigma_low, '3shp'), \
                                M2IMG.sharpen_image(path_img_t2, sharp_k, ch, sigma_high, cl, sigma_low, '3shp')
    
    clipLimit, tileGridSize = 0.2, 5
    #path_img_t1, path_img_t2 = M2IMG.enhance_image(path_img_t1, clipLimit, tileGridSize), M2IMG.enhance_image(path_img_t2, clipLimit, tileGridSize)
    img_t1, img_t2 = cv2.imread(path_img_t1, 1), cv2.imread(path_img_t2, 1)
    
    
    
    #[36] Detect displacment.
    list_good_matches, keypoints1, keypoints2 = M2IMG.find_good_matches_OpponentSIFT(img_t1, img_t2, LR)
        
    (list_coor1_x, list_coor1_y
    , list_coor2_x, list_coor2_y
    , list_d_x, list_d_y, list_d_l) = M2IMG.convert_match_to_lists(list_good_matches, keypoints1, keypoints2)
        
    (list_coor1_x, list_coor1_y
    , list_coor2_x, list_coor2_y
    , list_d_x, list_d_y, list_d_l
    , df_stat) = M2IMG.refining_match_quality(list_coor1_x, list_coor1_y
                                    , list_coor2_x, list_coor2_y
                                    , list_d_x, list_d_y, list_d_l
                                    , f1_dl_upper_limit_ratio, f2_dl_std_limit, dir_img_folder, cnt)
    
    #[40] Collect df_stat that is about the standard deviation of the displacement length in every filtering.
    text = f'\n{iter_bsn}\n--[df_stat]\n{tabulate(df_stat, headers="keys", tablefmt="orgtbl")}\n\n'
    #text = ''
    print(text)
    text_log += text

    large_font = False
    (list_d_l, list_coor2_x, list_coor2_y, path_marked_image) = M2IMG.mark_good_matches_on_img(list_d_l, 
                                                                                             list_coor2_x, 
                                                                                             list_coor2_y, 
                                                                                             large_font, 
                                                                                             path_img_t2,
                                                                                             '4mk')
    

    #[42] Generate df_one_pair.
    df_one_pair = M2IMG.generate_match_statistics_and_df_one_pair(list_coor1_x, list_coor1_y
                                                                , list_coor2_x, list_coor2_y
                                                                , list_d_x, list_d_y, list_d_l)
    
    
    #[44] Save data as excel.
    bsn_excel = f'{dir_bsn}.xlsx'
    sheet_name = f'step-{cnt:02d}'
    M1PCS.save_df_as_excel_add(df_one_pair, dir_img_folder, bsn_excel, sheet_name)
    M1PCS.adjust_multiple_column_widths(os.path.join(dir_img_folder, bsn_excel), sheet_name, ['B', 'C'], [20, 20])
    
    """
    [Note] Keep everything in Raster coordinate, including the rotational angle.
    1. Origin: Top-left of the image.
    2. Positive x-axis: Rightward.
    3. Positive y-axis: Downward.
    4. Positive rotation angle θ: clockwise.    
    """
    
    #[46] Calculate average_vector and number of matches.
    mean_d_x = np.mean(df_one_pair['d_x'], axis=0)
    mean_d_y = np.mean(df_one_pair['d_y'], axis=0)
    mean_d_l = np.mean(df_one_pair['d_l'], axis=0)
    
    d_dir_degree = np.degrees(np.arctan2(mean_d_y, mean_d_x))  # Do not flip y for converting from raster coordinate to Cartisian coordinate.
    list_mean_d_x.append(mean_d_x)
    list_mean_d_y.append(mean_d_y)
    list_mean_d_l.append(mean_d_l)
    list_d_dir_degree.append(d_dir_degree) 
    list_total_matches.append(len(df_one_pair.index))
    
    
    #[48] Find gradient
    (mean_grad_x
     , mean_grad_y
     , dist_to_gnd, text) = M2IMG.find_gradient_and_distance_to_z0_from_image_center(list_coor2_x
                                                                                , list_coor2_y
                                                                                , list_d_l)
    mean_grad_l = np.linalg.norm([mean_grad_x, mean_grad_y])
    grad_dir_degree = np.degrees(np.arctan2(mean_grad_y, mean_grad_x))  # Do not flip y for converting from raster coordinate to Cartisian coordinate.
    list_mean_grad_x.append(mean_grad_x)
    list_mean_grad_y.append(mean_grad_y) 
    list_mean_grad_l.append(mean_grad_l) 
    list_grad_dir_degree.append(grad_dir_degree) 
    list_dist_to_gnd.append(dist_to_gnd) 
    
    #[50] Estimate center of rotation (gnd) [Improtant] This part is criticle, check.
    h, w, _ = img_t1.shape
    half_h, half_w = h//2, w//2
    img_ctr_to_gnd_x = dist_to_gnd * np.cos(np.radians((180 + grad_dir_degree)))
    img_ctr_to_gnd_y = dist_to_gnd * np.sin(np.radians((180 + grad_dir_degree)))
    print(f'img_ctr_to_gnd_x, y: {img_ctr_to_gnd_x:.0f}, {img_ctr_to_gnd_y:.0f}')
    
    ctr_rot_x = img_ctr_to_gnd_x + half_w
    ctr_rot_y = img_ctr_to_gnd_y + half_h
    print(f'ctr_rot_x, y: {ctr_rot_x:.0f}, {ctr_rot_y:.0f}')    
    
    dist_ctr_rot = np.linalg.norm([ctr_rot_x, ctr_rot_y])
    rot_degree = np.degrees(np.arctan((mean_d_l/2)/dist_to_gnd)*2)     # rot_degree is always positive.
    list_ctr_rot_x.append(ctr_rot_x)
    list_ctr_rot_y.append(ctr_rot_y)
    list_dist_ctr_rot.append(dist_ctr_rot) 
    list_rot_degree.append(rot_degree) 
    
    
    
    
    #[51] The rotation is clockwise(+) or counter-clockwise()
    d_g_diff_degree = d_dir_degree - grad_dir_degree
    if d_g_diff_degree >= 180:
        d_g_diff_degree = d_g_diff_degree - 360
    elif d_g_diff_degree <=-180:
        d_g_diff_degree = d_g_diff_degree + 360
    
    if d_g_diff_degree > 0 :
        rot_ang_dir = +1
    else:
        rot_ang_dir = -1
    list_rot_ang_dir.append(rot_ang_dir)
    
    
    #[52] Center direction
    dir2ctr_degree = d_dir_degree + 90 * rot_ang_dir
    
    #[55] Handle the cases if angle is larger than +180 degree or samller than -180 degree.
    if dir2ctr_degree >= 180:
        dir2ctr_degree -= 360
    elif dir2ctr_degree <=-180:
        dir2ctr_degree += 360
        
    #[57] Handle the cases if angle is 0 or +-180 degrees
    if dir2ctr_degree == 0:
        dir2ctr_x, dir2ctr_y = 1, 0
    elif dir2ctr_degree == 90:
        dir2ctr_x, dir2ctr_y = 0, 1
    elif dir2ctr_degree == -90:
        dir2ctr_x, dir2ctr_y = 0, -1
    elif dir2ctr_degree == 180 or dir2ctr_degree == -180:
        dir2ctr_x, dir2ctr_y = -1, 0
    elif dir2ctr_degree > 90 or dir2ctr_degree < -90:
        dir2ctr_x = -1
        dir2ctr_y = -np.tan(np.radians(dir2ctr_degree))
    else:
        dir2ctr_x = 1
        dir2ctr_y = np.tan(np.radians(dir2ctr_degree))
        
    
    list_dir2ctr_degree.append(dir2ctr_degree)
    list_dir2ctr_x.append(dir2ctr_x)
    list_dir2ctr_y.append(dir2ctr_y)
    
    
    
    #[59] After calculate displacement with matches, perform overlap check.
    timestamp = datetime.now().strftime("%Y-%m%d-%H%M%S")
    iter_bsn_noExt = os.path.splitext(iter_bsn)[0]
    init_proposed_shift = [ctr_rot_x, ctr_rot_y, rot_degree*rot_ang_dir]        # Need to multiply direction here.
    arrimg_t1_rt = M3TRA.rotate_image(img_t1, init_proposed_shift)
    arrimg_overlap = M3TRA.overlap_two_images_together(arrimg_t1_rt, img_t2, color_1=[0, 128, 255], color_2=[255, 128, 0])
    path_arrimg_overlap = os.path.join(dir_img_folder, f'{timestamp}_{iter_bsn_noExt}_00_initial overlap check.jpg')
    cv2.imwrite(path_arrimg_overlap, arrimg_overlap) 


    #[62] Update image_t1. [Important]!
    image_t1 = iter_bsn
    
#[70] Length check
df_length_check = False
if df_length_check:
    print(f'[list_bsn] {len(list_bsn)}')
    print(f'[list_mean_d_x] {len(list_mean_d_x)}')
    print(f'[list_mean_d_y] {len(list_mean_d_y)}')
    print(f'[list_total_matches] {len(list_total_matches)}')

#[71] Report dataframe
#list_bsn.pop(0)
df_batch_pairs = pd.DataFrame(
    {'bsn': list_bsn
    , 'total_matches': list_total_matches
    , 'mean_d_x': list_mean_d_x
    , 'mean_d_y': list_mean_d_y
    , 'mean_d_l': list_mean_d_l
    , 'd_dir_degree': list_d_dir_degree
    , 'mean_grad_x': list_mean_grad_x
    , 'mean_grad_y': list_mean_grad_y
    , 'mean_grad_l': list_mean_grad_l
    , 'grad_dir_degree': list_grad_dir_degree
    , 'dist_to_gnd': list_dist_to_gnd
    , 'ctr_rot_x': list_ctr_rot_x
    , 'ctr_rot_y': list_ctr_rot_y
    , 'dist_ctr_rot':list_dist_ctr_rot
    , 'rot_degree': list_rot_degree
    , 'rot_ang_dir': list_rot_ang_dir
    , 'dir2ctr_degree':list_dir2ctr_degree
    , 'dir2ctr_x': list_dir2ctr_x
    , 'dir2ctr_y': list_dir2ctr_y
    })




#[72] Saver report dataframe
bsn_excel = f'{dir_bsn}.xlsx'
sheet_name = f'report'
M1PCS.save_df_as_excel_add(df_batch_pairs, dir_img_folder, bsn_excel, sheet_name)
M1PCS.adjust_multiple_column_widths(os.path.join(dir_img_folder, bsn_excel)
                                  , sheet_name
                                  , ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
                                  , [15,   15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15])

#[75] Add log
text = f'\n\n--[df_batch_matches]\n{tabulate(df_batch_pairs, headers="keys", tablefmt="orgtbl")}\n\n'
print(text)
text_log += text

#[76] Save log report
timestamp = datetime.now().strftime("%Y-%m%d-%H%M%S")
path_txt = os.path.join(dir_img_folder, f'{timestamp}_{dir_bsn}_1-match log.txt')
M1PCS.create_txt_and_save_text(path_txt, text_log)



#[77] Remove blured and sharpen images
list_keywords_select_to_dump = ['blurred', 'blur', 'blr', 'sharpen', 'shp', 'enh', 'kernel']
M1PCS.batch_file_relocate(dir_img_folder, 
                        list_keywords_select = list_keywords_select_to_dump,
                        dump_folder_name = '02_interm')


print(f'Complete')