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
from tkinter.filedialog import askdirectory



import numpy as np

import Module_1_Processing as M1PCS #type: ignore        PCS = Processing
import Module_2_Image as M2IMG #type: ignore        PCS = Process



#[1] Set image path
#dir_img_folder = askdirectory(title='Select Folder')
dir_img_folder = r'D:\01_Floor\a_Ed\09_EECS\10_Python\04_OngoingTools\2023-0919_Image Mismatch Detection\05-MRD_Enhance Test\F1'
list_path_img, list_bsn_img = M1PCS.filter_file_with_keywords(dir_img_folder, ['.png', 'jpg'], [])


for i, iter_path in enumerate(list_path_img):
    #[3] A little blur
    path_img = M2IMG.generate_strongly_blurred_image_as_blank(iter_path, r = 250, blur_times=1, bsn='1-smallBlur')

    #[8] Vignetting correction.
    path_img = M2IMG.vignetting_correction(path_img, bsn='2-vignetCorr')
    df_stat = M2IMG.print_arrimg_stat(path_img, 5, 'normalize')

    #[12] Get image statistics for color normalization
    df_stat, b_pks, b_nks, g_pks, g_nks, r_pks, r_nks = M2IMG.print_arrimg_stat(path_img, 7, 'normalize', print_status=True)
    path_img = M2IMG.normalize_single_channel(path_img, 'B', b_nks, b_pks, bsn='3-C')
    path_img = M2IMG.normalize_single_channel(path_img, 'G', g_nks, g_pks, bsn='3-C')
    path_img = M2IMG.normalize_single_channel(path_img, 'R', r_nks, r_pks, bsn='3-C')
    
    #[13] Make a copy to prevent being dumped.
    ext = os.path.splitext(path_img)[1]
    path_processed = os.path.join(os.path.dirname(path_img), f'{i:0d}_Enh{ext}')
    shutil.copyfile(path_img, path_processed)

    df_stat, b_pks, b_nks, g_pks, g_nks, r_pks, r_nks = M2IMG.print_arrimg_stat(path_img, 5, 'normalize', print_status=False)



#[77] Remove blured and sharpen images
dir_img_folder = os.path.dirname(path_img)
list_keywords_select = list_bsn_img
list_keywords_discard  = []
M1PCS.batch_file_relocate(dir_img_folder, 
                        list_keywords_select, 
                        list_keywords_discard,
                        dump_folder_name = '01_Raw')



#[77] Remove blured and sharpen images
dir_img_folder = os.path.dirname(path_img)
list_keywords_select = ['Blur', 'CB', ' True', 'False', 'Corr', '-B', '-G', '-R'] + list_bsn_img
list_keywords_discard  = []
M1PCS.batch_file_relocate(dir_img_folder, 
                        list_keywords_select, 
                        list_keywords_discard,
                        dump_folder_name = '02_interm')





        
    
