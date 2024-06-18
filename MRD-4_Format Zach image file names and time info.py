"""
[MRD-1_Format Zach image file names and time info.py]
Purpose: Rename images from the time informationin the file names of  Zach's material stacking system
Author: Meng-Chi Ed Chen
Date: 2024-03-04
Reference:
1.
2.

Status: Complete.
"""
import os, sys
import Module_5_Image_File_Info as M5IFI #type: ignore        IFI = Image File Info

dir_img_folder = r'D:\01_Floor\a_Ed\09_EECS\10_Python\04_OngoingTools\2023-0919_Image Mismatch Detection\05-MRD_Enhance Test\F1'
dir_bsn = os.path.basename(dir_img_folder)

#[0] Before creating EXIF data.
M5IFI.tabulate_creation_time_of_files_in_dir(dir_img_folder)

#[1] After creating EXIF data from file name.
M5IFI.extract_time_info_in_filename_into_EXIF_data(dir_img_folder)
#MRD.tabulate_creation_time_of_files_in_dir(dir_img_folder)

#[2] Rename files according to EXIF creation time.
list_keywords_select, list_keywords_discard = ['.jpg'], ['.xlsx']
list_image_path, list_image_bsn = M5IFI.rename_image_files_according_to_EXIF_time(dir_img_folder, list_keywords_select, list_keywords_discard, dir_bsn)
M5IFI.tabulate_creation_time_of_files_in_dir(dir_img_folder)

