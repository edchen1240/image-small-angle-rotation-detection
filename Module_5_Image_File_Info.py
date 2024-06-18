"""
[Module_5_Image_File_Info.py]
Purpose: Invert cone model
Author: Meng-Chi Ed Chen
Date: 2024-03-04
Reference:
    1.
    2.

Status: Complete.
"""
import os, sys, shutil
import pandas as pd
from tabulate import tabulate
from datetime import datetime, timedelta
import piexif
from PIL import Image


import Module_1_Processing as M1PCS #type: ignore        PCS = Process



#[1] Image Name and Time info process - Start

def tabulate_creation_time_of_files_in_dir(dir_folder, list_keywords_select=None, list_keywords_discard=None):
    """
    Select all files in a directory. Rank according to time.
    Create a dataframe, first column is the file basename, and the second column is the created time of the file, including both OS and EXIF data.
    """
    print('\n[tabulate_creation_time_of_files_in_dir]')
    #[1] List all files in the directory
    list_file_paths, list_file_names = M1PCS.filter_file_with_keywords(dir_folder, list_keywords_select, list_keywords_discard)
    #[2] Sort file paths by creation time (oldest first)
    list_file_paths.sort(key=lambda x: os.path.getctime(x))
    
    #[3] Create lists for the DataFrame
    files = [os.path.basename(path) for path in list_file_paths]
    creation_times_os = [os.path.getctime(path) for path in list_file_paths]
    creation_times_exif = []
    for path in list_file_paths:
        try:
            with Image.open(path) as img:
                exif_data = img._getexif()
                if exif_data is not None and 36867 in exif_data:
                    exif_time = datetime.strptime(exif_data[36867], '%Y:%m:%d %H:%M:%S')
                    creation_times_exif.append(exif_time.strftime('%Y-%m-%d, %H:%M:%S'))
                else:
                    creation_times_exif.append(None)
        except Exception as e:
            creation_times_exif.append(None)  # Append None if there's an error
    
    #[4] Create DataFrame
    df_file_time = pd.DataFrame({
        'File Name': files,
        'Creation Time (OS)': pd.to_datetime(creation_times_os, unit='s'),
        'Creation Time (EXIF)': pd.to_datetime(creation_times_exif)
    })

    #[5] Print DataFrame
    df_file_time = df_file_time.sort_values(by='File Name')
    print(tabulate(df_file_time, headers='keys', tablefmt='orgtbl'))
    
    return df_file_time

def extract_time_info_in_filename_into_EXIF_data(dir_img_folder):
    """
    Extract creation time from filename and register them into EXIF data of the files.
    """
    print('\n[extract_time_info_in_filename_into_EXIF_data]')
    #[1] Iterate over each file in the directory
    cnt_no_time_in_bsn = 0
    list_bsn = os.listdir(dir_img_folder)
    for cnt, iter_bsn in enumerate(list_bsn):
        print(f'[{cnt}] {iter_bsn}')
        filepath = os.path.join(dir_img_folder, iter_bsn)
        #[2] Check if the current file is an image
        if os.path.isfile(filepath) and iter_bsn.lower().endswith(('.jpg', '.jpeg', '.tiff')):
            
            #[3] Check if image has EXIF data, if not, create an empty dict
            img = Image.open(filepath)
            exif_data = img._getexif()

            if exif_data is None:
                # If no EXIF data exists, create an empty dictionary with an "Exif" key
                print(f'--There is no exif data in {iter_bsn}, just created one.')
                exif_dict = {"Exif": {}}
            elif 36867 in exif_data:
                # If EXIF data exists, load it
                exif_dict = piexif.load(img.info['exif'])
                print(f'--Tag 36867 is in the exif_data of {iter_bsn}. Skipping...')
                continue
                
            #[5] Check if iter_bsn contains "__", skip if not
            if '__' not in iter_bsn:
                print(f'--Filename {iter_bsn} does not follow the naming convention. Skipping...')
                cnt_no_time_in_bsn += 1
                continue

            #[6] Extract the timestamp from the iter_bsn. Assuming format is "Prefix__YYYY-MM-DDTHH-MM-SS.sss.jpg"
            try:
                timestamp_str = iter_bsn.split('__')[1].split('.')[0]
            except IndexError:
                # This block will now never be reached, but it's here if additional parsing logic is needed later
                continue

            #[5] Convert the timestamp into the EXIF date format
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H-%M-%S")
            exif_date = timestamp.strftime("%Y:%m:%d %H:%M:%S")

            #[7] Update the EXIF data with the new timestamp
            exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = exif_date
            exif_bytes = piexif.dump(exif_dict)

            #[8] Save the image with updated EXIF data
            img.save(filepath, "jpeg", exif=exif_bytes)
            img.close()
            bsn = os.path.basename(filepath)
            print(f'----File name:{bsn}. Extracted datetime: {exif_date}')

    if cnt_no_time_in_bsn == len(list_bsn):
        print(f'\n--All {cnt_no_time_in_bsn} files missing time information in file name.\n'
              '--Rank the file name alphebatically and assign yesterday current time with +1 mins interval.')

        #[15] Sort the list of filenames alphabetically
        list_bsn = sorted(list_bsn)
        list_path = [os.path.join(dir_img_folder, basename) for basename in list_bsn]
        list_image_file_sort_and_add_datetime(list_path)

    print()
    return



def list_image_file_sort_and_add_datetime(list_image):
    #[1] Get the current date and time
    current_datetime = datetime.now() - timedelta(days=1)  # Yesterday's date and time
    list_processed= []
    for idx, iter_path in enumerate(list_image):
    #[2] Check if the current file is an image
        if os.path.isfile(iter_path) and iter_path.lower().endswith(('.jpg', '.jpeg', '.tiff')):
            #[4] Increment the time by 1 minute for each file
            new_datetime = current_datetime + timedelta(minutes=idx)
            exif_date = new_datetime.strftime("%Y:%m:%d %H:%M:%S")

            #[5] Check if image has EXIF data, if not, create an empty dict
            img = Image.open(iter_path)
            exif_dict = piexif.load(img.info['exif']) if 'exif' in img.info else {"Exif": {}}
            #[6] Update the EXIF data with the new timestamp
            exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = exif_date
            exif_bytes = piexif.dump(exif_dict)

            #[7] Save the image with updated EXIF data
            img.save(iter_path, "jpeg", exif=exif_bytes)
            img.close()
            bsn = os.path.basename(iter_path)
            print(f'--{bsn}, {exif_date}')
            list_processed.append(iter_path)
    return list_processed
        



def get_exif_creation_time(file_path):
    try:
        with Image.open(file_path) as img:
            exif_data = img._getexif()
            if exif_data and 36867 in exif_data:
                creation_time = datetime.strptime(exif_data[36867], "%Y:%m:%d %H:%M:%S").strftime('%Y-%m-%d, %H:%M:%S')
                return creation_time
            else:
                print(f"--EXIF data is missing or does not contain 'DateTimeOriginal' for file: {os.path.basename(file_path)}. Using 'os.path.getctime'.")
    except Exception as e:
        print(f"Missing EXIF time information or error occurred: {e}")
    creation_time = datetime.fromtimestamp(os.path.getctime(file_path)).strftime('%Y-%m-%d, %H:%M:%S')
    return creation_time

def rename_image_files_according_to_EXIF_time(dir_folder
                                        , list_keywords_select=None
                                        , list_keywords_discard=None
                                        , rename_prefix=None):
    print('\n[rename_image_files_according_to_EXIF_time]')
    #[1] Filter file.
    list_file_paths, list_file_names = M1PCS.filter_file_with_keywords(dir_folder, list_keywords_select, list_keywords_discard)
    
    #[2] Sort file paths by EXIF creation time (oldest first)
    list_file_paths.sort(key=get_exif_creation_time)
    
    #[3] Rename and accumulate new paths
    list_file_paths_renamed = []
    list_file_names_renamed = []
    for index, old_path in enumerate(list_file_paths):
        new_file_name = f"{rename_prefix}_step-{index:02d}{os.path.splitext(old_path)[1]}"
        new_path = os.path.join(dir_folder, new_file_name)
        shutil.move(old_path, new_path)
        list_file_paths_renamed.append(new_path)
        list_file_names_renamed.append(os.path.basename(new_path))
    
    return list_file_paths_renamed, list_file_names_renamed

#[1] Image Name and Time info process - End













"""
#[3] Visualizing training result - Start
"""



#[3] Visualizing training result - End
