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
import os, sys, shutil, openpyxl, re
import pandas as pd
from sklearn.linear_model import RANSACRegressor










#[2] About Files - Start

def filter_file_with_keywords(dir_folder
                              , list_keywords_select=None
                              , list_keywords_discard=None):
    """
    [2023-1001] 
    Filters only files (no folders) in a given directory based on presence or absence of keywords in filenames.
    Parameters:
    - dir_folder: Directory containing files to filter.
    - list_keywords_select: List of keywords that selected files must contain.
    - list_keywords_discard: List of keywords that selected files must not contain.
    Returns:
    - A tuple containing a list of file paths and a list of file names that match the criteria.
    Raises:
    - Warning if the directory is not found.
    """
    print('\n[filter_file_with_keywords]')
    if list_keywords_select is None:
        list_keywords_select = []
    if list_keywords_discard is None:
        list_keywords_discard = []
    # Check if directory exists
    if not os.path.isdir(dir_folder):
        raise FileNotFoundError(f"The directory '{dir_folder}' was not found.")
        sys.exit(1)
    # Get list of files in the directory
    list_file_names = [file for file in os.listdir(dir_folder)
                       if os.path.isfile(os.path.join(dir_folder, file))]
    # Filter names by keywords to select
    if list_keywords_select:
        list_file_names = [file for file in list_file_names 
                           if any(keyword in file for keyword in list_keywords_select)]
    # Filter names by keywords to discard
    if list_keywords_discard:
        list_file_names = [file for file in list_file_names 
                           if not any(keyword in file for keyword in list_keywords_discard)]
    # Generate file paths
    list_file_paths = [os.path.join(dir_folder, file) for file in list_file_names]
    # Print result
    print('--list_file_names:', list_file_names, end='\n\n')
    return list_file_paths, list_file_names

def batch_file_relocate(dir_folder, list_keywords_select = [], list_keywords_discard  = [], dump_folder_name = 'dump', clean_dump_folder = False):
    print('\n[batch_file_relocate]')

    filtered_files = filter_file_with_keywords(dir_folder, list_keywords_select, list_keywords_discard)
    list_file_paths = [file for file in filtered_files[0] if not os.path.isdir(os.path.join(dir_folder, file))]  # Don't move folders.
    path_destination = os.path.join(dir_folder, dump_folder_name)

    if clean_dump_folder:
        if os.path.exists(path_destination) and os.path.isdir(path_destination):
            for file in os.listdir(path_destination):
                file_path = os.path.join(path_destination, file)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # Remove files and links
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # Remove directories
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
            print('--[Dump folder cleaned]')
                    
    if len(list_file_paths) == 0:
        return  # Exit the function if there is nothing to move.

    try:
        os.mkdir(path_destination)
        print("--Directory wasn't exists. Just made one.")
    except FileExistsError:
        print("--Directory already exists. Skipping directory creation.")
    
    print(f'--[ {len(list_file_paths)} files to move]:', filtered_files[1])

    full_directory_list = [os.path.join(dir_folder, file) for file in list_file_paths]

    for iter_file in full_directory_list:
        file_name = os.path.basename(iter_file)
        try: 
            shutil.move(iter_file, path_destination)  # Move the files.
        except shutil.Error:  # If file already existed in dump folder, generate a new file name with a number suffix.
            file_name_parts = os.path.splitext(file_name)
            new_file_name = file_name_parts[0] + '_dup' + file_name_parts[1]
            new_destination_path = os.path.join(path_destination, new_file_name)
            try:
                shutil.move(iter_file, new_destination_path)
            except:
                print('--[Error] Failed to move or rename the file.')
                
    print('--batch_file_relocate complete')

    list_file_names = [os.path.basename(file) for file in full_directory_list]
    print('--list_file_names:', list_file_names, end='\n\n')
    
    return list_file_paths, list_file_names


def rename_the_only_jpg_file(dir):
    """
    1. There should be only one jpg in the directory. If not, raise error.
    2. Name the only jpg as the basename of dir, and return the path of that jpg.
    """
    #[1] Find jpg file in a dir and count.
    list_bsn = os.listdir(dir)
    bsn_jpgs = [i_bsn for i_bsn in list_bsn if i_bsn.lower().endswith('.jpg')]
    if len(bsn_jpgs) != 1:
        raise ValueError("\nThere should be exactly one jpg file in the directory.")
    bsn_that_jpg = bsn_jpgs [0]
    
    #[2] Get the base name of the directory
    bsn_dir = os.path.basename(dir)
    
    
    #[3] Construct the full paths for the old and new file names
    old_jpg_path = os.path.join(dir, bsn_that_jpg)
    new_jpg_path = os.path.join(dir, f'{bsn_dir}.jpg')
    
    #[4] Rename the file
    os.rename(old_jpg_path, new_jpg_path)
    
    return new_jpg_path



#[2] About Files - End







#[4] Text file .TXT - Start


def create_txt_and_save_text(path_txt, text):
    os.makedirs(os.path.dirname(path_txt), exist_ok=True)       # Ensure the directory where the file will be saved exists
    # Open the file in write mode ('w') which will create the file if it doesn't exist
    with open(path_txt, 'w') as file:
        file.write(text)
        
        
def read_GT_from_gen_setting_txt_file(dir_img_folder):
    print('\n[read_GT_from_gen_setting_file]')
    latest_path_txt = None
    latest_string_txt = None
    latest_mod_time = -1
    file_name = 'gen setting'

    #[1] Iterate through the files in the directory
    for path_file in os.listdir(dir_img_folder):
        #[2] Check if the file name contains the specified string and if it's a text file
        if file_name in path_file and path_file.endswith('.txt'):
            path_txt = os.path.join(dir_img_folder, path_file)
            #[3] Get the modification time of the file
            mod_time = os.path.getmtime(path_txt)
            if mod_time > latest_mod_time:
                latest_mod_time = mod_time
                latest_path_txt = path_txt
                #[4] Read the content of the file
                with open(latest_path_txt, 'r') as file:
                    latest_string_txt = file.read()
                    
    #[10] Find match text
    GT_match = re.search(r'\[GT\]\s+(-?\d+),\s+(-?\d+),\s+(-?\d+)', latest_string_txt)           
    if GT_match:
        x_corr = int(GT_match.group(1))
        y_corr = int(GT_match.group(2))
        r_real = int(GT_match.group(3))
    else:
        raise ValueError("The input string does not contain the required information in the expected format.")
    
    return latest_path_txt, x_corr, y_corr, r_real



#[4] Text file .TXT - End











#[5] Dataframe - Start

#This function seems not in use. (2024-05-28)
def recover_lists_from_df_one_pair(df_one_pair):
    """
    This function reads the following dataframe into lists
    df_one_pair = pd.DataFrame({
        'x_coor1': list_x_coor1,
        'y_coor1': list_y_coor1,
        'x_coor2': list_x_coor2,
        'y_coor2': list_y_coor2,
        'x_vector': list_d_x,
        'y_vector': list_d_y,
        'dspls': list_d_l})
    """
    # Extracting each list from the dataframe columns
    list_coor1_x = df_one_pair['coor1_x'].tolist()
    list_coor1_y = df_one_pair['coor1_y'].tolist()
    list_coor2_x = df_one_pair['coor2_x'].tolist()
    list_coor2_y = df_one_pair['coor2_y'].tolist()
    list_d_x = df_one_pair['d_x'].tolist()
    list_d_y = df_one_pair['d_y'].tolist()
    list_d_l = df_one_pair['d_l'].tolist()
    return (list_coor1_x, list_coor1_y
            , list_coor2_x, list_coor2_y
            , list_d_x, list_d_y, list_d_l)



def recover_lists_from_df_batch_pairs(df_batch_pairs, pop_none=True):
    """
    This function reads the following dataframe into lists:
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

    If `pop_none` is True, rows where 'total_matches' is None are removed before extracting lists.
    """
    if pop_none:
        # Create a boolean mask where True corresponds to rows with non-None 'total_matches'
        mask = df_batch_pairs['total_matches'].notna()
        # Apply the mask to filter out rows where 'total_matches' is None
        df_batch_pairs = df_batch_pairs[mask]
    
    df_batch_pairs = truncate_df_at_first_all_nan(df_batch_pairs)

    # Extracting each list from the dataframe columns
    list_bsn = df_batch_pairs['bsn'].tolist()
    list_total_matches = df_batch_pairs['total_matches'].tolist()
    list_mean_d_x = df_batch_pairs['mean_d_x'].tolist()
    list_mean_d_y = df_batch_pairs['mean_d_y'].tolist()
    list_mean_d_l = df_batch_pairs['mean_d_l'].tolist()
    list_d_dir_degree = df_batch_pairs['d_dir_degree'].tolist()
    list_mean_grad_x = df_batch_pairs['mean_grad_x'].tolist()
    list_mean_grad_y = df_batch_pairs['mean_grad_y'].tolist()
    list_mean_grad_l = df_batch_pairs['mean_grad_l'].tolist()
    list_grad_dir_degree = df_batch_pairs['grad_dir_degree'].tolist()
    list_dist_to_gnd = df_batch_pairs['dist_to_gnd'].tolist()
    list_ctr_rot_x = df_batch_pairs['ctr_rot_x'].tolist()
    list_ctr_rot_y = df_batch_pairs['ctr_rot_y'].tolist()
    list_rot_degree = df_batch_pairs['rot_degree'].tolist()
    list_rot_ang_dir = df_batch_pairs['rot_ang_dir'].tolist()
    list_dir2ctr_degree = df_batch_pairs['dir2ctr_degree'].tolist()
    list_dir2ctr_x = df_batch_pairs['dir2ctr_x'].tolist()
    list_dir2ctr_y = df_batch_pairs['dir2ctr_y'].tolist()

    return (
        list_bsn, list_total_matches, 
        list_mean_d_x, list_mean_d_y, list_mean_d_l, list_d_dir_degree, 
        list_mean_grad_x, list_mean_grad_y, list_mean_grad_l,
        list_grad_dir_degree, list_dist_to_gnd, 
        list_ctr_rot_x, list_ctr_rot_y, list_rot_degree, list_rot_ang_dir,
        list_dir2ctr_degree, list_dir2ctr_x, list_dir2ctr_y
    )


def truncate_df_at_first_all_nan(df):
    print('\n[truncate_df_at_first_all_nan]')
    #[1] Identify rows with all NaN values
    all_nan_rows = df[df.isnull().all(axis=1)]
    
    #[2] Check if there are any all-NaN rows
    if not all_nan_rows.empty:
        #[3] Get the index of the first all-NaN row
        idx_first_all_nan = all_nan_rows.index[0]
        
        #[4] Truncate the DataFrame up to that row
        df_truncated = df.loc[:idx_first_all_nan-1]
    else:
        #[5] If there are no all-NaN rows, return the original DataFrame
        df_truncated = df.copy()
    
    return df_truncated

#[5] Dataframe - End










#[6] Excel and xlsx - Start


def read_excel_with_multiple_sheets_into_list_of_dfs(path_xlsx):
    print('\n[read_excel_with_multiple_sheets_into_list_of_dfs]')
    #[1] Read all sheets into a dictionary of dataframes
    dfs = pd.read_excel(path_xlsx, sheet_name=None)
    
    #[2] Separate the dictionary into a list of dataframes and a list of sheet names
    list_of_dfs = list(dfs.values())
    list_of_sheetnames = list(dfs.keys())
    
    #[2] Optionally print the headers of the first few rows of each dataframe
    for sheet_name, df in dfs.items():
        print(f'--[dataframe headers for {sheet_name}]', df.columns.tolist())
    print()
    return list_of_dfs, list_of_sheetnames


def save_df_as_excel_add(df, dir_excel, basename_excel, sheet_name_or_index):
    print('\n[save_df_as_excel_add]')

    #[1] Convert sheet index to string name if an integer is provided
    if isinstance(sheet_name_or_index, int):
        sheet_name = f'Sheet{sheet_name_or_index}'
    else:
        sheet_name = str(sheet_name_or_index)  # Ensure sheet_name is always a string

    #[2] Ensure the directory exists
    if not os.path.exists(dir_excel):
        os.makedirs(dir_excel)
    path_excel = os.path.join(dir_excel, basename_excel)

    #[3] Try to save the DataFrame and handle potential permission errors
    last_sheet_name = sheet_name  # Initialize last_sheet_name with the intended sheet name
    try:
        if os.path.isfile(path_excel):
            book = openpyxl.load_workbook(path_excel)
            if sheet_name in book.sheetnames:
                # Increment sheet name if it exists to avoid overwriting
                original_sheet_name = sheet_name
                suffix = 1
                while sheet_name in book.sheetnames:
                    sheet_name = f"{original_sheet_name}{suffix}"
                    suffix += 1
                last_sheet_name = sheet_name  # Update last_sheet_name with the new sheet name
            book.close()  # Close the workbook
            
            # Proceed to save data with possibly updated sheet name
            with pd.ExcelWriter(path_excel, engine='openpyxl', mode='a', if_sheet_exists='new') as writer:
                df.to_excel(writer, sheet_name=last_sheet_name, index=True)
        else:  # If file doesn't exist, create a new one
            with pd.ExcelWriter(path_excel, engine='openpyxl', mode='w') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=True)

    except PermissionError as e:
        print(f'Error: {e}\nThe file "{path_excel}" is open in another application.')
        input("Please close the file and press Enter to try again...")
        return save_df_as_excel_add(df, dir_excel, basename_excel, sheet_name_or_index)  # Recursive call to try again

    #[4] Return the last used sheet name
    print(f'--Dataframe saved to {path_excel}\n-- in sheet {last_sheet_name}', end='\n\n')
    return last_sheet_name




def add_dictionary_to_excel(dic, path_xlsx, sheet_name, cell_location='B20'):
    print('\n[add_dictionary_to_excel]')
    #[1] Load the workbook
    wb = openpyxl.load_workbook(path_xlsx)
    
    #[2] Select the worksheet
    sheet = wb[sheet_name]
    
    #[3] Calculate the column letter for values before entering the loop
    key_column = cell_location[0]
    value_column = chr(ord(key_column) + 1)  # Column immediately to the right of the key column
    
    #[4] Loop through the dictionary items and add them to the specified cell location
    for key, value in dic.items():
        #[5] Write the key in the specified cell
        key_cell = sheet[cell_location]
        key_cell.value = key

        #[6] Write the value in the next column to the right
        value_cell = sheet[value_column + cell_location[1:]]
        value_cell.value = value

        #[7] Move to the next row, updating the row number while keeping the column the same
        row_number = int(cell_location[1:]) + 1
        cell_location = key_column + str(row_number)
    
    #[9] Save the workbook
    wb.save(path_xlsx)
    
    



def adjust_multiple_column_widths(path_excel, sheet_name, list_column_letters, list_column_widths):
    print('\n[adjust_multiple_column_widths]')
    if len(list_column_letters) != len(list_column_widths) or type(list_column_letters) != list or type(list_column_widths) != list:
        raise TypeError('The length of list_column_letters and list_column_widths are different, or input is not list, please check.')
    workbook = openpyxl.load_workbook(path_excel)
    if sheet_name not in workbook.sheetnames:
        print(f"--Sheet '{sheet_name}' does not exist in the workbook.")
        return
    worksheet = workbook[sheet_name]
    if len(list_column_letters) != len(list_column_widths):
        print("--The lengths of the column letters and column widths lists do not match.")
        return
    for col_letter, col_width in zip(list_column_letters, list_column_widths):
        worksheet.column_dimensions[col_letter].width = col_width
        print(f"--Column '{col_letter}' in sheet '{sheet_name}' has been adjusted to width {col_width}.")
    workbook.save(path_excel)
    print("--All specified columns have been adjusted.", end='\n\n')
    

def return_name_of_last_excel_sheet(path_xlsx):
    # Load the workbook from the specified path
    workbook = openpyxl.load_workbook(path_xlsx)
    # Get all sheet names
    sheet_names = workbook.sheetnames
    # Return the name of the last sheet
    last_sheet_name = sheet_names[-1]
    return last_sheet_name



def read_specific_columns_in_excel_sheet_into_dataframe(path_xlsx, sheet_name, col_start, col_end, max_row):
    """
    Read specific columns from an Excel sheet into a pandas DataFrame.
    Parameters:
        path_xlsx (str): Path to the Excel file.
        sheet_name (str): Name of the sheet within the Excel file.
        col_start (str): Starting column (e.g., 'A', 'B', 'C', etc.).
        col_end (str): Ending column (e.g., 'A', 'B', 'C', etc.).
        max_row (int): Maximum row to read.
    Returns:
        DataFrame: Pandas DataFrame containing the specified columns.
    """
    #[1] Check if col_start < col_end
    if col_start >= col_end:
        raise ValueError("col_start must be less than col_end.")
    
    #[2] Load excel workbook and sheet
    wb = openpyxl.load_workbook(path_xlsx, data_only=True)
    sheet = wb[sheet_name]
    
    #[3] Initialize an empty DataFrame
    df = pd.DataFrame()
    
    #[4] Extract column headers from the first row
    headers = []
    for col in range(ord(col_start), ord(col_end) + 1):
        col_letter = chr(col)
        cell_value = sheet[col_letter + '1'].value
        headers.append(cell_value)
    
    #[5] Iterate over the specified range of columns
    for col, header in zip(range(ord(col_start), ord(col_end) + 1), headers):
        col_letter = chr(col)
        col_data = []
        #[6] Iterate over rows until a None value is encountered
        for row in range(2, max_row + 1):  # Start from the second row since the first row is headers
            cell_value = sheet[col_letter + str(row)].value
            if cell_value is None:
                break
            col_data.append(cell_value)
        df[header] = col_data
    
    return df



#[6] Excel and xlsx - End













