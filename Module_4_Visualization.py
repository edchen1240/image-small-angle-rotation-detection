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
import os, sys, math
import numpy as np
import pandas as pd
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy.optimize import curve_fit

from PIL import Image
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import Normalize



#[1] Define the quadratic function
def quadratic(x, a, b, c):
        return a * x**2 + b * x + c

def two_lists_find_quadratic_extrema(list_x, list_y):
    
    #[2] Convert lists to numpy arrays
    arr_x = np.array(list_x)
    arr_y = np.array(list_y)
    
    #[3] Fit the quadratic model to the data
    params, _ = curve_fit(quadratic, arr_x, arr_y)
    a, b, c = params
    
    #[4] Calculate R-squared value
    y_pred = quadratic(arr_x, a, b, c)
    ss_res = np.sum((arr_y - y_pred) ** 2)
    ss_tot = np.sum((arr_y - np.mean(arr_y)) ** 2)
    rsq = 1 - (ss_res / ss_tot)
    
    #[5] Calculate extrema
    x_ext = - b / ( 2 * a )
    y_ext = - (b**2 - 4*a*c)/(4*a)
    
    return a, b, c, rsq, x_ext, y_ext
    
    
    


def find_location_with_min_loss_eff(list_x, list_y, list_z, list_l, 
                                    text_title, list_text_axis_names, 
                                    path_save):
    #[1] Settings
    fs_title = 16
    fs_axis = 14
    fit_pos_x = 0.05
    fit_pos_y, dy = 0.95, -0.05
    window_range_ext = 0.2
    ylim_adj = 1.5
    
    #[2] Collect into dataframe    
    df_xyzl = pd.DataFrame({
                    'list_x': list_x,
                    'list_y': list_y,
                    'list_z': list_z,
                    'list_l': list_l})
    
    #[3] Convert to arrays
    arr_x, arr_y, arr_z, arr_l = np.array(list_x), np.array(list_y), np.array(list_z), np.array(list_l)
    ylim_max = np.max(arr_l) * ylim_adj
    
    
    #[4] Fit quadratic between x and le.
    x_a, x_b, x_c, x_rsq, x_x_ext, x_le_ext = two_lists_find_quadratic_extrema(arr_x, arr_l)
    max_x = max(np.max(arr_x), x_x_ext)
    min_x = min(np.min(arr_x), x_x_ext)
    ext = window_range_ext*(max_x - min_x)
    arr_x_fit = np.linspace(min_x - ext, max_x + ext, 500)
    
    #[5] Fit quadratic between y and le.
    y_a, y_b, y_c, y_rsq, y_x_ext, y_le_ext = two_lists_find_quadratic_extrema(arr_y, arr_l)
    max_y = max(np.max(arr_y), y_x_ext)
    min_y = min(np.min(arr_y), y_x_ext)
    ext = window_range_ext*(max_y - min_y)
    arr_y_fit = np.linspace(min_y - ext, max_y + ext, 500)
    
    #[6] Fit quadratic between z and le.
    z_a, z_b, z_c, z_rsq, z_x_ext, z_le_ext = two_lists_find_quadratic_extrema(arr_z, arr_l)
    max_z = max(np.max(arr_z), z_x_ext)
    min_z = min(np.min(arr_z), z_x_ext)
    ext = window_range_ext*(max_z - min_z)
    arr_z_fit = np.linspace(min_z - ext, max_z + ext, 500)
    
    #[10] Create figure and 3D axis
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    #[12] Use the sequence of list_l as epoch, and normalize from 0 to 1 for viridis color bar.
    arr_epoch = np.arange(len(arr_l))
    norm = Normalize(vmin=arr_epoch.min(), vmax=arr_epoch.max())
    colors = [plt.cm.viridis(norm(epoch)) for epoch in arr_epoch]
    
    
    #[15] Plot for x vs l
    axes[0].scatter(arr_x, arr_l, c=colors, label='Data points')
    arr_x_yfit = x_a * arr_x_fit**2 + x_b * arr_x_fit + x_c
    axes[0].plot(arr_x_fit, arr_x_yfit, color='lightgray', label='Quadratic fit')
    axes[0].scatter([x_x_ext], [x_le_ext], marker='*', s=300, color='red', alpha=0.5, label='Extrema')
    axes[0].set_title(f"{list_text_axis_names[0]} vs {list_text_axis_names[3]}")
    axes[0].set_xlabel(list_text_axis_names[0], fontsize = fs_axis)
    axes[0].set_ylabel(list_text_axis_names[3], fontsize = fs_axis)
    axes[0].set_ylim(0, ylim_max)
    axes[0].legend(loc='upper right')
    
    #[16] Adding equation, R-squared, and extrema info to the plot
    equation_text = f"y = {x_a:.2e}x^2 + {x_b:.2e}x + {x_c:.2e}"
    rsq_text = f"$R^2$ = {x_rsq:.2f}"
    extrema_text = f"Extrema: ({x_x_ext:.2f}, {x_le_ext:.2f})"
    axes[0].text(fit_pos_x, fit_pos_y, equation_text, transform=axes[0].transAxes, fontsize=10, verticalalignment='top')
    axes[0].text(fit_pos_x, fit_pos_y + dy, rsq_text, transform=axes[0].transAxes, fontsize=10, verticalalignment='top')
    axes[0].text(fit_pos_x, fit_pos_y + 2 * dy, extrema_text, transform=axes[0].transAxes, fontsize=10, verticalalignment='top')

    
    #[20] Plot for y vs l
    axes[1].scatter(arr_y, arr_l, c=colors, label='Data points')
    arr_y_yfit = y_a * arr_y_fit**2 + y_b * arr_y_fit + y_c
    axes[1].plot(arr_y_fit, arr_y_yfit, color='lightgray', label='Quadratic fit')
    axes[1].scatter([y_x_ext], [y_le_ext], marker='*', s=300, color='red', alpha=0.5, label='Extrema')
    axes[1].set_title(f"{list_text_axis_names[1]} vs {list_text_axis_names[3]}")
    axes[1].set_xlabel(list_text_axis_names[1], fontsize = fs_axis)
    axes[1].set_ylabel(list_text_axis_names[3], fontsize = fs_axis)
    axes[1].set_ylim(0, ylim_max)
    axes[1].legend(loc='upper right')
    
    #[21] Adding equation, R-squared, and extrema info to the plot
    equation_text = f"y = {y_a:.2e}x^2 + {y_b:.2e}x + {y_c:.2e}"
    rsq_text = f"$R^2$ = {y_rsq:.2f}"
    extrema_text = f"Extrema: ({y_x_ext:.2f}, {y_le_ext:.2f})"
    axes[1].text(fit_pos_x, fit_pos_y, equation_text, transform=axes[1].transAxes, fontsize=10, verticalalignment='top')
    axes[1].text(fit_pos_x, fit_pos_y + dy, rsq_text, transform=axes[1].transAxes, fontsize=10, verticalalignment='top')
    axes[1].text(fit_pos_x, fit_pos_y + 2 * dy, extrema_text, transform=axes[1].transAxes, fontsize=10, verticalalignment='top')
    
    
    #[25] Plot for z vs l
    axes[2].scatter(arr_z, arr_l, c=colors, label='Data points')
    arr_z_yfit = z_a * arr_z_fit**2 + z_b * arr_z_fit + z_c
    axes[2].plot(arr_z_fit, arr_z_yfit, color='lightgray', label='Quadratic fit')
    axes[2].scatter([z_x_ext], [z_le_ext], marker='*', s=300, color='red', alpha=0.5, label='Extrema')
    axes[2].set_title(f"{list_text_axis_names[2]} vs {list_text_axis_names[3]}")
    axes[2].set_xlabel(list_text_axis_names[2], fontsize = fs_axis)
    axes[2].set_ylabel(list_text_axis_names[3], fontsize = fs_axis)
    axes[2].set_ylim(0, ylim_max)
    axes[2].legend(loc='upper right')
    
    #[26] Adding equation, R-squared, and extrema info to the plot
    equation_text = f"y = {z_a:.2e}x^2 + {z_b:.2e}x + {z_c:.2e}"
    rsq_text = f"$R^2$ = {z_rsq:.2f}"
    extrema_text = f"Extrema: ({z_x_ext:.2f}, {z_le_ext:.2f})"
    axes[2].text(fit_pos_x, fit_pos_y, equation_text, transform=axes[2].transAxes, fontsize=10, verticalalignment='top')
    axes[2].text(fit_pos_x, fit_pos_y + dy, rsq_text, transform=axes[2].transAxes, fontsize=10, verticalalignment='top')
    axes[2].text(fit_pos_x, fit_pos_y + 2 * dy, extrema_text, transform=axes[2].transAxes, fontsize=10, verticalalignment='top')
   
    
    #[30] Save plot
    fig.suptitle(text_title, fontsize = fs_title)
    plt.tight_layout()
    plt.savefig(path_save)
    plt.close()
    
    return df_xyzl, (x_x_ext, y_x_ext, z_x_ext)
        
    
    
    
    
    
















#[5] Visulization - Start
def plot_2loss_and_3learning_rate_record(list_loss, list_loss_eff, 
                                         list_ss_x, list_ss_y, list_ss_r, rot_lr_constraint, 
                                         iter_bsn, text_note, path_save):
    epochs = range(1, len(list_loss) + 1)  # Assuming epochs start from 1 to the length of list_loss
    fig, ax1 = plt.subplots(figsize=(8, 8))

    #[1] Plot Loss and Effective Loss on primary axis
    fs_axis = 16
    fs_title = 20
    fs_info = 12
    ax1.set_xlabel('Epoch', fontsize = fs_axis)
    ax1.set_ylabel('Loss', fontsize = fs_axis)  # Normal axis color for loss
    
    #[2] Calculate scaling factor for effective loss
    mean_list_loss = np.mean(list_loss)
    mean_list_loss_eff = np.mean(list_loss_eff)
    ratio = mean_list_loss_eff / mean_list_loss  # Reversed ratio to scale list_loss
    mag_diff_zeros = (int(math.log10(ratio)) if ratio != 0 else 0) -2  # Check for ratio being zero
    leading_nums = math.floor(ratio / (10 ** mag_diff_zeros)) - 5 # Two digits
    loss_mtp = leading_nums * (10 ** (mag_diff_zeros))
    text_print = f'\n\n[ratio] {ratio}\n'\
                    f'[mag_diff_zeros] {mag_diff_zeros}\n'\
                    f'[leading_nums] {leading_nums}\n'\
                    f'[loss_mtp] {loss_mtp}\n\n'
    print(text_print)
    list_loss_adj = [x * loss_mtp for x in list_loss]  # Apply scaling to list_loss

    #[3] Plot Loss and Effective Loss on primary axis
    l1 = ax1.plot(epochs, list_loss_adj, 'turquoise', label=f'Loss x{loss_mtp:.1e}', linewidth=3)  # Scaled Loss curve
    l2 = ax1.plot(epochs, list_loss_eff, 'orange', label='Effective Loss', linewidth=3) 

    #[4] Set ylim for ax1
    list_loss_max, list_loss_min = max(max(list_loss_adj), max(list_loss_eff)), min(min(list_loss_adj), min(list_loss_eff))
    ax1.set_ylim(list_loss_min, list_loss_max * 1.5)
    ax1.tick_params(axis='y')
    ax1.set_title(f'Training Loss and Learning Rate of {iter_bsn}', fontsize = fs_title)

    #[6] Plot Learning Rate on secondary axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Learning Rate', fontsize = fs_axis) 
    list_ss_r = (np.array(list_ss_r)/rot_lr_constraint).tolist()
    l3 = ax2.plot(epochs, list_ss_x, color=[140/255, 220/255, 80/255], label='Step size of x', linestyle='-', linewidth=2)
    l4 = ax2.plot(epochs, list_ss_y, color=[100/255, 180/255, 120/255], label='Step size of y', linestyle='--', linewidth=2)
    l5 = ax2.plot(epochs, list_ss_r, color=[60/255, 140/255, 160/255], label=f'Step size of rot (x{1/rot_lr_constraint:.1e})', linestyle=':', linewidth=2)


    #[7] Set ylim for ax2
    list_ss_x_max, list_ss_x_min = max(list_ss_x), min(list_ss_x)
    list_ss_y_max, list_ss_y_min = max(list_ss_y), min(list_ss_y)
    list_ss_r_max, list_ss_r_min = max(list_ss_r), min(list_ss_r)

    #[8] Calculate overall maximum and minimum
    list_ss_max = max(list_ss_x_max, list_ss_y_max, list_ss_r_max)
    list_ss_min = min(list_ss_x_min, list_ss_y_min, list_ss_r_min)
    ax2.set_ylim(list_ss_min, list_ss_max * 1.5)
    ax2.tick_params(axis='y')

    #[9] Combine legends from both axes
    handles, labels = [], []
    for l in [l1, l2, l3, l4, l5]:
        handles.extend(l)
        labels.extend([h.get_label() for h in l])
    ax1.legend(handles, labels, loc='upper right')

    #[15] [AddText] Adding the additional text.
    plt.subplots_adjust(left=0.1, right=0.90, bottom=0.35, top=0.9)
    plt.figtext(0.05, 0.27, text_note, fontsize=fs_info, va="top", ha="left", fontfamily='Consolas')

    #[16] Save the figure
    plt.savefig(path_save)







def plot_2D_optimization_process_xyLoss_separate(list_x, title_x, 
                                                 list_y, title_y, 
                                                 list_z, title_z, 
                                                 list_loss, title_loss, 
                                                 GT_xyrl, list_marker_idx, title_plot, text_note, path_save):
    """
    1. Generate 2 subplots arranged as 1 by 2, figsize=(10, 6).
    2. The upper left plot has list_x in X-axis and list_z in Y-axis. 
        Use color_epoch_viridis in the code as a color indicator of epoch.
        Use size_norm_loss in the code as a size indicator of loss.
        Mark GT (ground truth) at (GT_x, GT_y) as "ax.scatter(xs[0], ys[0], marker='+', s=400, color='chartreuse')"
        Mark the starting point with "ax.scatter(xs[list_marker_idx[0]], ys[list_marker_idx[0]], marker='+', s=300, color='dodgerblue')"
        Mark the optimized point with "ax.scatter(xs[list_marker_idx[-1]], ys[list_marker_idx[-1]], marker='*', s=300, color='red', alpha=0.5)"
    3. The upper center plot has list_y in X-axis and list_loss in Y-axis, the rest are the same.
    4. The upper right plot has list_z in X-axis and list_loss in Y-axis, the rest are the same.
    5. The lower part has no plot. Put text_note over there.
    6. Keep the comment numbering style.
    """
    flip_y_axis = False
    
    #[1] Create figure and subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    fig.subplots_adjust(hspace=0.4, wspace=0.22, top=0.90, bottom=0.38, left=0.08, right=0.87)
    fs_title = 16
    fs_axis = 10
    fs_info = 10
    lim_k = 1

    #[2] Convert lists to numpy array for easier manipulation
    xs = np.array(list_x)
    ys = np.array(list_y)
    zs = np.array(list_z)
    ls = np.array(list_loss)
    if GT_xyrl:
        GT_x, GT_y, GT_z, GT_l = GT_xyrl

    #[3] Normalize epoch from 0 to 1 for viridis color bar.
    arr_epoch = np.arange(len(ls))
    norm = Normalize(vmin=arr_epoch.min(), vmax=arr_epoch.max())
    color_epoch_viridis = [plt.cm.viridis(norm(epoch)) for epoch in arr_epoch]

    #[6] Normalize loss values for size scaling
    min_loss, max_loss = ls.min(), ls.max()
    if GT_xyrl:
        min_loss = np.min([min_loss, GT_l])
        size_norm_GT_l = (GT_l - min_loss) / (max_loss - min_loss) * 200 + 50
    size_norm_loss = (ls - min_loss) / (max_loss - min_loss) * 200 + 50
    
    

    plots= [(xs, title_x, axs[0]),
            (ys, title_y, axs[1])]
    
    for data, title, ax in plots:
        # [7-1] Plot each point with size corresponding to its loss value.
        sc = ax.scatter(data, zs, s=size_norm_loss, c=color_epoch_viridis, alpha=0.7)
        # [7-2] Mark the first and last points with special markers.
        if GT_xyrl:
            GT_data = GT_x if data is xs else GT_y if data is ys else GT_z
            ax.scatter(GT_data, GT_z, marker='+', s=400, color='chartreuse')  # GT
            ax.scatter(GT_data, GT_z, s=size_norm_GT_l, c='chartreuse', alpha=0.7)
        # [7-3] Add other special markers list_marker_idx = [1st_end, 2nd_start, opt]
        ax.scatter(data[list_marker_idx[0]], zs[list_marker_idx[0]], marker='+', s=400, color='dodgerblue')  # Start
        ax.scatter(data[list_marker_idx[-1]], zs[list_marker_idx[-1]], marker='*', s=300, color='red', alpha=0.5)  # Opt
                    
        # [12] Labels and title.
        ax.set_xlabel(title, fontsize=fs_axis)
        ax.set_ylabel(title_z, fontsize=fs_axis)
        
        # [13] Axes appearance adjustments.
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        #[14] Flip the y-axis to simulate raster coordinates.
        if flip_y_axis:
            ax.set_ylim(ax.get_ylim()[::-1])
            
        #[15] Ensure the subplot is square
        #ax.set_aspect('equal', 'box')
    
    # [18] Adding the additional text to the fourth subplot.
    plt.figtext(0.05, 0.30, text_note, fontsize=fs_info, va="top", ha="left", fontfamily="monospace")
    
    # [19] Adding color bar to indicate the scale of loss.
    cbar_ax = fig.add_axes([0.90, 0.10, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = plt.colorbar(sc, cax=cbar_ax)
    cbar.set_label('Epoch')
    
    # [20] Add a large title for the whole plot.
    fig.suptitle(title_plot, fontsize=fs_title, y=0.95)

    # [25] Save or show plot.
    if path_save:
        plt.savefig(path_save)
        print(f'--Plot saved at {path_save}')
    else:
        plt.show()

    return fig, axs






def plot_3D_optimization_process(list_x, title_x, 
                                list_y, title_y, 
                                list_z, title_z,
                                list_loss, title_loss, 
                                GT_xyrl, list_marker_idx, text_title, text_note, path_save):
    """
    Plot control rotation points in 3D space. The Z-axis represents the cone slope.
    Mark the epoch (extract from the sequence of list_loss) in color gradient of points, using viridis (blue - green - yellow).
    The value of list_loss is used for the size of the points, where larger values have larger sizes.
    """
    #[1] Create figure and 3D axis
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    fs_axis = 14
    fs_title = 20
    fs_info = 12
    
    # [2] Convert lists to numpy array for easier manipulation
    xs = np.array(list_x)
    ys = np.array(list_y)
    zs = np.array(list_z)
    ls = np.array(list_loss)
    if GT_xyrl:
        GT_x, GT_y, GT_z, GT_l = GT_xyrl
    
    #[3] Normalize epoch from 0 to 1 for viridis color bar.
    arr_loss = np.array(list_loss)
    arr_epoch = np.arange(len(arr_loss))
    norm = Normalize(vmin=arr_epoch.min(), vmax=arr_epoch.max())
    color_epoch_viridis = [plt.cm.viridis(norm(epoch)) for epoch in arr_epoch]

    #[6] Normalize loss values for size scaling
    min_loss, max_loss = arr_loss.min(), arr_loss.max()
    if GT_xyrl:
        min_loss = np.min([min_loss, GT_l])
        size_norm_GT_l = (GT_l - min_loss) / (max_loss - min_loss) * 200 + 50
    size_norm_loss = (arr_loss - min_loss) / (max_loss - min_loss) * 200 + 50
    
    #[7-1] Plot each point with size corresponding to its loss value.
    sc = ax.scatter(xs, ys, zs, s=size_norm_loss, c=color_epoch_viridis, alpha=0.7)
    #[7-2] Mark the first and last points with special markers.
    if GT_xyrl:
        ax.scatter(GT_x, GT_y, GT_z, marker='+', s=400, color='chartreuse')  # GT
        ax.scatter(GT_x, GT_y, GT_z, s=size_norm_GT_l, c='chartreuse', alpha=0.7)
    #[7-3] Add other special markers list_marker_idx = [1st_end, 2nd_start, opt]
    ax.scatter(xs[list_marker_idx[0]], ys[list_marker_idx[0]], zs[list_marker_idx[0]], marker='+', s=400, color='dodgerblue')        # Start
    ax.scatter(xs[list_marker_idx[-1]], ys[list_marker_idx[-1]], zs[list_marker_idx[-1]], marker='*', s=300, color='red', alpha=0.5)        # Opt
    
    #[8] Determine the range for the axes
    range_x = np.max(xs) - np.min(xs)
    range_y = np.max(ys) - np.min(ys)
    range_for_xy = max(range_x, range_y) * 2  # Need to be larger. 1.5 will Missed the first point.
    xlim_min, xlim_max = np.mean(xs) - 0.5 * range_for_xy, np.mean(xs) + 0.5 * range_for_xy
    ylim_min, ylim_max = np.mean(ys) - 0.5 * range_for_xy, np.mean(ys) + 0.5 * range_for_xy
    
    #[9] Use the broader of the ranges to set both axes
    ax.set_xlim(xlim_min, xlim_max)
    ax.set_ylim(ylim_min, ylim_max)

    #[12] Labels and title.
    ax.set_xlabel(title_x, fontsize = fs_axis)
    ax.set_ylabel(title_y, fontsize = fs_axis)
    ax.set_zlabel(title_z, fontsize = fs_axis)
    ax.set_title(text_title, fontsize = fs_title)
    
    #[13] Axes appearance adjustments.
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.zaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.ticklabel_format(style='sci', axis='z', scilimits=(0,0))
    
    #[14] Flip the y-axis to simulate raster coordinates.
    ax.set_ylim(ax.get_ylim()[::-1])
    
    #[15] [AddText] Adding the additional text.
    plt.subplots_adjust(bottom=0.30, right=0.85)
    plt.figtext(0.05, 0.27, text_note, fontsize=fs_info, va="top", ha="left", fontfamily="monospace")

    
    #[16] Adding color bar to indicate the scale of loss.
    cbar_ax = fig.add_axes([0.90, 0.25, 0.02, 0.7])  # [left, bottom, width, top]
    cbar = plt.colorbar(sc, cax=cbar_ax, aspect=10, norm=norm)
    cbar.set_label('Epoch')

    #[17] Save or show plot.
    if path_save:
        plt.savefig(path_save)
        print(f'--Plot saved at {path_save}')
    else:
        plt.show()

    return fig, ax






def generate_circular_view_init(elve_azim_center, elve_azim_start, cnt_point, clockwise=True):
    """
    Default: elev=30 and azim=-60
    elve_start -> elve_min -> elve_center -> elve_max -> elve_start
    azim_start -> azim_center -> azim_max -> azim_center -> azim_start
    
    elve_azim_center = (30, 40) # Center elevation and azimuth
    elve_azim_start = (30, 70) # Start elevation and azimuth
    
    """
    elve_center, azim_center = elve_azim_center
    elve_start, azim_start = elve_azim_start
    #[1] Calculate the angular distances
    delta_elve = elve_start - elve_center
    delta_azim = azim_start - azim_center
    #[2] Determine the direction of rotation
    direction = 1 if clockwise else -1
    #[3] Calculate the step size for a full circle
    angle_step = 360 / cnt_point
    view_init_list = []
    for i in range(cnt_point):
        #[4] Calculate the new angle
        angle = i * angle_step * direction
        rad_angle = np.radians(angle)
        #[5] Calculate the new positions
        elve = elve_center + delta_elve * np.cos(rad_angle) - delta_azim * np.sin(rad_angle)
        azim = azim_center + delta_elve * np.sin(rad_angle) + delta_azim * np.cos(rad_angle)
        view_init_list.append([int(elve), int(azim)])
    view_init_list = view_init_list[:-1]  # Remove the last viewpoint that duplicate with the first viewpoint.
    return view_init_list




def convert_plot_to_gif(fig, ax, list_view_init, path_save, duration=100):
    """
    Converts a 3D Matplotlib plot to a GIF by rotating the plot using predefined view angles.
    Parameters:
    - fig: Matplotlib figure object.
    - ax: Matplotlib 3D axis object.
    - list_view_init: List of tuples/lists containing (elevation, azimuth) angles.
    - filename: Name of the output GIF file.
    - duration: Duration of each frame in the GIF in milliseconds.
    """
    def update_view(angle):
        ax.view_init(elev=angle[0], azim=angle[1])
        return fig,
    #[2] Create an animation
    anim = FuncAnimation(fig, update_view, frames=list_view_init, blit=False, repeat=True)
    #[3] Save animation as GIF
    anim.save(path_save, writer='pillow', fps=1000/duration)



def dir_of_images_into_GIF(directory, fps=10, output_filename='output.gif', bounce=True, compress_ratio=1, rotate=0):
    #[1] Collect all images in the directory and sort them alphabetically
    images = sorted([img for img in os.listdir(directory) if img.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    #[2] Create a list to hold the image data
    image_data = []
    
    #[3] Load each image, apply compression if needed, and apply rotation
    for image in images:
        image_path = os.path.join(directory, image)
        img = Image.open(image_path)
        
        if compress_ratio != 1:
            # Calculate new dimensions
            new_width = int(img.width * compress_ratio)
            new_height = int(img.height * compress_ratio)
            img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Apply rotation
        if rotate in [0, 90, 180, 270]:
            img = img.rotate(rotate, expand=True)
        
        img_array = np.array(img)  # Convert Pillow image to a NumPy array
        image_data.append(img_array)
    
    #[4] If bounce is true, reverse the image list (excluding the first and last image) and append it to the original list
    if bounce:
        reversed_data = image_data[-2:0:-1]  # Reverse the list but exclude the first and last items to avoid repetition at the ends
        image_data.extend(reversed_data)

    #[5] Create GIF
    path_gif = os.path.join(directory, output_filename)
    imageio.mimsave(path_gif, image_data, format='GIF', fps=fps, loop=0)
    
    return path_gif

#[5] Visulization - End