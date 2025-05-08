import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import binned_statistic_2d
from utils.process_data import filter_outliers


def descriptive_statistics(data, column, title="Statistics"):
    print(f"{title} for {column}:")
    print(f"  Mean: {round(data[column].mean(), 3)}")
    print(f"  Median: {round(data[column].median(), 3)}")
    print(f"  Standard Deviation: {round(data[column].std(), 3)}")
    print(f"  Skewness: {round(data[column].skew(), 3)}")
    print(f"  Kurtosis: {round(data[column].kurtosis(), 3)}")


def plot_histogram(dataframe, column, filter_column=None, bins=50, log_scale=False, 
                   remove_outliers=False, restricted=False, dpi=100):
    if remove_outliers:
        filtered_data = filter_outliers(dataframe, [filter_column])
    else:
        filtered_data = dataframe

    data_min = filtered_data[column].min()
    data_max = filtered_data[column].max()

    if data_min < 0 < data_max:
        max_abs = max(abs(data_min), abs(data_max))
        bin_width = (2 * max_abs) / bins
        bin_edges = np.arange(-max_abs, max_abs + bin_width, bin_width)
    else:
        bin_edges = np.linspace(data_min, data_max, bins)
    
    hist, edges = np.histogram(filtered_data[column], bins=bin_edges)

    non_empty_bins = hist > 0
    hist = hist[non_empty_bins]
    bin_edges = edges[:-1][non_empty_bins]

    middle_common = []
    for i in range(1, len(edges) - 1):
        if edges[i - 1] in bin_edges and edges[i + 1] in bin_edges:
            middle_common.append(edges[i])
    
    tick_pos = np.unique(sorted(middle_common + list(bin_edges)))

    plt.figure(figsize=(15, 6), dpi=dpi)
    plt.hist(filtered_data[column], bins=tick_pos, edgecolor='black')
    plt.xticks(ticks=tick_pos[::2], rotation=90, fontsize=13)
    plt.yticks(fontsize=13)

    if log_scale:
        plt.yscale('log')
        plt.ylabel('Count', fontsize=16)
        if restricted:
            plt.title(f'Restricted Data Histogram of {column} (Bins: {len(bin_edges)}) (Log Scale)', fontsize=20)
        else:
            plt.title(f'Full Data Histogram of {column} (Bins: {len(bin_edges)}) (Log Scale)', fontsize=20)

    else:
        plt.ylabel('Count', fontsize=16)
        if restricted:
            plt.title(f'Restricted Data Histogram of {column} (Bins: {len(bin_edges)}) (Linear Scale)', fontsize=20)
        else:
            plt.title(f'Full Data Histogram of {column} (Bins: {len(bin_edges)}) (Linear Scale)', fontsize=20)
    
    plt.show()


def plot_time_series(dataframe, column, dpi=100, filter_column=None, remove_outliers=False, resample_method=None):
    if remove_outliers:
        filtered_data = filter_outliers(dataframe, [filter_column])
    else:
        filtered_data = dataframe
    
    filtered_data.index = filtered_data['Time']

    if resample_method:
        resampled_df = filtered_data.resample(resample_method).mean()
    else:
        resampled_df = filtered_data

    plt.figure(figsize=(15, 6), dpi=dpi)
    plt.plot(resampled_df.index, resampled_df[column])

    plt.title(f'{column} vs Time')
    plt.xlabel('Time')
    plt.ylabel(f'{column}')
    plt.xticks(rotation=45)
    plt.show()


def plot_2d_histogram(dataframe, q1, q2, xlab, ylab, remove_outliers=False, bins=30,
                      dpi=100, cmap='plasma', cmin=0, xlim=None, ylim=None, log_scale=False,
                      colorbar_round_decimals=0):
    filtered_matrix = dataframe[[q1, q2]].dropna()

    if xlim:
        filtered_matrix = filtered_matrix[(filtered_matrix[q1] >= xlim[0]) & (filtered_matrix[q1] <= xlim[1])]
    if ylim:
        filtered_matrix = filtered_matrix[(filtered_matrix[q2] >= ylim[0]) & (filtered_matrix[q2] <= ylim[1])]

    x_min, x_max = min(filtered_matrix[q1].min(), 0), max(filtered_matrix[q1].max(), 0)
    y_min, y_max = min(filtered_matrix[q2].min(), 0), max(filtered_matrix[q2].max(), 0)
    
    xedges = np.linspace(x_min, x_max, bins + 1)
    yedges = np.linspace(y_min, y_max, bins + 1)

    if 0 not in xedges:
        xedges = np.insert(xedges, 0, 0)
    if 0 not in yedges:
        yedges = np.insert(yedges, 0, 0)

    xedges = np.sort(xedges)
    yedges = np.sort(yedges)

    H, xedges, yedges = np.histogram2d(filtered_matrix[q1], filtered_matrix[q2],
                                       bins=[xedges, yedges])

    norm = LogNorm() if log_scale else None

    plt.figure(figsize=(10, 8), dpi=dpi)

    plt.hist2d(filtered_matrix[q1], filtered_matrix[q2],
               bins=[xedges, yedges], cmap=cmap, cmin=cmin, 
               linewidth=0.5, edgecolor='black', norm=norm)
    
    cbar = plt.colorbar()
    cbar.set_label(label='Number of Points', fontsize=16)
    
    if np.nanmax(H) > 1e6:
        cbar.formatter = plt.FuncFormatter(lambda x, _: f'{x:.{colorbar_round_decimals}e}')
    else:
        cbar.formatter = plt.FuncFormatter(lambda x, _: f'{x:.{colorbar_round_decimals}f}')

    plt.xlabel(f'{xlab}', fontsize=16)
    plt.ylabel(f'{ylab}', fontsize=16)
    plt.title(f'2D Heatmap: {xlab} vs {ylab}', fontsize=20)

    plt.xticks(xedges[::2], rotation=90, fontsize=13)
    plt.yticks(yedges[::2], fontsize=13)

    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    
    plt.show()


def plot_2d_avg_histogram(dataframe, q1, q2, q3, xlab, ylab, clab, dpi=100, 
                          cmap='plasma', bins=30, colorbar_round_decimals=2, 
                          log_scale=False, xlim=None, ylim=None):
    filtered_matrix = dataframe[[q1, q2, q3]].dropna()

    if xlim:
        filtered_matrix = filtered_matrix[(filtered_matrix[q1] >= xlim[0]) & (filtered_matrix[q1] <= xlim[1])]
    if ylim:
        filtered_matrix = filtered_matrix[(filtered_matrix[q2] >= ylim[0]) & (filtered_matrix[q2] <= ylim[1])]

    x_min, x_max = min(filtered_matrix[q1].min(), 0), max(filtered_matrix[q1].max(), 0)
    y_min, y_max = min(filtered_matrix[q2].min(), 0), max(filtered_matrix[q2].max(), 0)

    xedges = np.linspace(x_min, x_max, bins + 1)
    yedges = np.linspace(y_min, y_max, bins + 1)

    if 0 not in xedges:
        xedges = np.insert(xedges, 0, 0)
    if 0 not in yedges:
        yedges = np.insert(yedges, 0, 0)

    xedges = np.sort(xedges)
    yedges = np.sort(yedges)

    H, _, _, _ = binned_statistic_2d(
        filtered_matrix[q1], filtered_matrix[q2], filtered_matrix[q3],
        statistic='mean', bins=[xedges, yedges]
    )

    plt.figure(figsize=(10, 8), dpi=dpi)

    norm = LogNorm() if log_scale else None

    im = plt.pcolormesh(xedges, yedges, H.T, cmap=cmap, norm=norm, shading='auto', 
                        linewidth=0.5, edgecolors='black')
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    cbar = plt.colorbar(im)
    cbar.set_label(clab, fontsize=16)

    if np.nanmax(H) > 1e6:
        cbar.formatter = plt.FuncFormatter(lambda x, _: f'{x:.{colorbar_round_decimals}e}')
    else:
        cbar.formatter = plt.FuncFormatter(lambda x, _: f'{x:.{colorbar_round_decimals}f}')
    
    plt.xticks(xedges[::2], rotation=90, fontsize=13)
    plt.yticks(yedges[::2], fontsize=13)
        
    plt.xlabel(xlab, fontsize=16)
    plt.ylabel(ylab, fontsize=16)
    plt.xticks(fontsize=13, rotation=90)
    plt.yticks(fontsize=13)
    plt.title(f'{xlab} vs {ylab} ({clab})', fontsize=20)

    plt.show()
