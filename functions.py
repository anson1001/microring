# written on 7/1/2026

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import scipy.signal as signal
from scipy.optimize import curve_fit
from lmfit.models import LorentzianModel, LinearModel, ConstantModel   # or ConstantModel, QuadraticModel, etc.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

# ============================================================
# 1 Data Input Functions
# ============================================================

def read_data(channel_path, trigger_path, data_length=1562500):
    '''
    Input:  path of channel and trigger csv files
    Output: the sliced channel data within the sweeping laser
            shape: (DATA_LENGTH, ) presumably 1562500
    '''
    channel = pd.read_csv(channel_path, skiprows=15)
    channel.columns = ["Time", "Channel"]
    trigger = pd.read_csv(trigger_path, skiprows=15)
    trigger.columns = ["Time", "Channel"]
    # define the frame within the sweeping laser

    # take last trigger (sometimes there is a strange trigger before)
    start_index = np.where(trigger['Channel'] > 2.5)[0][-1]
    # start_index = np.where(trigger['Channel'] > 2.5)[0][0]

    sliced_channel = channel.iloc[start_index:start_index + data_length]
    return sliced_channel['Channel'].to_numpy()

# ============================================================
# 2 Data Output Functions
# ============================================================

# plot data of a single data
def single_plot(data, device_name="Device", start_index=0, end_index=0):
    '''
    Input: the sliced channel data of 1 dimension with arbitrary length,
           device name
    Output the plot of the data
    '''

    if len(data.shape) != 1:
        print(f"Error: single dimension data required, got shape {data.shape}")
        return
    
    if end_index == 0 and start_index == 0:
        end_index = len(data)

    x_points = np.arange(len(data)) * (end_index - start_index) / (len(data)) + start_index

    plt.figure(figsize=(10, 6))
    plt.plot(x_points, data)
    plt.title(f"{device_name} Channel Data")
    plt.xlabel("Data Points")
    plt.ylabel("Channel Value")
    plt.grid()
    plt.savefig(f"{device_name}_data_plot.png")
    plt.show()
    return

def multi_plot(data_tensor, start_index=0, end_index=0, legend = []):
    '''
    Input: data_tensor of shape (device_no, DATA_LENGTH)
           device_set (dict), list of devices and their paths
    Output: multiple plots for each device
    '''
    if len(data_tensor.shape) != 2:
        print(f"Error: 2D data tensor required, got shape {data_tensor.shape}")
        return
    
    if end_index == 0 and start_index == 0:
        end_index = data_tensor.shape[1]
    
    plt.figure(figsize=(10, 6))

    # Determine the number of lines to colorize
    num_lines = data_tensor.shape[0]

    for i in range(num_lines):
        data = data_tensor[i]
        x_points = np.arange(len(data)) * (end_index - start_index) / (len(data)) + start_index
        
        # Calculate color: 0.0 is purple, 1.0 is yellow (using plasma)
        # We use a conditional to avoid division by zero if there's only one line
        # line_color = plt.cm.viridis(i / (num_lines - 1)) if num_lines > 1 else 'purple'
        
        plt.plot(x_points, data, label=legend[i] if legend else f"Device {i+1}")

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Channel Value")
    plt.legend()
    plt.grid()
    plt.show()

    # save multiplot as csv file in the output folder, with the first column as wavelength and the following columns as data for each device
    axis_range = np.linspace(start_index, end_index, data_tensor.shape[-1])
    np.savetxt("multiplot_data.csv", np.column_stack([axis_range] + [data_tensor[i] for i in range(data_tensor.shape[0])]), delimiter=",")

    return

# package data of device sets into stacked data tensor
def package_data(device_set, data_length, folder_path, subset=[]):
    '''
    Input: device_set (dict), list of devices and their paths,
           data_length (int), length of data to be read
           folder_path (str), path to the folder containing the data files
           subset (list), list of device names to include; if empty, include all
    Ouput: tensor of shape (device_no, DATA_LENGTH)
    '''
    if subset == []:
        data_tensor = np.zeros((len(device_set), data_length))
        for i, (device_name, paths) in enumerate(device_set.items()):
            channel_path = folder_path + paths["channel"]
            trigger_path = folder_path + paths["trigger"]
            data_tensor[i] = read_data(channel_path, trigger_path)
        return data_tensor
    else:
        data_tensor = np.zeros((len(subset), data_length))
        for i, device_name in enumerate(subset):
            if device_name not in device_set:
                print(f"Warning: device {device_name} not found in device set.")
                continue
            paths = device_set[device_name]
            print(f"Processing device: {device_name}")
            channel_path = folder_path + paths["channel"]
            trigger_path = folder_path + paths["trigger"]
            data_tensor[i] = read_data(channel_path, trigger_path)
        return data_tensor

# package data that has been organized using a record table
def package_csv_data(record, index, data_length):
    '''
    Input: device_set (dict), list of devices and their paths,
           data_length (int), length of data to be read
           folder_path (str), path to the folder containing the data files
           subset (list), list of device names to include; if empty, include all
    Ouput: tensor of shape (device_no, DATA_LENGTH)
    '''
    data_tensor = np.zeros((1, data_length))
    print(f"Processing device: {'_'.join(record.loc[index, ['date', 'device', 'sub-device', 'port']].astype(str)) + '.csv'}")
    channel_path = record['folder'][index] + record['channel path'][index] + '.csv'
    trigger_path = record['folder'][index] + record['trigger path'][index] + '.csv'
    data_tensor[0] = read_data(channel_path, trigger_path)
    return data_tensor

# save data as csv for TMM
def save_data_for_TMM(file_name, data_tensor, start_index=0, end_index=0, legend = []):
    '''
    Input: data_tensor of shape (2, DATA_LENGTH)
           device_set (dict), list of devices and their paths
    Output: saved data as csv
    '''
    if len(data_tensor.shape) != 2:
        print(f"Error: 2D data tensor required, got shape {data_tensor.shape}")
        return

    if data_tensor.shape[0] != 1:
        print(f"Error: To use TMM fitting, there must only be one set of data")
    
    if end_index == 0 and start_index == 0:
        end_index = data_tensor.shape[1]

    # save multiplot as csv file in the output folder, with the first column as wavelength and the second column as transmission data
    axis_range = np.linspace(start_index, end_index, data_tensor.shape[-1])
    txt = np.column_stack([axis_range, data_tensor[0]])
    np.savetxt(file_name, txt, delimiter=",", header="wavelength_nm,transmission_dB", comments="")

    return file_name

def multi_save(file_name, data_tensor, start_index=0, end_index=0, legend = []):
    '''
    Input: data_tensor of shape (device_no, DATA_LENGTH)
           device_set (dict), list of devices and their paths
    Output: multiple plots for each device
    '''
    if len(data_tensor.shape) != 2:
        print(f"Error: 2D data tensor required, got shape {data_tensor.shape}")
        return
    
    if end_index == 0 and start_index == 0:
        end_index = data_tensor.shape[1]
    
    plt.figure(figsize=(10, 6))

    # Determine the number of lines to colorize
    num_lines = data_tensor.shape[0]

    # save multiplot as csv file in the output folder, with the first column as wavelength and the following columns as data for each device
    axis_range = np.linspace(start_index, end_index, data_tensor.shape[-1])
    np.savetxt(file_name, np.column_stack([axis_range] + [data_tensor[i] for i in range(data_tensor.shape[0])]), delimiter=",")

    return

# ============================================================
# 3 Signal Processing Functions
# ============================================================

# moving average
def moving_average(data, window_size):
    '''
    Input: data of shape of 1 dimension with arbitrary length
           factor: window size for moving average
    Output: data of 1 dimension with the same length
    '''

    if len(data.shape) != 1:
        print(f"Error: single dimension data required, got shape {data.shape}")
        return None

    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='same')

# multiplication & dB transformer
def normalize_data(data_tensor, basis, dB = True):
    '''
    Input: data_tensor of any shape with the last dimension (:,:,DATA_LENGTH)
           basis: basis data of shape (DATA_LENGTH)
    Output: normalized data_tensor in dB scale with the same shape
    '''

    if basis.shape[0] != data_tensor.shape[-1]:
        print(f"Error: basis length {basis.shape[0]} does not match data tensor last dimension {data_tensor.shape[-1]}")
        return None

    if dB:
        normalized_tensor = 10 * np.log10(data_tensor / (basis))# + 1e-12) + 1e-12)  # avoid division by zero and log of zero
    else:
        normalized_tensor = data_tensor / (basis + 1e-12)  # avoid division by zero
    return normalized_tensor

# self-normalization
def normalize_each_date(data_tensor, window_size=None):
    '''
    Input: data_tensor of any shape with the last dimension (:,:,DATA_LENGTH)
    Outpput: normalized data_tensor with the same shape, each data normalized by its own moving average basis
    '''
    normalized_tensor = np.zeros_like(data_tensor)
    for index in np.ndindex(data_tensor.shape[:-1]):
        if window_size is None:
            basis = np.max(data_tensor[index])
        else:
            basis = moving_average(data_tensor[index], window_size=window_size)
        normalized_tensor[index] = data_tensor[index] / basis
    return normalized_tensor

# downsample data
def downsample_data(data_tensor, points = 50):
    '''
    Input: data_tensor of any shape with the last dimension
           points: number of points as a group of mean
    Output: data tensor with the same shape except the last dimension is reduced by factor
    '''
    new_shape = data_tensor.shape[:-1] + (data_tensor.shape[-1] // points,)
    downsampled_tensor = np.zeros(new_shape)
    for index in np.ndindex(data_tensor.shape[:-1]):
        for i in range(new_shape[-1]):
            downsampled_tensor[index + (i,)] = np.mean(data_tensor[index + (slice(i*points, (i+1)*points),)])
    return downsampled_tensor


def reciprocal_axis_data(data_tensor, start_index=1260, end_index=1360):
    '''
    Input: data_tensor of any shape
    Output: reciprocal data tensor with the same shape
            values are slightly adjusted to avoid division by zero
    '''
    axis_range = np.linspace(start_index, end_index, data_tensor.shape[-1])
    reciprical_axis_origin = 1 / (axis_range + 1e-6)  # avoid division by zero
    reciprical_axis = np.linspace(reciprical_axis_origin[0], reciprical_axis_origin[-1], data_tensor.shape[-1])
    reciprical_data_tensor = np.zeros_like(data_tensor)
    for index in np.ndindex(data_tensor.shape[:-1]):
        reciprical_data_tensor[index] = np.interp(np.flip(reciprical_axis), np.flip(reciprical_axis_origin), np.flip(data_tensor[index]))
    return reciprical_data_tensor

def subdivide_data(data_tensor, num_segments):
    '''
    Input: data_tensor of any shape with the last dimension
           segment_length: length of each segment
    Output: data tensor subdivided into segments along the last dimension
            new shape: (..., num_segments, segment_length)
    '''
    segment_length = data_tensor.shape[-1] // num_segments
    new_shape = data_tensor.shape[:-1] + (num_segments, segment_length)
    subdivided_tensor = np.zeros(new_shape)
    for index in np.ndindex(data_tensor.shape[:-1]):
        for i in range(num_segments):
            subdivided_tensor[index + (i,)] = data_tensor[index + (slice(i*segment_length, (i+1)*segment_length),)]
    return subdivided_tensor

def find_slope(data_tensor, x_values, factor=1000):
    '''
    Input: data_tensor of 2D shape (n, DATA_LENGTH), DATA_LENGTH should be regarded as measured points
           x_values: 1D array of length n
    Output: slope_tensor of shape (DATA_LENGTH // factor,)
    '''

    if len(data_tensor.shape) != 2:
        print(f"Error: 2D data tensor required, got shape {data_tensor.shape}")
        return None
    
    if data_tensor.shape[0] != len(x_values):
        print(f"Error: data tensor rows {data_tensor.shape[0]} do not match x_values length {len(x_values)}")
        return None

    slope_tensor = np.zeros((data_tensor.shape[1] // factor + 1,))
    intercept_tensor = np.zeros((data_tensor.shape[1] // factor + 1,))
    for data_index in range(data_tensor.shape[1]):
        if data_index % factor == 0:
            slope, intercept, r_value, p_value, std_err = linregress(x_values, data_tensor[:,data_index])
            slope_tensor[data_index // factor] = round(slope, 7)  # Round slope to 3 decimal places
            intercept_tensor[data_index // factor] = round(intercept, 7)
    return slope_tensor, intercept_tensor

# peak finder
def find_peaks(data, peak_number=10, distance=10):
    '''
    Input: data of shape of 1 dimension with arbitrary length
           peak_number:number of peaks to find
    Output: array of peak indices,
            intervals of each lorentzian peak, with start and end indices
    '''
    data_length = len(data)
    if peak_number != 0:
        widest = data_length // peak_number
        thinnest = max(1, widest // 8)
        peak_indices = signal.find_peaks_cwt(data, widths=np.arange(thinnest, widest))
        peak_indices = np.sort(np.array(peak_indices))
    else:
        peak_indices,_ = signal.find_peaks(data, distance=distance)
        peak_indices=np.array(peak_indices)
        peak_indices = np.sort(np.array(peak_indices))
    intervals = (peak_indices[:-1] + peak_indices[1:]) // 2
    intervals = np.concatenate(([0], intervals, [data_length]))
    return peak_indices, intervals

def subdivide_peaks(data, intervals):
    '''
    Input: data of shape of 1 dimension with arbitrary length
           intervals: array of start and end indices for each peak
    Output: list of data segments for each peak, with each segment as 1D array, arbitrary length
    '''
    for i in range(len(intervals) - 1):
        yield data[intervals[i]:intervals[i+1]]
        
def find_lorentz(data, plot=False):
    '''
    Input: data of shape of 1 dimension with arbitrary length
           start_index and end_index: indices for the whole data range
           plot: whether to plot the fitting result
    Output: FWHM value (width at half maximum)
            Center value (peak position)
            Amplitude value (peak height)
    '''
    if len(data.shape) != 1:
        print(f"Error: single dimension data required, got shape {data.shape}")
        return None

    if len(data) < 4:
        return 0, 0, 0

    peak_index = np.argmax(data)
    peak_value = np.max(data)
    model = LorentzianModel() + ConstantModel()
    params = model.make_params()
    peak_height = peak_value - np.min(data)
    rough_width = len(data) // 2
    params['center'].set(value=peak_index, min=0, max=len(data)-1)
    params['amplitude'].set(value=peak_height * np.pi * rough_width / 2, min=0)
    params['sigma'].set(value=rough_width / 2, min=0.01)   # sigma = HWHM
    params['c'].set(value=np.min(data))

    result = model.fit(data, params, x=np.arange(len(data)))

    # ─── Results ───────────────────────────────────────────────────────────
    # print(result.fit_report())

    # Easy access
    #print("\nKey values:")
    #print(f"Center    = {result.params['center'].value:.3f} ± {result.params['center'].stderr:.4f}")
    #print(f"FWHM      = {result.params['fwhm'].value:.3f}")
    #print(f"Amplitude = {result.params['amplitude'].value:.3f}")
    #print(f"Background = {result.params['c'].value:.4f}")

    fwhm = result.params['fwhm'].value
    center = result.params['center'].value
    amplitude = result.params['amplitude'].value
    
    reduced_chi_square = result.redchi
    r_squared = result.rsquared

    if plot:
        # ─── Plot ──────────────────────────────────────────────────────────────
        plt.plot(np.arange(len(data)), data, 'o', ms=4, alpha=0.6, label='data')
        plt.plot(np.arange(len(data)), result.best_fit, 'r-', lw=2.5, label='best fit')
        plt.plot(np.arange(len(data)), result.eval_components(x=np.arange(len(data)))['constant'], '--k', label='background')
        plt.plot(peak_index, data[peak_index], 'ro', label='peak')
        plt.legend()
        plt.title('Single Lorentzian Fit — lmfit')
        plt.show()

    return fwhm, center, amplitude, reduced_chi_square, r_squared

def find_mode_average(data, division=20, gate=False, plot=False):
    '''
    Input: data of shape of 1 dimension with arbitrary length
           division: division number for histogram
           gate: whether to apply gating lower 10% and upper 10%
    Output: average value around the mode
    '''
    if len(data.shape) != 1:
        print(f"Error: single dimension data required, got shape {data.shape}")
        return None

    if gate:
        lower_bound = np.percentile(data, 10)
        upper_bound = np.percentile(data, 90)
        data = data[(data >= lower_bound) & (data <= upper_bound)]

    density, edges = np.histogram(data, division, density=True)
    # density with length of division
    # edges with length of division + 1
    max_dense_index = np.argmax(density)
    min_edge = edges[max_dense_index]
    max_edge = edges[max_dense_index + 1]
    selected_data_value = data[(data>=min_edge)&(data<=max_edge)]

    if plot:
        edges_center = (edges[:-1] + edges[1:]) / 2
        plt.bar(edges_center, density, width=edges[1]-edges[0], alpha=0.6, label='Histogram')
        plt.axvline(min_edge, color='r', linestyle='--', label='Mode Range')
        plt.axvline(max_edge, color='r', linestyle='--')
        plt.title('Histogram and Mode Range')
        plt.xlabel('Data Value')
        plt.ylabel('Density')
        plt.legend()
        plt.show()

    return np.average(selected_data_value)

def calculate_fwhm_whole(data, peak_number, distance, gate=True, start_index=1260, end_index=1360, length=9000000):
    '''
    Input: data: numpy array of 1 dimension
           peak_number: manually count
           distance: minimum distance between peaks
           gate: whether to gate the results based on r_squared
           start_index and end_index: indices for the whole data range
    Output: fwhm_mode_average
            center_diff_mode_average
            q_factor_mode_average
            q_factor_average
            q_factor_std

    '''
    peak_indices, intervals = find_peaks(data, peak_number=peak_number, distance=distance)
    print("Peak indices:", peak_indices)
    print("Intervals:", intervals)
    sampled_data = np.zeros_like(data, dtype=float)
    sampled_data[peak_indices] = data[peak_indices]
    multi_plot(np.stack([data, sampled_data], axis=0), start_index=0, end_index=0)
    subdivided_data_tensor = list(subdivide_peaks(data, intervals))
    print("Number of subdivided peaks:", len(subdivided_data_tensor))
    fwhm_list = []
    center_list = []
    center_count = 0
    reduced_chi_square_list = []
    r_squared_list = []
    ng_list = []
    for i, segment in enumerate(subdivided_data_tensor):
        fwhm, center, amplitude, reduced_chi_square, r_squared = find_lorentz(segment, plot=False)
        fwhm = fwhm * (end_index - start_index) / data.shape[0]
        center = center + center_count
        fwhm_list.append(fwhm)
        center_list.append(center)
        center_count += segment.shape[0]
        reduced_chi_square_list.append(reduced_chi_square)
        r_squared_list.append(r_squared)
        if i:
            center = center * (end_index - start_index) / data.shape[0] + start_index
            last_center = center_list[-2] * (end_index - start_index) / data.shape[0] + start_index
            ng_list.append(center ** 2 / length / (center - last_center))
    fwhm_list = np.array(fwhm_list)
    reduced_chi_square_list = np.array(reduced_chi_square_list)
    r_squared_list = np.array(r_squared_list)
    center_list = np.array(center_list)
    center_list = center_list * (end_index - start_index) / data.shape[0] + start_index 
    q_factor_list = center_list / (fwhm_list + 1e-10)  # avoid division by zero
    center_diff = center_list[1:] - center_list[:-1]
    
    print("Average FWHM:", np.mean(fwhm_list))
    print("Standard Deviation of FWHM:", np.std(fwhm_list))
    print("Average Center Difference:", np.mean(center_diff))
    print("Standard Deviation of Center Difference:", np.std(center_diff))
    print("Average Q factor:", np.mean(q_factor_list))
    print("Standard Deviation of Q factor:", np.std(q_factor_list))

    reduced_chi_square_mode_average = find_mode_average(reduced_chi_square_list, gate=True, division = 30, plot=True)
    r_squared_mode_average = find_mode_average(r_squared_list, gate=True, division = 30, plot=True)

    print("Reduced Chi-Square Mode Average:", reduced_chi_square_mode_average)
    print("R-Squared Mode Average:", r_squared_mode_average)

    if gate:
        sample_list = r_squared_list > np.percentile(r_squared_list, 70)
        fwhm_list = fwhm_list[(sample_list)]
        q_factor_list = q_factor_list[(sample_list)]
        center_diff = center_diff[(sample_list[:-1] & sample_list[1:])]
        fwhm_mode_average = find_mode_average(fwhm_list, gate=True, division=10, plot=True)
        center_diff_mode_average = find_mode_average(center_diff, gate=False, division=50, plot=True)
        q_factor_mode_average = find_mode_average(q_factor_list, gate=True, division=20, plot=True)
        ng_mode_average = find_mode_average(np.array(ng_list), gate=True, division=50, plot=True)
        q_factor_average = np.mean(q_factor_list)
        q_factor_std = np.std(q_factor_list)
        ng_average = np.mean(np.array(ng_list))
    else:
        fwhm_mode_average = find_mode_average(fwhm_list, gate=True, plot=True)
        center_diff_mode_average = find_mode_average(center_diff, gate=False, division=50, plot=True)
        q_factor_mode_average = find_mode_average(q_factor_list, gate=True, division=50, plot=True)

    print("fwhm mode:", round(fwhm_mode_average, 5))
    print("fsr mode:", round(center_diff_mode_average,5))
    print("Q factor mode:", round(q_factor_mode_average, 5))
    print("Q factor average:", round(q_factor_average, 5))
    print("Q factor std:", round(q_factor_std, 5))
    print("ng mode:", round(ng_mode_average, 10))


    top_r_squared_argument = np.argsort(r_squared_list)[::len(r_squared_list)//10][:10]
    for idx in top_r_squared_argument:
        segment = subdivided_data_tensor[idx]
        fwhm, center, amplitude, reduced_chi_square, r_squared = find_lorentz(segment, plot=True)
        fwhm = fwhm * (end_index - start_index) / data.shape[0]
        print("current fit FWHM: ", fwhm)

    return fwhm_mode_average, center_diff_mode_average, q_factor_mode_average, q_factor_average, q_factor_std

def plot_3D(data, x_axis, sample_rate=1, annotate=True):
    '''
    Input: data:   shape (device_no, DATA_LENGTH)
           x_axis: 1D array of length device_no
    Output: 3D surface plot
    '''
    print("Plotting 3D surface...")
    plt.figure(figsize=(10, 6))
    for i in range(data.shape[1]):
        if i % sample_rate == 0:
            plt.plot(x_axis, data[:, i], marker='o')
            for x, y in zip(x_axis, data[:, i]):     
                xy = (round(x, 3), round(y, 3))
                if annotate:
                    plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
    plt.title("Sampled data over spectrum")
    plt.xlabel("Device Length")
    plt.ylabel("Normalized Amplitude (dB)")
    plt.show()
    return