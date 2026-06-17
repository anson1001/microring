from sympy import factor

from functions import *
import TMM

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit

SWEEP_RANGE = 100 # from 1260nm to 1360nm
SWEEP_RATE = 2 # nm /2
NUM_ROWS = 2*10**6
TIME_AMOUNT = 64
DATA_LENGTH = int(NUM_ROWS / TIME_AMOUNT * SWEEP_RANGE / SWEEP_RATE)

# def case_3_1():
#     START_INDEX = 1260
#     END_INDEX = 1360
#     subset = ["temp_22.5"]
#     '''
#     data_tensor = package_data_without_trigger(DEVICE_SET_14, 2000000, FOLDER_PATH_14, subset=subset)
#     print("Data tensor shape:", data_tensor.shape)
#     #normalized_data_tensor = normalize_each_date(data_tensor, window_size=10000)
#     #print("Normalized data tensor shape:", normalized_data_tensor.shape)
#     downsampled_data_tensor = downsample_data(data_tensor, points=10)
#     print("Downsampled data tensor shape:", downsampled_data_tensor.shape)
#     '''
#     data_tensor_1 = pd.read_csv(FOLDER_PATH_18 + DEVICE_SET_18[subset[0]]["channel"], skiprows=15)
#     data_tensor_2 = pd.read_csv(FOLDER_PATH_18 + DEVICE_SET_18[subset[0]]["trigger"], skiprows=15)
#     print("Data tensor 1 shape:", data_tensor_1.shape)
#     print("Data tensor 2 shape:", data_tensor_2.shape)
#     plt.plot(data_tensor_1)
#     plt.plot(data_tensor_2)
#     plt.plot()
#     plt.legend()
#     plt.show()

# def case_3_2():
#     subset = ["device_02_through", "device_02_drop", "device_03_through", "device_03_drop", "device_04_through", "device_04_drop", "device_05_through", "device_05_drop", "device_06_through", "device_06_drop", "device_07_through", "device_07_drop", "device_08_through", "device_08_drop", "device_09_through", "device_09_drop", "device_10_through", "device_10_drop"]
#     for device in subset:
#         channel_path, trigger_path = FOLDER_PATH_15 + DEVICE_SET_15[device]["channel"], FOLDER_PATH_15 + DEVICE_SET_15[device]["trigger"]
#         data_tensor_channel = pd.read_csv(channel_path, skiprows=15)
#         data_tensor_trigger = pd.read_csv(trigger_path, skiprows=15)
#         data_tensor_channel = data_tensor_channel[:]["CH1"]
#         data_tensor_trigger = data_tensor_trigger[:]["CH2"]
#         print(device)
#         print(f"Channel shape: {data_tensor_channel.shape}, Trigger shape: {data_tensor_trigger.shape}")
#         print(f"Channel max: {np.max(data_tensor_channel)}, Trigger max: {np.max(data_tensor_trigger)}")
#         if np.max(data_tensor_trigger) > 10 *  np.mean(data_tensor_trigger):
#             data_tensor_trigger = data_tensor_trigger * np.max(data_tensor_channel) / np.max(data_tensor_trigger)
#         plt.plot(data_tensor_channel)
#         plt.plot(data_tensor_trigger)
#         plt.title(device)
#         plt.savefig(f"{FOLDER_PATH_15}{device}.png")
#         plt.close()

FOLDER_PATH_8 = "./27_3_2026_vernier/"
DEVICE_SET_8 = {
    # row 4 V10
    # voltage unit: 0.1V
    # output scale: 200mV * 4
    # recording outout 1, through port
    "voltage_05": {
        "trigger": "TEK00049.csv",
        "channel": "TEK00048.csv"
    },
    # fwhm mode: 0.0907
    # fsr mode: 0.08756
    # Q factor mode: 13683.65868
    # Q factor average: 14473.44625
    # Q factor std: 3573.17202
    "voltage_06": {
        "trigger": "TEK00051.csv",
        "channel": "TEK00050.csv"
    },
    # fwhm mode: 0.08357
    # fsr mode: 0.08643
    # Q factor mode: 15690.55405
    # Q factor average: 14526.1814
    # Q factor std: 3457.75566
    "voltage_07": {
        "trigger": "TEK00053.csv",
        "channel": "TEK00052.csv"
    },
    # fwhm mode: 0.07304
    # fsr mode: 0.0854
    # Q factor mode: 16198.48706
    # Q factor average: 14477.22511
    # Q factor std: 3575.50313
    "voltage_08": { #retest
        "trigger": "TEK00071.csv",
        "channel": "TEK00070.csv"
    },
    # fwhm mode: 0.08109
    # fsr mode: 0.08785
    # Q factor mode: 15131.77077
    # Q factor average: 14303.44377
    # Q factor std: 3238.3053
    "voltage_09": { #retest
        "trigger": "TEK00073.csv",
        "channel": "TEK00072.csv"
    },
    # fwhm mode: 0.08007
    # fsr mode: 0.08773
    # Q factor mode: 13677.30069
    # Q factor average: 14322.82197
    # Q factor std: 3465.6421
    "voltage_10": {
        "trigger": "TEK00059.csv",
        "channel": "TEK00058.csv"
    },
    # fwhm mode: 0.08559
    # fsr mode: 0.09058
    # Q factor mode: 14693.17604
    # Q factor average: 14244.25979
    # Q factor std: 3346.65403
    "voltage_11": {
        "trigger": "TEK00061.csv",
        "channel": "TEK00060.csv"
    },
    # fwhm mode: 0.07807
    # fsr mode: 0.08664
    # Q factor mode: 14263.24846
    # Q factor average: 14500.06402
    # Q factor std: 3446.0433
    "voltage_12": { #retest
        "trigger": "TEK00075.csv",
        "channel": "TEK00074.csv"
    },
    # fwhm mode: 0.09731
    # fsr mode: 0.08804
    # Q factor mode: 13908.35187
    # Q factor average: 14213.9647
    # Q factor std: 3370.1399
    "voltage_13": { #retest
        "trigger": "TEK00077.csv",
        "channel": "TEK00076.csv"
    },
    # fwhm mode: 0.07888
    # fsr mode: 0.08749
    # Q factor mode: 15535.50044
    # Q factor average: 14423.06491
    # Q factor std: 3531.29794
    "voltage_14": {
        "trigger": "TEK00067.csv",
        "channel": "TEK00066.csv"
    },
    # fwhm mode: 0.07966
    # fsr mode: 0.08643
    # Q factor mode: 12367.51057
    # Q factor average: 14262.34758
    # Q factor std: 3435.44962
    "voltage_15": {
        "trigger": "TEK00069.csv",
        "channel": "TEK00068.csv"
    },
    # fwhm mode: 0.08028
    # fsr mode: 0.08743
    # Q factor mode: 13967.31993
    # Q factor average: 14290.76553
    # Q factor std: 3193.88138
}

DEVICE_SET_9 = {
    # output of the overall port of vernier device
    "voltage_05": {
        "trigger": "TEK00079.csv",
        "channel": "TEK00078.csv"
    },
    "voltage_06": {
        "trigger": "TEK00081.csv",
        "channel": "TEK00080.csv"
    },
    "voltage_07": {
        "trigger": "TEK00083.csv",
        "channel": "TEK00082.csv"
    },
    "voltage_08": {
        "trigger": "TEK00085.csv",
        "channel": "TEK00084.csv"
    },
    "voltage_09": {
        "trigger": "TEK00087.csv",
        "channel": "TEK00086.csv"
    },
    "voltage_10": {
        "trigger": "TEK00089.csv",
        "channel": "TEK00088.csv"
    },
    "voltage_11": {
        "trigger": "TEK00091.csv",
        "channel": "TEK00090.csv"
    },
    "voltage_12": {
        "trigger": "TEK00093.csv",
        "channel": "TEK00092.csv"
    },
    "voltage_13": {
        "trigger": "TEK00095.csv",
        "channel": "TEK00094.csv"
    },
    "voltage_14": {
        "trigger": "TEK00097.csv",
        "channel": "TEK00096.csv"
    },
    "voltage_15": {
        "trigger": "TEK00099.csv",
        "channel": "TEK00098.csv"
    }
}

FOLDER_PATH_10 = "./30_3_2026_vernier/"
DEVICE_SET_10 = {
    # output of through port of the first microring
    "voltage_06": {
        "trigger": "TEK00101.csv",
        "channel": "TEK00100.csv"
    },
    "voltage_15": {
        "trigger": "TEK00103.csv",
        "channel": "TEK00102.csv"
    },
}

FOLDER_PATH_11 = "./31_3_2026_vernier/"
DEVICE_SET_11 = {
    # output of through port of the second microring
    "voltage_05": {
        "trigger": "TEK00116.csv",
        "channel": "TEK00115.csv"
    },
    "voltage_10": {
        "trigger": "TEK00112.csv",
        "channel": "TEK00111.csv"
    },
    "voltage_15": {
        "trigger": "TEK00114.csv",
        "channel": "TEK00113.csv"
    },
    "voltage_20": {
        "trigger": "TEK00109.csv",
        "channel": "TEK00108.csv"
    },
    "voltage_25": {
        "trigger": "TEK00107.csv",
        "channel": "TEK00106.csv"
    },
    "voltage_30": {
        "trigger": "TEK00105.csv",
        "channel": "TEK00104.csv"
    },
}

FOLDER_PATH_12 = "./31_3_2026_vernier/"
DEVICE_SET_12 = {
    # output of through port of the third microring
    "voltage_10": {
        "trigger": "TEK00049.csv",
        "channel": "TEK00048.csv"
    },
    "voltage_15": {
        "trigger": "TEK00051.csv",
        "channel": "TEK00050.csv"
    },
    "voltage_20": {
        "trigger": "TEK00053.csv",
        "channel": "TEK00052.csv"
    },
    "voltage_25": {
        "trigger": "TEK00055.csv",
        "channel": "TEK00054.csv"
    },
    "voltage_30": {
        "trigger": "TEK00057.csv",
        "channel": "TEK00056.csv"
    },
}

FOLDER_PATH_13 = "./31_3_2026_vernier_AC/"
DEVICE_SET_13 = {
    "voltage_3.8kHz": {
        "channel": "TEK00058.csv",
        "AC": "TEK00059.csv"
    },
    "voltage_10kHz": {
        "channel": "TEK00061.csv",
        "AC": "TEK00062.csv"
    },
    "voltage_20kHz": {
        "channel": "TEK00063.csv",
        "AC": "TEK00064.csv"
    },
    "voltage_30kHz": {
        "channel": "TEK00066.csv",
        "AC": "TEK00067.csv"
    },
    "voltage_2kHz": {
        "channel": "TEK00068.csv",
        "AC": "TEK00069.csv"
    },
    "voltage_1kHz": {
        "channel": "TEK00071.csv",
        "AC": "TEK00072.csv"
    },
    "voltage_500Hz": {
        "channel": "TEK00073.csv",
        "AC": "TEK00074.csv"
    },
    "voltage_300Hz": {
        "channel": "TEK00075.csv",
        "AC": "TEK00076.csv"
    },
    "voltage_100Hz": {
        "channel": "TEK00077.csv",
        "AC": "TEK00078.csv"
    },
}

FOLDER_PATH_14 = "./31_3_2026_vernier/"
DEVICE_SET_14 = {
    "voltage_00": {
        "trigger": "TEK00080.csv",
        "channel": "TEK00079.csv"
    },
    "voltage_10": {
        "trigger": "TEK00082.csv",
        "channel": "TEK00081.csv"
    },
    "voltage_15": {
        "trigger": "TEK00084.csv",
        "channel": "TEK00083.csv"
    },
    "voltage_20": {
        "trigger": "TEK00086.csv",
        "channel": "TEK00085.csv"
    },
    "voltage_25": {
        "trigger": "TEK00088.csv",
        "channel": "TEK00087.csv"
    },
    "voltage_30": {
        "trigger": "TEK00090.csv",
        "channel": "TEK00089.csv"
    },
}

FOLDER_PATH_15 = "./1_4_2026_vernier_all/"
DEVICE_SET_15 = {
    "device_02_through": {
        "trigger": "TEK00092.csv",
        "channel": "TEK00091.csv"
    },
    "device_02_drop": {
        "trigger": "TEK00094.csv",
        "channel": "TEK00093.csv"
    },
    "device_03_through": {
        "trigger": "TEK00096.csv",
        "channel": "TEK00095.csv"
    },
    "device_03_drop": {
        "trigger": "TEK00098.csv",
        "channel": "TEK00097.csv"
    },
    "device_04_through": {
        "trigger": "TEK00100.csv",
        "channel": "TEK00099.csv"
    },
    "device_04_drop": {
        "trigger": "TEK00102.csv",
        "channel": "TEK00101.csv"
    },
    "device_05_through": {
        "trigger": "TEK00104.csv",
        "channel": "TEK00103.csv"
    },
    "device_05_drop": {
        "trigger": "TEK00106.csv",
        "channel": "TEK00105.csv"
    },
    "device_06_through": {
        "trigger": "TEK00108.csv",
        "channel": "TEK00107.csv"
    },
    "device_06_drop": {
        "trigger": "TEK00110.csv",
        "channel": "TEK00109.csv"
    },
    "device_07_through": {
        "trigger": "TEK00112.csv",
        "channel": "TEK00111.csv"
    },
    "device_07_drop": {
        "trigger": "TEK00114.csv",
        "channel": "TEK00113.csv"
    },
    "device_08_through": {
        "trigger": "TEK00116.csv",
        "channel": "TEK00115.csv"
    },
    "device_08_drop": {
        "trigger": "TEK00118.csv",
        "channel": "TEK00117.csv"
    },
    "device_09_through": {
        "trigger": "TEK00120.csv",
        "channel": "TEK00119.csv"
    },
    "device_09_drop": {
        "trigger": "TEK00122.csv",
        "channel": "TEK00121.csv"
    },
    "device_10_through": {
        "trigger": "TEK00124.csv",
        "channel": "TEK00123.csv"
    },
    "device_10_drop": {
        "trigger": "TEK00126.csv",
        "channel": "TEK00125.csv"
    },
}
# bad ones: 02_through 02_drop 03_through 03_drop 04_drop 09_through
# good ones: 07 08

FOLDER_PATH_16 = "./1_4_2026_vernier_device_10_EO/"
DEVICE_SET_16 = {
    "voltage_00": {
        "trigger": "TEK00128.csv",
        "channel": "TEK00127.csv"
    },
    "voltage_05": {
        "trigger": "TEK00130.csv",
        "channel": "TEK00129.csv"
    },
    "voltage_10": {
        "trigger": "TEK00132.csv",
        "channel": "TEK00131.csv"
    },
    "voltage_15": {
        "trigger": "TEK00134.csv",
        "channel": "TEK00133.csv"
    },
    "voltage_20": {
        "trigger": "TEK00136.csv",
        "channel": "TEK00135.csv"
    },
    "voltage_25": {
        "trigger": "TEK00138.csv",
        "channel": "TEK00137.csv"
    },
}

FOLDER_PATH_17 = "./9_4_2026_V8_through_EO/"
DEVICE_SET_17 = {
    "voltage_00": {
        "trigger": "TEK00141.csv",
        "channel": "TEK00140.csv"
    },
    "voltage_05": {
        "trigger": "TEK00143.csv",
        "channel": "TEK00142.csv"
    },
    "voltage_10": {
        "trigger": "TEK00145.csv",
        "channel": "TEK00144.csv"
    },
    "voltage_15": {
        "trigger": "TEK00147.csv",
        "channel": "TEK00146.csv"
    },
    "voltage_20": {
        "trigger": "TEK00155.csv",
        "channel": "TEK00154.csv"
    },
    "voltage_25": {
        "trigger": "TEK00151.csv",
        "channel": "TEK00150.csv"
    },
    "voltage_30": {
        "trigger": "TEK00153.csv",
        "channel": "TEK00152.csv"
    },
}

FOLDER_PATH_18 = "./9_4_2026_V8_drop2_EO_TO/"
DEVICE_SET_18 = {
    "temp_22.5": {
        "trigger": "TEK00161.csv",
        "channel": "TEK00160.csv"
    },
    "temp_20.2": {
        "trigger": "TEK00159.csv",
        "channel": "TEK00158.csv"
    }
}

FOLDER_PATH_19 = "./9_4_2026_V8_drop2_EO/"
DEVICE_SET_19 = {
    "voltage_00": {
        "trigger": "TEK00163.csv",
        "channel": "TEK00162.csv"
    },
    "voltage_05": {
        "trigger": "TEK00165.csv",
        "channel": "TEK00164.csv"
    },
    "voltage_10": {
        "trigger": "TEK00167.csv",
        "channel": "TEK00166.csv"
    },
    "voltage_15": {
        "trigger": "TEK00169.csv",
        "channel": "TEK00168.csv"
    },
    "voltage_20": {
        "trigger": "TEK00171.csv",
        "channel": "TEK00170.csv"
    },
    "voltage_25": {
        "trigger": "TEK00173.csv",
        "channel": "TEK00172.csv"
    },
    "voltage_30": {
        "trigger": "TEK00175.csv",
        "channel": "TEK00174.csv"
    },
}

FOLDER_PATH_20 = "./9_4_2026_V8_ring1_drop_EO/"
DEVICE_SET_20 = {
    "voltage_00": {
        "trigger": "TEK00092.csv",
        "channel": "TEK00091.csv"
    },
    "voltage_05": {
        "trigger": "TEK00094.csv",
        "channel": "TEK00093.csv"
    },
    "voltage_10": {
        "trigger": "TEK00096.csv",
        "channel": "TEK00095.csv"
    },
    "voltage_15": {
        "trigger": "TEK00098.csv",
        "channel": "TEK00097.csv"
    },
    "voltage_20": {
        "trigger": "TEK00100.csv",
        "channel": "TEK00099.csv"
    },
    "voltage_25": {
        "trigger": "TEK00102.csv",
        "channel": "TEK00101.csv"
    },
    "voltage_30": {
        "trigger": "TEK00104.csv",
        "channel": "TEK00103.csv"
    },
}

FOLDER_PATH_21 = "./9_4_2026_V8_ring1_through/"
DEVICE_SET_21 = {
    "single_bus":{
        "trigger": "V8_single_bus_trigger.csv",
        "channel": "V8_single_bus_trigger.csv"
    },
    "device": {
        "trigger": "TEK00106.csv",
        "channel": "TEK00105.csv"
    }
}

FOLDER_PATH_22 = "./9_4_2026_V8_single_bus/"
DEVICE_SET_22 = {
    "device": {
        "trigger": "V8_single_bus_trigger.csv",
        "channel": "V8_single_bus_channel.csv"
    }
}

FOLDER_PATH_23 = "./9_4_2026_V8_ring2_drop/"
DEVICE_SET_23 = {
    "device": {
        "trigger": "TEK00110.csv",
        "channel": "TEK00109.csv"
    }
}

# V5 First Single Ring Through
FOLDER_PATH_24 = "./vernier_4_14_2026/"
DEVICE_SET_24 = {
    "single_bus":{
        "trigger": "V8_single_bus_trigger.csv",
        "channel": "V8_single_bus_channel.csv"
    },
    "V5_1st_single_ring_through":{
        "trigger": "TEK00008.csv",
        "channel": "TEK00009.csv"
    },
}

# V10 Second Single Ring Through
FOLDER_PATH_25 = "./vernier_4_14_2026/"
DEVICE_SET_25 = {
    "single_bus":{
        "trigger": "V10_single_bus_trigger.csv",
        "channel": "V10_single_bus_trigger.csv"
    },
    "V5_1st_single_ring_through":{
        "trigger": "TEK00090.csv",
        "channel": "TEK00091.csv"
    },
}

record = pd.read_csv('record.csv', on_bad_lines='warn')

def case_1():
    START_INDEX = 1260
    END_INDEX = 1360
    subset = ["single_bus", "output_5"]
    LENGTH = 27000 * 2 * math.pi
    data_tensor = package_data(DEVICE_SET_2, DATA_LENGTH, FOLDER_PATH_2, subset=subset)
    print("Data tensor shape:", data_tensor.shape)
    basis = moving_average(data_tensor[0], window_size=10000)
    print("basis shape:", basis.shape)
    normalized_data_tensor = normalize_data(data_tensor, basis, dB=False)
    print("Normalized data tensor shape:", normalized_data_tensor.shape)
    downsampled_data_tensor = downsample_data(normalized_data_tensor, points=10)
    print("Downsampled data tensor shape:", downsampled_data_tensor.shape)
    multi_plot(downsampled_data_tensor[1:], start_index=START_INDEX, end_index=END_INDEX)

    TESTING_ID = 1
    REVERSE = False
    testing_array = downsampled_data_tensor[TESTING_ID]
    if REVERSE:
        testing_array = np.max(testing_array) - testing_array
    fwhm, fsr, q_factor_mode, q_factor, q_factor_std = calculate_fwhm_whole(testing_array,peak_number=0, distance=3000, start_index=START_INDEX, end_index=END_INDEX, length=LENGTH)
    print("fwhm mode:", round(fwhm, 5))
    print("fsr mode:", round(fsr,5))
    print("Q factor mode:", round(q_factor_mode, 5))
    print("Q factor average:", round(q_factor, 5))
    print("Q factor std:", round(q_factor_std, 5))

# with single bus normalization
def case_2(data_tensor, device_name):
    START_INDEX = 1260
    END_INDEX = 1360

    # data_tensor = package_data(device_set, DATA_LENGTH, folder_path, subset=subset)
    # print("Data tensor shape:", data_tensor.shape)
    # basis = moving_average(data_tensor[0], window_size=1000)
    # print("basis shape:", basis.shape)
    # normalized_data_tensor = normalize_data(data_tensor, basis, dB=True)
    # print("Normalized data tensor shape:", normalized_data_tensor.shape)
    # downsampled_data_tensor = downsample_data(normalized_data_tensor, points=1)
    # print("Downsampled data tensor shape:", downsampled_data_tensor.shape)
    # multi_plot(downsampled_data_tensor[1:], start_index=START_INDEX, end_index=END_INDEX, legend=subset[1:])

    single_bus = package_data(DEVICE_SET_22, DATA_LENGTH, FOLDER_PATH_22, subset=[])
    print("Data tensor shape:", data_tensor.shape)
    basis = moving_average(single_bus[0], window_size=1000)
    print("basis shape:", basis.shape)
    normalized_data_tensor = normalize_data(data_tensor, basis, dB = False)
    print("Normalized data tensor shape:", normalized_data_tensor.shape)
    downsampled_data_tensor = downsample_data(normalized_data_tensor, points=1)
    print("Downsampled data tensor shape:", downsampled_data_tensor.shape)
    multi_save(device_name, downsampled_data_tensor, start_index=START_INDEX, end_index=END_INDEX)



# display, with self normalization, for all voltage levels
def case_3():
    START_INDEX = 1260
    END_INDEX = 1360

    # data_tensor = package_data(device_set, DATA_LENGTH, folder_path, subset=subset)
    # print("Data tensor shape:", data_tensor.shape)
    # normalized_data_tensor = normalize_each_date(data_tensor, window_size=None)
    # print("Normalized data tensor shape:", normalized_data_tensor.shape)
    # downsampled_data_tensor = downsample_data(normalized_data_tensor, points=10)
    # print("Downsampled data tensor shape:", downsampled_data_tensor.shape)

    data_tensor = package_data(device_set, DATA_LENGTH, folder_path, subset=subset)
    single_bus = package_data(DEVICE_SET_22, DATA_LENGTH, FOLDER_PATH_22, subset=[])
    print("Data tensor shape:", data_tensor.shape)
    basis = moving_average(single_bus[0], window_size=1000)
    print("basis shape:", basis.shape)
    normalized_data_tensor = normalize_data(data_tensor, basis, dB = False)
    print("Normalized data tensor shape:", normalized_data_tensor.shape)
    downsampled_data_tensor = downsample_data(normalized_data_tensor, points=10)
    print("Downsampled data tensor shape:", downsampled_data_tensor.shape)


    import matplotlib.pyplot as plt

    start_index=START_INDEX
    end_index=END_INDEX
    data = np.transpose(single_bus)
    x_points = np.arange(len(data)) * (end_index - start_index) / (len(data)) + start_index
    plt.plot(x_points, data, label=f"Single Bus")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Channel Value")
    plt.show()

    for i in range(len(subset)):
        downsampled_data_tensor[i] = downsampled_data_tensor[i] #+ b*i
        start_index=START_INDEX
        end_index=END_INDEX
        data = downsampled_data_tensor[i]
        x_points = np.arange(len(data)) * (end_index - start_index) / (len(data)) + start_index
        plt.subplot(len(subset),1,i+1)
        plt.plot(x_points, data, label=f"Device {i+1}")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Channel Value")
        plt.legend()
        plt.grid()
    plt.tight_layout()
    plt.show()
    #multi_plot(downsampled_data_tensor, start_index=START_INDEX, end_index=END_INDEX)

def Handle_data(device_set, device_name):
    START_INDEX = 1260
    END_INDEX = 1360
    device_name = device_name + '.csv'
    data_tensor = np.zeros((len(device_set), DATA_LENGTH))
    for i, device_set in enumerate(device_set):
        data_tensor[i] = package_csv_data(record, device_set, DATA_LENGTH)
    single_bus = package_data(DEVICE_SET_22, DATA_LENGTH, FOLDER_PATH_22, subset=[])
    print("Data tensor shape:", data_tensor.shape)
    basis = moving_average(single_bus[0], window_size=1000)
    print("basis shape:", basis.shape)
    normalized_data_tensor = normalize_data(data_tensor, basis, dB = False)
    print("Normalized data tensor shape:", normalized_data_tensor.shape)
    downsampled_data_tensor = downsample_data(normalized_data_tensor, points=1)
    print("Downsampled data tensor shape:", downsampled_data_tensor.shape)
    multi_save(device_name, downsampled_data_tensor, start_index=START_INDEX, end_index=END_INDEX)

# Made compatible with TMM.py (made by Jiantao)
def TMM_fitting(record, device_set, data_tensor, device_name, ring_radius_um):
    START_INDEX = 1260
    END_INDEX = 1360
    avg_df = pd.DataFrame(columns=["a", "t", "Qi", "Qc_single", "Ql"])
    for device_set in device_set:
        device_name_save = device_name + '_' + str(device_set + 1) + '.csv'
        data_tensor = package_csv_data(record, device_set, DATA_LENGTH)
        single_bus = package_data(DEVICE_SET_22, DATA_LENGTH, FOLDER_PATH_22, subset=[])
        print("Data tensor shape:", data_tensor.shape)
        basis = moving_average(single_bus[0], window_size=1000)
        print("basis shape:", basis.shape)
        normalized_data_tensor = normalize_data(data_tensor, basis, dB = True)
        print("Normalized data tensor shape:", normalized_data_tensor.shape)
        downsampled_data_tensor = downsample_data(normalized_data_tensor, points=10)
        print("Downsampled data tensor shape:", downsampled_data_tensor.shape)
        file_name = save_data_for_TMM(device_name_save, downsampled_data_tensor, start_index=START_INDEX, end_index=END_INDEX)
        results_df = TMM.fit(file_path = file_name, ring_radius_um=ring_radius_um)
        os.remove(file_name)

        # Find mean
        def safe_mean(series, positive_only=False):
            s = pd.to_numeric(series, errors="coerce")
            if positive_only:
                s = s[s > 0]
            s = s[np.isfinite(s)]
            if len(s) == 0:
                return np.nan
            return float(np.mean(s))

        avg_a = safe_mean(results_df["a"])
        avg_t = safe_mean(results_df["t"])
        avg_Qi = safe_mean(results_df["Qi"], positive_only=True)
        avg_Qc_single = safe_mean(results_df["Qc_single"], positive_only=True)
        avg_Ql = safe_mean(results_df["Ql"], positive_only=True)
        avg_df = pd.concat([avg_df, pd.DataFrame([{"a": avg_a, "t": avg_t, "Qi": avg_Qi, "Qc_single": avg_Qc_single, "Ql": avg_Ql}])], ignore_index=True)
        avg_df.to_csv(f"{device_name}_TMM_results.csv", index=False)



# To see Vernier effect
def see_drop_times_drop():
    START_INDEX = 1260
    END_INDEX = 1360
    a(20)
    data_tensor_r1d = package_data(device_set, DATA_LENGTH, folder_path, subset=subset)
    print("Data tensor r1d shape:", data_tensor_r1d.shape)
    a(23)
    data_tensor_r2d = package_data(DEVICE_SET_23, DATA_LENGTH, FOLDER_PATH_23, subset=subset)
    for _ in range(len(data_tensor_r1d)-1):
        data_tensor_r2d = np.append(data_tensor_r2d, data_tensor_r2d[0].reshape(1, -1), axis=0)
    data_tensor = data_tensor_r1d * data_tensor_r2d
    #data_tensor = np.append(data_tensor_r1d, data_tensor_r2d, axis=0)

    print("Data tensor shape:", data_tensor.shape)
    normalized_data_tensor = normalize_each_date(data_tensor, window_size=None)
    print("Normalized data tensor shape:", normalized_data_tensor.shape)
    downsampled_data_tensor = downsample_data(normalized_data_tensor, points=10)
    print("Downsampled data tensor shape:", downsampled_data_tensor.shape)
    '''
    import matplotlib.pyplot as plt
    for i in range(int(downsampled_data_tensor.shape[0])) : #/2
        downsampled_data_tensor[i] = downsampled_data_tensor[i] + b*i
        #downsampled_data_tensor[int(downsampled_data_tensor.shape[0]/2)+i] = downsampled_data_tensor[int(downsampled_data_tensor.shape[0]/2)+i] + b*i
    '''
    multi_plot(downsampled_data_tensor, start_index=START_INDEX, end_index=END_INDEX)

# To find the required data
def find_row_index(
        df,
        device, sub_device, port, EO_ring_1, 
        device_col='device', sub_device_col='sub-device', port_col='port', EO_ring_1_col='EO (ring 1)'
        ):
        """
        Find the index of the row in a CSV file that matches exactly the given device_name, device_port, and voltage.
        
        Parameters:
        df (tensor): data record.
        
        
        Returns:
        int: The index of the matching row (0-based). Raises ValueError if no match or multiple matches.
        """
        mask = (df[device_col] == device) & (df[sub_device_col] == sub_device) & (df[port_col] == port) & (df[EO_ring_1_col] == EO_ring_1)
        matching_indices = df[mask].index
        # if len(matching_indices) == 0:
        #     raise ValueError("No row matches the given criteria.")
        # elif len(matching_indices) > 1:
        #     raise ValueError("Multiple rows match the given criteria.")
        return matching_indices[0]


if __name__ == "__main__":
    def a(a):
        global subset, device_set, folder_path, b
        if a == 1:
            # all devices drop port
            subset = ["device_05_drop", "device_06_drop", "device_07_drop", "device_08_drop", "device_09_drop", "device_10_drop"]
            device_set = DEVICE_SET_15
            folder_path = FOLDER_PATH_15
            b = 1
        if a == 2:
            # all devices through port
            subset = ["device_04_through", "device_05_through", "device_06_through", "device_07_through", "device_08_through", "device_10_through"]
            device_set = DEVICE_SET_15
            folder_path = FOLDER_PATH_15
            b = 2
        if a == 3:
            subset = ["voltage_00", "voltage_10", "voltage_15", "voltage_20", "voltage_25"]
            device_set = DEVICE_SET_16
            folder_path = FOLDER_PATH_16
            b = 0
        if a == 4:
            subset = ["voltage_00", "voltage_10", "voltage_15", "voltage_25", "voltage_30"]
            device_set = DEVICE_SET_14
            folder_path = FOLDER_PATH_14
            b = 1
        if a == 5:
            subset = ["voltage_05", "voltage_06", "voltage_07", "voltage_08", "voltage_09", "voltage_10", "voltage_11", "voltage_12", "voltage_13", "voltage_14", "voltage_15"]
            device_set = DEVICE_SET_8
            folder_path = FOLDER_PATH_8
            b = 0.2
        if a == 6:
            # V8 total Vernier through port, with EO tuning
            subset = ["voltage_00", "voltage_05", "voltage_10", "voltage_15", "voltage_20", "voltage_25", "voltage_30"]
            device_set = DEVICE_SET_17
            folder_path = FOLDER_PATH_17
            b = 1
        if a == 7:
            # V8 total Vernier drop port, with thermal tuning
            subset = ["temp_20.2", "temp_22.5"]
            device_set = DEVICE_SET_18
            folder_path = FOLDER_PATH_18
            b = 1
        if a == 8:
            # V8 total Vernier drop port, with EO tuning
            subset = ["voltage_00", "voltage_05", "voltage_10", "voltage_15", "voltage_20", "voltage_25", "voltage_30"]
            device_set = DEVICE_SET_19
            folder_path = FOLDER_PATH_19
            b = 1
        if a == 20:
            # V8 ring 1 drop port, with EO tuning
            subset = ["voltage_00", "voltage_05", "voltage_10", "voltage_15", "voltage_20", "voltage_25", "voltage_30"]
            device_set = DEVICE_SET_20
            folder_path = FOLDER_PATH_20
            b = 1
        if a == 21:
            # ########################## TMM ##########################
            # V8 ring 1 through port
            subset = []
            device_set = DEVICE_SET_21
            folder_path = FOLDER_PATH_21
            b = 1
        if a == 22:
            # V8 single bus
            subset = ["device"]
            device_set = DEVICE_SET_22
            folder_path = FOLDER_PATH_22
            b = 1
        if a == 23:
                # V8 ring 2 drop port
            subset = ["device"]
            device_set = DEVICE_SET_23
            folder_path = FOLDER_PATH_23
            b = 1
        if a == 24:
            # V5 First Single Ring Through
            subset = []
            device_set = DEVICE_SET_24
            folder_path = FOLDER_PATH_24
        if a == 25:
            # V5 Second Single Ring Through
            subset = ["V5_1st_single_ring_through"]
            device_set = DEVICE_SET_25
            folder_path = FOLDER_PATH_25
        data_tensor = package_data(device_set, DATA_LENGTH, folder_path, subset=subset)
  
    # device_set = [
    #     ['V1', 'single_ring_1', 'through', 0],
    #     #['V2', 'single_ring_1', 'through', 0],
    #     ['V3', 'single_ring_1', 'through', 0],
    #     #['V4', 'single_ring_1', 'through', 0],
    #     #['V5', 'single_ring_1', 'through', 0], # data is short
    #     ['V6', 'single_ring_1', 'through', 0],
    #     ['V7', 'single_ring_1', 'through', 0],
    #     ['V8', 'single_ring_1', 'through', 0],
    #     ['V9', 'single_ring_1', 'through', 0],
    #     #['V10', 'single_ring_1', 'through', 0],
    #     ['V5', 'single_ring_2', 'through', 0],
    #     #['V10', 'single_ring_2', 'through', 0],
    # ]

    # for device_set in device_set:
    #     index = find_row_index(record, device_set[0], device_set[1], device_set[2], device_set[3])
    #     #index_sb = find_row_index(record, device_set[0], device_set[1], device_set[2], device_set[3])
    #     TMM_fitting(record, index, data_tensor, '_'.join([device_set[0], device_set[1], device_set[2], str(device_set[3])]) + '.csv', ring_radius_um=27)

    # which ones to see
    def see_data(index):
        if index == 1:
            # 23/4/2026 TEC and TO
            device_name = 'TO_TEC_coefficients'
            device_set = list(range(50-1,61))
            Handle_data(device_set, device_name)
        if index == 2:
            # 23/4/2026 TEC and TO TMM Fitting
            device_name = 'TO_TEC_coefficients_TMM'
            device_set = list(range(50-1,61))
            results_df = TMM_fitting(record, device_set, None, device_name, ring_radius_um=27)
        if index == 3:
            # 4/14/2026 V8 EO
            device_name = 'V8_EO_coefficients'
            device_set = list(range(10-1,15))
            Handle_data(device_set, device_name)
        if index == 4:
            # 4/14/2026 V8 EO TMM Fitting
            device_name = 'V8_EO_coefficients_TMM'
            device_set = list(range(10-1,15))
            results_df = TMM_fitting(record, device_set, None, device_name, ring_radius_um=27)

    see_data(3)


    # a(25)
    # case_3()