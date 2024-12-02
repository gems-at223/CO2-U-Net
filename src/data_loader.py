import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class ReservoirDataset(Dataset):
    def __init__(self, directory, verbose=True):
        self.inputs = []
        self.targets = []
        self.load_and_process_data(directory, verbose)

    def load_and_process_data(self, directory, verbose):
        file_list = glob.glob(os.path.join(directory, "*.npz"))
        target_shape = (96, 200)  # Shape for input features
        target_output_shape = (96, 200, 24)  # Shape for target (padded to this shape)
        file_count = 0
        for file_name in file_list:
            if file_count % 10 == 0 and verbose:
                print(
                    f"Processing {file_name}: {file_count / len(file_list) * 100:.2f}%"
                )
            file_count += 1

            with np.load(file_name, allow_pickle=True) as data:
                input_features = []
                target = None
                perf_interval = data.get("perf_interval", None)
                if perf_interval is None or len(perf_interval) != 2:
                    print("Invalid perf_interval format. Skipping file.")
                    continue
                start, end = perf_interval

                # Create a new feature map of size (96, 200)
                feature_map = np.zeros(target_shape, dtype=int)
                feature_map[start : end + 1, :] = 1

                for var_name in sorted(data.keys()):
                    array = data[var_name]

                    if var_name == "gas_saturation":
                        target = self.pad_or_reshape_to_shape(
                            array, target_output_shape
                        )
                    else:
                        if np.isscalar(array) or np.ndim(array) == 0:
                            array = self.create_filled_array(array, target_shape)
                        elif np.ndim(array) == 2:
                            array = self.pad_to_shape(array, target_shape)
                        else:
                            continue
                        input_features.append(torch.from_numpy(array).float())

                input_features.append(torch.from_numpy(feature_map).float())

                if target is not None:
                    combined_inputs = torch.stack(input_features, dim=0)
                    self.inputs.append(combined_inputs)
                    self.targets.append(target)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inputs = self.inputs[idx]
        targets = self.targets[idx]

        return inputs, targets

    @staticmethod
    def pad_to_shape(arr, target_shape, fill_value=0):
        pad_width = [
            (0, max(0, target - arr.shape[i])) for i, target in enumerate(target_shape)
        ]
        padded_arr = np.pad(arr, pad_width, mode="constant", constant_values=fill_value)
        return padded_arr

    @staticmethod
    def pad_or_reshape_to_shape(arr, target_shape, fill_value=0):
        if arr.shape == target_shape:
            return arr
        elif arr.shape[0] < target_shape[0]:
            pad_width = [(0, target_shape[0] - arr.shape[0])] + [(0, 0)] * (
                len(target_shape) - 1
            )
            return np.pad(arr, pad_width, mode="constant", constant_values=fill_value)
        else:
            return arr[: target_shape[0], ...]

    @staticmethod
    def create_filled_array(scalar, target_shape):
        filled_array = np.full(target_shape, scalar, dtype=np.float32)
        return filled_array
