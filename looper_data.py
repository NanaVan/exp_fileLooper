#
# Multiprocessing looper for .tdms
#
# (2025) NanaVan@github
# modified from looper.py by (2025) xaratustrah@github
#

import os, time, multiprocessing, pickle, tomli, argparse
from pathlib import Path
from loguru import logger
from functools import partial
from preprocessing import Preprocessing
from psd_array import *
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Declare the global variable
PROCESSED_FILES = set()

# font settings for plot
font = {"weight": "bold", "size": 12}  #'family' : 'normal',
plt.rc("font", **font)

def read_and_verify_settings(toml_file):
    """
    Read and validate settings from a TOML file using a required_keys structure.
    """
    try:
        # Define required sections and keys
        required_keys = {
            "paths": [
                "monitor_dir",
                "state_file",
            ],
            "processing": ["num_cores", "interval_seconds", "file_ready_seconds"],
            "analysis": ["iq_per_frame", "n_average", "todo"],
        }

        # Load the TOML file
        with open(toml_file, "rb") as file:
            config = tomli.load(file)

        # Validate required sections and keys
        for section, keys in required_keys.items():
            if section not in config:
                raise KeyError(f"Missing section: {section}")
            for key in keys:
                if key not in config[section]:
                    raise KeyError(f"Missing key: {key} in section: {section}")

        logger.info("Settings successfully read and validated.")
        return config

    except KeyError as e:
        logger.error(f"Configuration error: {e}")
        print(f"Configuration error: {e}")  # Ensure visibility in console
        raise
    except Exception as e:
        logger.error(f"Error reading or validating settings: {e}")
        print(f"Error: {e}")
        raise

# Load processed files state from disk
def load_processed_files(state_file):
    global PROCESSED_FILES
    if os.path.exists(state_file):
        with open(state_file, "rb") as file:
            PROCESSED_FILES = pickle.load(file)
        logger.info(f"Loaded {len(PROCESSED_FILES)} processed files from state.")


# Save processed files state to disk
def save_processed_files(state_file):
    with open(state_file, "wb") as file:
        pickle.dump(PROCESSED_FILES, file)
    logger.info("Processed files state saved.")


# Check if file is ready for processing
def is_file_ready(filepath, settings):
    '''
    check if file is ready by confirming size stablilizes and reaches approx size threshold
    '''
    approx_size_bytes = settings["processing"]["approx_file_size_mb"] * 1024 * 1024
    threshold_ratio = 0.80
    check_interval = settings["processing"]["file_ready_seconds"]

    prev_size = -1
    while True:
        try:
            current_size = os.path.getsize(filepath)
        except Exception as e:
            # file might not be accessible yet, wait and retry
            time.sleep(check_interval)
            continue

        if current_size >= approx_size_bytes * threshold_ratio:
            if current_size == prev_size:
                print("file {:} is transferred completed!".format(Path(filepath).stem))
                return True
            else:
                prev_size = current_size
        time.sleep(check_interval)
        print("file {:} is waiting to transfer completed! (file size: {:.2f} MB)".format(Path(filepath).stem, current_size/1024/1024))


# Process the file
def process_file(filename, settings):
    win_len = settings["analysis"]["iq_per_frame"]
    n_average = settings["analysis"]["n_average"]
    monitor_dir = settings["paths"]["monitor_dir"]
    output_dir = settings["paths"]["output_dir"]
    todo = settings["analysis"]["todo"]

    start_time = time.time()

    bud = Preprocessing(monitor_dir + filename)
    x_frequency, y_time, z_psd_array, _ = psd_array_welch(bud, offset=0, window_length=win_len, n_average=n_average, overlap_ratio=0, n_frame=-1, n_hop=0, padding_ratio=0, window='kaiser', beta=14)
    if 'data_spectrogram' in todo:
        np.savez(output_dir + filename + "_spectrogram.npz", x_frequency + bud.center_frequency, y_time, z_psd_array)
    if 'png_spectrogram' in todo:
        norm = colors.LogNorm(vmin=z_psd_array.min(), vmax=z_psd_array.max())
        fig, ax = plt.subplots(figsize=(12,10))
        waterfall = ax.pcolormesh(
                    x_frequency * 1e-3, # [kHz]
                    y_time * 1e3, # [ms]
                    z_psd_array,
                    shading = 'flat', cmap = 'viridis', norm = norm)
        ax.set_title('Waterfall Plot | File: {:},\nRecording from {:}, duration: {:.3f} ms'.format(filename, bud.date_time.astype('datetime64[ms]'), bud.n_sample/bud.sampling_rate*1e3))
        ax.set_xlabel('Center frequency {:} MHz [kHz]'.format(bud.center_frequency*1e-6))
        ax.set_ylabel('Time [ms]')
        ax.set_xlim((-bud.span*1e-3/2, bud.span*1e-3/2))
        cbar = fig.colorbar(waterfall, ax=ax)
        cbar.set_label('Power Spectral Density [arb. unit]')
        plt.tight_layout()
        plt.savefig(output_dir + filename + '_spectrogram.png')
        plt.close()
    if 'data_spectrum': 
        np.savez(output_dir + filename + "_spectrum.npz", x_frequency[:-1] + bud.center_frequency, np.mean(z_psd_array, axis=0))
    if 'png_spectrum':
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(x_frequency[:-1]*1e-3, np.mean(z_psd_array, axis=0))
        ax.set_yscale('log')
        ax.set_title('Averaged Spectrum | File: {:},\nRecording from {:}, duration: {:.3f} ms'.format(filename, bud.date_time.astype('datetime64[ms]'), bud.n_sample/bud.sampling_rate*1e3))
        ax.set_xlabel('Center frequency {:} MHz [kHz]'.format(bud.center_frequency*1e-6))
        ax.set_ylabel('Power Spectral Density [arb. unit]')
        ax.set_xlim((-bud.span*1e-3/2, bud.span*1e-3/2))
        ax.grid(True, which='both', ls='--', alpha=0.5)
        plt.savefig(output_dir + filename + '_spectrum.png', transparent=False)
        plt.close()

    end_time = time.time()  # Record end time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    logger.info(f"Finished processing {filename} in {elapsed_time:.2f} seconds.")

# Monitor and process files
def monitor_directory(settings):
    global PROCESSED_FILES

    state_file = settings["paths"]["state_file"]
    monitor_dir = settings["paths"]["monitor_dir"]
    num_cores = settings["processing"]["num_cores"]
    interval_seconds = settings["processing"]["interval_seconds"]

    load_processed_files(state_file)  # Load state at startup
    try:
        while True:
            files = [f for f in os.listdir(monitor_dir) if f.lower().endswith(".tdms")]
            unprocessed_files = [f for f in files if f not in PROCESSED_FILES]

            # Check for files that are ready to process
            ready_files = []
            for file in unprocessed_files:
                filepath = os.path.join(monitor_dir, file)
                if os.path.isfile(filepath) and is_file_ready(filepath, settings):
                    ready_files.append(file)

            if ready_files:
                logger.info(f"Files to process: {ready_files}")

                # Prepare the partial function with the additional argument
                process_file_partial = partial(process_file, settings=settings)

                # Process files using multiprocessing pool
                with multiprocessing.Pool(num_cores) as pool:
                    pool.map(process_file_partial, ready_files)

                # Update the list of processed files
                PROCESSED_FILES.update(ready_files)

                # Save state after processing files
                save_processed_files(state_file)

            time.sleep(interval_seconds)  # Monitor at regular intervals
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
        save_processed_files(state_file)  # Save state on exit

def main():
    # Setup argument parser for command-line arguments
    parser = argparse.ArgumentParser(
        description="Monitor a directory and process files."
    )
    parser.add_argument(
        "--config", required=True, help="Path to the TOML configuration file."
    )
    args = parser.parse_args()

    logger.add(
        "processing.log",
        format="{time} {level} {message}",
        level="INFO",
        rotation="1 MB",
    )

    # Load settings from the provided TOML file
    settings = read_and_verify_settings(args.config)
    monitor_directory(settings)


# ----------------------------

if __name__ == '__main__':
    main()
