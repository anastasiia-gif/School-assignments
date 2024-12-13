import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Load the CSV file (use the correct path to your file)
file_path = "csv_data.csv"  # Update this path to your file location

# Load data with semicolon delimiter (ensure correct parsing)
data = pd.read_csv(
    file_path, delimiter=";", header=None, names=["Force", "Strain", "Time"]
)

# Clean up the Force, Strain, and Time columns by replacing commas with periods
data["Force"] = data["Force"].str.replace(",", ".").str.strip(".").astype(float)
data["Strain"] = data["Strain"].str.replace(",", ".").str.strip(".").astype(float)
data["Time"] = data["Time"].str.replace(",", ".").str.strip(".").astype(float)

# Check if there are any NaN values in the Force, Strain, or Time columns
print("Checking for NaN values in data:")
print(data.isna().sum())

# Remove rows where 'Strain' is NaN (or Force if necessary)
data = data.dropna(subset=["Strain"])

# Filter data for Time > 1800 seconds
data_after_1800 = data[data["Time"] > 1800]

# Sort data by time to avoid issues with non-increasing time values
data_after_1800 = data_after_1800.sort_values(by="Time")

# Apply Gaussian smoothing to the Strain data after 1800 seconds
sigma = 500  # Standard deviation for Gaussian filter (adjust as necessary)
data_after_1800["Smoothed Strain"] = gaussian_filter1d(
    data_after_1800["Strain"], sigma=sigma
)

# Plot the original vs smoothed strain data (only after 1800 seconds)
plt.figure(figsize=(10, 6))
plt.plot(data["Time"], data["Strain"], label="Original Strain", alpha=0.7)
plt.plot(
    data_after_1800["Time"],
    data_after_1800["Smoothed Strain"],
    label="Gaussian Smoothed Strain",
    color="red",
    lw=2,
)
plt.title("Creep and Recovery Over Time (Smoothed After 1800 seconds)", fontsize=14)
plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("Strain", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
