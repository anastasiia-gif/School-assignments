import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress

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

# Set the start and end times for analysis
start_time = 500  # seconds
end_time = 1800  # seconds

# Filter the data for the specified time range
data_window = data[(data["Time"] >= start_time) & (data["Time"] <= end_time)]

# Option to toggle Gaussian smoothing
apply_smoothing = True
sigma = 500  # Standard deviation for Gaussian filter, adjust as necessary

if apply_smoothing:
    data_window["Smoothed Strain"] = gaussian_filter1d(
        data_window["Strain"], sigma=sigma
    )
    plot_label = "Smoothed Strain"
else:
    data_window["Smoothed Strain"] = data_window["Strain"]
    plot_label = "Original Strain"


# Linear regression model (y = a * t + b)
def linear_model(t, a, b):
    return a * t + b


# Perform linear regression on the data using scipy's linregress
slope, intercept, _, _, _ = linregress(
    data_window["Time"], data_window["Smoothed Strain"]
)

# Generate fitted values from the linear regression model
time_smooth = np.linspace(data_window["Time"].min(), data_window["Time"].max(), 500)
strain_smooth = linear_model(time_smooth, slope, intercept)

# Plot the original and fitted strain data
plt.figure(figsize=(10, 6))
plt.plot(data_window["Time"], data_window["Strain"], label="Original Strain", alpha=0.7)
plt.plot(
    data_window["Time"],
    data_window["Smoothed Strain"],
    label=plot_label,
    color="red",
    lw=2,
)
plt.plot(time_smooth, strain_smooth, label="Linear Regression Fit", color="blue", lw=2)

# Title and labels
plt.title(
    f"Strain vs. Time (Linear Regression, {start_time}s to {end_time}s)", fontsize=14
)
plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("Strain", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Output the linear regression parameters
print(f"Linear Regression Parameters: Slope (a) = {slope}, Intercept (b) = {intercept}")
