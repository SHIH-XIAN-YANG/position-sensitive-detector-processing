
import csv
import matplotlib.pyplot as plt
import numpy as np

# Open the CSV file
with open('./data/tri_blink_signal_1Hz_2ms.Dat', 'r') as file:

    # Create a CSV reader object
    csv_reader = csv.reader(file)
    
    time = []
    x = []
    y = []
    sigma = []
    diffs = []
    diffs2 = []
    
    
    i=0
    ts = 0.002
    # Iterate over each row in the CSV file
    for row in (csv_reader):
        # Assuming each row has 4 columns
        # if float(row[3])>1:

            time.append(float(row[0]))
            x.append(float(row[1]))
            y.append(float(row[2]))
            sigma.append(float(row[3]))

            i+=1

    diffs = np.diff(sigma)
    diffs = np.insert(diffs, 0, diffs[0])
    diffs2 = np.diff(diffs)
    diffs2 = np.insert(diffs2, 0, diffs2[0])


# Plotting the data
# time = time[1500:1600]
# x = x[1500:1600]
# y = y[1500:1600]
# sigma = sigma[1500:1600]
# diff = diff[1500:1600]
x_temp = []
y_temp = []
sigma_temp = [] 
time_temp = []     

for i, diff in enumerate(diffs2):
    if diff > 0:
        x_temp.append(x[i])
        y_temp.append(y[i])
        sigma_temp.append(sigma[i])
        time_temp.append(time[i])

start = 800
end = 1200
time = time[start:end]
x = x[start:end]
y = y[start:end]
sigma = sigma[start:end]
diffs = diffs[start:end]
diffs2 = diffs2[start:end]

# x = []
# y = []
# sigma = []
# x = x_temp.copy()
# y = y_temp.copy()
# sigma = sigma_temp.copy()
# time = time_temp.copy()
# print(len(x), len(y), len(sigma))
# Create subplots
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 8), sharex=True)

# Plot X data
ax1.plot(time, x, label='X', color='r')
ax1.set_ylabel('X Value')
ax1.set_title('Data Plots')
ax1.grid(True)
ax1.legend()

# Plot Y data
ax2.plot(time, y, label='Y', color='g')
ax2.set_ylabel('Y Value')
ax2.grid(True)
ax2.legend()

# Plot Sigma data
ax3.plot(time, sigma, label='Sigma', color='b')
ax3.set_xlabel('Time')
ax3.set_ylabel('Sigma Value')
ax3.grid(True)
ax3.legend()

# Plot diff data
ax4.plot(time, diffs, label='Sigma', color='k')
ax4.set_xlabel('Time')
ax4.set_ylabel('diff Value')
ax4.grid(True)
ax4.legend()

# Plot diff data
ax5.plot(time, diffs2, label='Sigma', color='k')
ax5.set_xlabel('Time')
ax5.set_ylabel('diff2 Value')
ax5.grid(True)
ax5.legend()

plt.tight_layout()
# plt.show()


plt.figure(figsize=(8, 6))
scatter = plt.scatter(x, y, marker='o', label='Data', s=0.5,c=sigma, cmap='viridis', alpha=0.75)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('X vs Y Data')
plt.xlim([-2.25, 2.25])
plt.ylim([-2.25, 2.25])
plt.grid(True)
plt.legend()

# Add annotations for indicating the color scale
# Add a color bar indicating the magnitude scale
cbar = plt.colorbar(scatter)
cbar.set_label('Magnitude')

plt.show()
