# v0.0.1-a
import matplotlib.pyplot as plt
import numpy as np

# Load the data
data = np.loadtxt('sensor_data.csv', delimiter=',')

# Get the timestamp values
timestamps = data[1:, -1]

# Get the MPU data
mpu_data = data[1:, :6]

# Get the flex sensor data
flex_data = data[1:, 6:-1]

# Create a line chart of the MPU data
plt.plot(timestamps, mpu_data)
plt.xlabel('Timestamp')
plt.ylabel('Acceleration / Gyroscope (m/s^2 / deg/s)')
plt.legend(['mpu1_ax', 'mpu1_ay', 'mpu1_az', 'mpu1_gx', 'mpu1_gy', 'mpu1_gz'], loc='upper left')
plt.show()

# Create a scatter plot of the flex sensor data
plt.scatter(timestamps, flex_data)
plt.xlabel('Timestamp')
plt.ylabel('Flex Sensor Value')
plt.legend(['flex1', 'flex2', 'flex3', 'flex4', 'flex5', 'flex6'], loc='upper left')
plt.show()