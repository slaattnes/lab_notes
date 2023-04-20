# v0.0.1-a
import time
import numpy as np
import board
import busio
import adafruit_mpu6050
import RPi.GPIO as GPIO
from adafruit_ads1x15.ads1015 import ADS1015
from adafruit_ads1x15.analog_in import AnalogIn


class SensorData:
    def __init__(self, mpu_sample_rate=100, mpu_num_samples=1000, flex_window_size=10):
        # initialize i2c bus and sensors
        i2c = busio.I2C(board.SCL, board.SDA)
        self.mpu1 = adafruit_mpu6050.MPU6050(i2c, address=0x68)
        self.mpu2 = adafruit_mpu6050.MPU6050(i2c, address=0x69)

        # initialize GPIO and flex sensors
        GPIO.setmode(GPIO.BCM)
        self.ads1 = ADS1015(i2c)
        self.ads2 = ADS1015(i2c, address=0x49)
        self.chan0 = AnalogIn(self.ads1, ADS1015.P0)
        self.chan1 = AnalogIn(self.ads1, ADS1015.P1)
        self.chan2 = AnalogIn(self.ads1, ADS1015.P2)
        self.chan3 = AnalogIn(self.ads1, ADS1015.P3)
        self.chan4 = AnalogIn(self.ads2, ADS1015.P0)
        self.chan5 = AnalogIn(self.ads2, ADS1015.P1)

        # define parameters
        self.mpu_sample_rate = mpu_sample_rate  # Hz
        self.mpu_num_samples = mpu_num_samples  # number of samples to collect for MPU
        self.flex_window_size = flex_window_size  # window size for rolling window preprocessing on flex sensors

        # initialize data array
        num_sensors = 14
        max_samples = self.mpu_num_samples
        self.data = np.empty((max_samples, num_sensors+1))

        # add header to data array
        self.data[0, :-1] = ['mpu1_ax', 'mpu1_ay', 'mpu1_az', 'mpu1_gx', 'mpu1_gy', 'mpu1_gz',
                             'mpu2_ax', 'mpu2_ay', 'mpu2_az', 'mpu2_gx', 'mpu2_gy', 'mpu2_gz',
                             'flex1', 'flex2', 'flex3', 'flex4', 'flex5', 'flex6']
        self.data[0, -1] = 'timestamp'

    def collect_data(self):
        # collect and preprocess data
        i = 1
        flex_samples = []
        mpu1_samples = []
        mpu2_samples = []
        while i <= self.mpu_num_samples:
            # collect data from MPU6050 sensors
            mpu1_data = self.mpu1.acceleration + self.mpu1.gyro
            mpu2_data = self.mpu2.acceleration + self.mpu2.gyro

            # collect data from flex sensors
            flex_data = [self.chan0.voltage, self.chan1.voltage, self.chan2.voltage,
                         self.chan3.voltage, self.chan4.voltage, self.chan5.voltage]

            # preprocess data
            mpu1_samples.append(mpu1_data)
            mpu2_samples.append(mpu2_data)
            flex_samples.append(flex_data)

# apply rolling window to MPU data
if len(mpu1_samples) >= mpu_sample_rate:
    window_mean = np.mean(mpu1_samples[-mpu_sample_rate:], axis=0)
    window_std = np.std(mpu1_samples[-mpu_sample_rate:], axis=0)
    mpu1_samples[-mpu_sample_rate:] = (mpu1_samples[-mpu_sample_rate:] - window_mean) / window_std
    
    window_mean = np.mean(mpu2_samples[-mpu_sample_rate:], axis=0)
    window_std = np.std(mpu2_samples[-mpu_sample_rate:], axis=0)
    mpu2_samples[-mpu_sample_rate:] = (mpu2_samples[-mpu_sample_rate:] - window_mean) / window_std

# apply rolling window to flex data
if len(flex_samples) >= flex_window_size:
    window_mean = np.mean(flex_samples[-flex_window_size:], axis=0)
    window_std = np.std(flex_samples[-flex_window_size:], axis=0)
    flex_samples[-flex_window_size:] = (flex_samples[-flex_window_size:] - window_mean) / window_std

    # add data to data array and increment counter
    timestamp = time.monotonic()
    data[i, :12] = mpu1_samples[-1] + mpu2_samples[-1]
    data[i, 12:18] = flex_samples[-1]
    data[i, -1] = timestamp
    i += 1

# stop program if user presses Ctrl-C
except KeyboardInterrupt:
    print('\n\nKeyboard interrupt detected. Stopping program.\n')
    break

# save data to file
np.savetxt('data.csv', data, delimiter=',', fmt='%s')
print('\nData saved to file data.csv.\n')