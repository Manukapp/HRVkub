import csv
import itertools
from collections import Counter
import statistics
import numpy as np
from numpy import ndarray
import statsmodels.graphics.tsaplots as sm
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
from scipy import signal

#This code attempts to do HRV calculations on HR data csv files.
#Next code would be to extract from csv results files, and create a new csv table with all info

# Set the path to the directory containing the files
directory = "C:\\Users\\lm9172\\OneDrive - Anglia Ruskin University\\HR_data\\"


def extract_sample(filename, start_time, sample_duration, sample_title):
    # Open csv file for reading

    total_RR = []
    total_HR = []
    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file)
        # skip to row 7, i.e. where data starts, i.e. skipping Header information
        for i in range(6):
            next(reader)

        # Convert sample interval time into seconds
        start_seconds = sum(x * int(t) for x, t in zip([3600, 60, 1], start_time.split(":")))
        end_seconds: int = start_seconds + sum(x * int(t) for x, t in zip([3600, 60, 1], sample_duration.split(":")))
        print("this is the start interval :", start_seconds)
        print("this is the end interval :", end_seconds)

        # Go to sample's starting row_index in columns
        for _ in itertools.islice(reader, start_seconds):
            pass

        sample = []

        end_index = None
        # EXtract sample HR, and RR data
        for row_index, row in enumerate(reader):
            if not row or len(row) < 2:
                continue

            # print(row[1])
            time = row[1]

            # Convert timing column into seconds
            time_seconds = sum(x * int(t) for x, t in zip([3600, 60, 1], time.split(":")))
            print(time_seconds)

            if time_seconds < start_seconds:
                continue

            # identify interval and extract HR and RR only within this interval
            elif time_seconds >= start_seconds and time_seconds <= end_seconds:
                # Extracting Sample data
                HR = int(row[2])

                total_HR.append(HR)
                print(total_HR)

                RR = float(row[6]) / 1000  # RR in seconds
                total_RR.append(RR)
                print(total_RR)

                """Because this data is from HR (BPM) transformation to RR interval directly, it is an indirect measure of R to R peaks, as calculated in Tarvainen, et al, 2002"""


            else:
                break

        ### Sample calculations ###

        def gaussian_basis(z, mu, sigma):
            """Defines a Gaussian basis function for the basis vectors of H"""
            return np.exp(-(z - mu) ** 2 / (2 * sigma ** 2))

        def detrend_RR(RR, sigma=0.1, lambda_val=5):
            """
            Detrends the given RR interval data using a Gaussian basis function.

            Parameters:
            total_RR (numpy.ndarray): An array containing the RR interval data
            sigma (float): The width of the Gaussian basis function (default=0.1)
            lambda_val (float): The regularization or penalty term that controls the smoothing (default=10)

            Returns:
            numpy.ndarray: The stationary component of the RR interval data
            """
            N = len(RR)
            z = np.array(RR).reshape(-1, 1)  # Data reshaped as column vector

            # Construct observation matrix H with N-1 rows and N columns, with mutually orthonormal basis vectors psi_i as columns.
            psi = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    if j == 0:
                        psi[i, j] = gaussian_basis(z[j], z[i], sigma)
                    elif j == N:
                        psi[i, j] = -gaussian_basis(z[j], z[i], sigma)
                    else:
                        try:
                            psi[i, j] = gaussian_basis(z[j], z[i], sigma) - gaussian_basis(z[j], z[i + 1], sigma)
                        except IndexError:
                            var = i + 1 == None
            H = psi[:, :]  # removing the first row of the psi matrix

            'Centre observation matrix H'

            H = H - np.mean(H, axis=0)

            # Check if columns of psi are orthogonal
            inner_products = np.dot(psi.T, psi)
            np.fill_diagonal(inner_products, 0)
            max_inner_product = np.max(np.abs(inner_products))
            print(f"Maximum inner product (excluding diagonal): {max_inner_product}\n"
                  f"If maximum inner product is close to 0, columns of psi are almost orthogonal. "
                  f"If significantly greater than 0, then not")

            # Compute the second-order difference matrix, which smoothes the trend line
            D2 = np.diag([1] * (N - 2), k=-1) + np.diag([-2] * (N - 1), k=0) + np.diag([1] * (N - 2),
                                                                                       k=1)  # In Tarvainen, et al, 2002, D2 is a N-3 x N-1 matrix, we differ due to reason stated above.
            # Add an extra row of zeros to the bottom of D2 & add an extra column of zeros to the right of D2 to have equal dimensions between matrices for dot products further
            D2 = np.pad(D2, [(0, 1), (0, 0)], mode='constant')
            D2 = np.pad(D2, [(0, 0), (0, 1)], mode='constant')

            print("this is D2 shape", np.shape(D2))
            # Compute the stationary component using the observation matrix H and the second-order difference matrix D2
            HT = np.transpose(H)
            HTH = np.dot(HT, H)

            HTDTdDdH = np.dot(HT, np.dot(D2.T, D2))

            I = np.identity(N)

            reg = lambda_val **2 * np.dot(D2.T, D2)
            A = I + reg

            #U, s, V = np.linalg.svd(HTDTdDdH)
            """Testing singularity of matrix: singularity value decomposition"""
            #print(s)

            print("The other matrices", np.shape(H), np.shape(HTH), np.shape(HTDTdDdH), np.shape(A), np.shape(reg),
                  np.shape(z))

            #z_stat = np.dot(np.linalg.pinv(A), z - np.dot(H, np.linalg.solve(HTH + lambda_val ** 2 * HTDTdDdH, np.dot(HT, z))))
            #z_stat = np.dot(np.linalg.inv(A), np.linalg.solve(A, np.dot(HT, z)))
            z_stat = np.dot((I - np.linalg.inv(A)), z)
            z_stat = z_stat[:,0]
            print(z_stat)

            # Define high-pass filter parameters
            fs = 4  # Sampling frequency (Hz)
            nyq = 0.5 * fs
            cutoff_freq = 0.01  # Cutoff frequency (Hz)
            filter_order = 3  # Filter order

            # Create high-pass filter coefficients
            b, a = signal.butter(filter_order, cutoff_freq / nyq, btype='highpass')

            # Apply filter to RR interval data
            filtered_rr_data = signal.filtfilt(b, a, z_stat)

            """Check if detrending was successful with autocorrelation function plot"""

            # Compute ACF of z and z_stat
            acf_z = sm.acf(z, nlags=len(z) - 1)
            acf_z_stat = sm.acf(filtered_rr_data, nlags=len(filtered_rr_data) - 1)

            # Plot ACFs
            fig, ax = plt.subplots(nrows=2, figsize=(10, 10))
            plot_acf(acf_z, ax=ax[0])
            ax[0].set_title('Autocorrelation of original data')
            plot_acf(acf_z_stat, ax=ax[1])
            ax[1].set_title('Autocorrelation of detrended data')
            plt.show()
            return filtered_rr_data

        z_stationary = detrend_RR(total_RR)

        ## Stress Index ##

        # Count most common RR occurance and obtain amplitude of most frequent RR
        count_RR = Counter(z_stationary)
        most_common_RRvalue = count_RR.most_common(1)[0][0]
        most_common_RRcount = count_RR.most_common(1)[0][1]

        print("Counting RR data:", count_RR)
        print("Number of occurrences:", most_common_RRcount)

        AMo = round((round((most_common_RRcount / int(len(z_stationary))), 3) * 100), 1)

        #bin_width = 0.05  # 50ms in seconds
        #bin_edges = np.arange(min(z_stationary), max(z_stationary) + bin_width, bin_width)
        #hist, bin_edges = np.histogram(z_stationary, bins=bin_edges, density=True)

        #AMo = round(max(hist) * 100, 1)
        print("Amplitude of most frequent RR :", AMo)

        # Calculate Mode and difference between Max and Min of RR list/distribution

        mode = statistics.median(z_stationary)
        diff = max(z_stationary) - min(z_stationary)
        print("here are: ", diff, max(z_stationary), min(z_stationary), mode)

        # Stress Index formula:

        SI = AMo / (( 2 * mode) * diff)

        detrend_HR = z_stationary

        ## Mean Heart Rate ##
        meanHR = round((sum(total_HR) / int(len(total_HR))), 2)

        ## Mean Heart Rate with Kubios formula ##
        meanRR = sum(total_RR) / int(len(total_RR))
        meanHR2 = round((60 / meanRR), 2)

        sample_data = sample.append((meanHR, meanHR2, SI))

        sample_info = {'title': sample_title, 'start_time': start_time, 'sample_duration': sample_duration,
                       'data': sample}
        print(sample_info)


filename = directory + "PT-14 S01_RR.csv"
start_time = "00:03:45"
sample_duration = "00:03:00"
sample_title = "Resting State"

sample = extract_sample(filename, start_time, sample_duration, sample_title)
# print(sample)
