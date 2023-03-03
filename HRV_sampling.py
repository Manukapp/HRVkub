# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 18:39:55 2023

@author: lm9172
"""
import csv
import itertools
from collections import Counter
import statistics
import numpy as np
from numpy import ndarray

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


            else:
                break

        ### Sample calculations ###

        ## Mean Heart Rate ##
        meanHR = round((sum(total_HR) / int(len(total_HR))), 2)

        ## Mean Heart Rate with Kubios formula ##
        meanRR = sum(total_RR) / int(len(total_RR))
        meanHR2 = round((60 / meanRR), 2 )

        # RR interval series
        RR_interval_diff = []
        RRindex = len(total_RR)
        for r in range(RRindex):
            interval = total_RR[r] - total_RR[r - 1]
            RR_interval_diff.append(interval)

        print(RR_interval_diff)

        # Detrending RR data with regularized least squares solution #

            #Gathering metrics from RR_interval data

        N = len(total_RR) # length of series
        z = total_RR # following notation used in Karjalainen, et al, 1997 & Tarvainen, et al, 2002


            #Construct observation matrix H, with mutually orthonomal basis vectors psi_i as columns. A certain linear combination of psi_i represents the z without low-frequency trend

        def gaussian_basis(x, mu, sigma):
            return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) #basis function for basis vectors of H

        x = total_RR #testing with RR-difference series, but might be recommended to use original R-R series
        sigma = 0.1 #width of Gaussian basis function

        psi: ndarray = np.zeros((N - 1, N)) # because I define N as the number of elements in RR difference series
        for i in range(N - 1):
            for j in range(N):
                psi[i, j] = gaussian_basis(x[j], x[i], sigma) - gaussian_basis(x[j], x[i + 1], sigma)

        H = np.dot(psi, psi.T) #psi matrix times its transpose

        # Check if columns of psi are orthogonal
        inner_products = np.dot(psi.T, psi)
        np.fill_diagonal(inner_products, 0)
        #print("Inner product matrix:\n", inner_products)
        print("Maximum inner product (excluding diagonal):", np.max(np.abs(inner_products)), "/n If maximum inner product is close to 0, columns of psi are almost orthogonal. If significantly greater than 0, then not")

        lambda_val = 10 # regulisation or penalty term that controls the smoothing. The larger, the broader and less overfitting. This number is used in Tarvainen, et al, 2002.
        D2 = np.diag([1] * (N - 2), k=-1) + np.diag([-2] * (N - 1), k=0) + np.diag([1] * (N - 2), k=1) # Second order difference matrix, which smoothes the trend line

        HT = np.transpose(H)
        HTH = np.matmul(HT, H)
        HTDTdDdH = np.matmul(HT,np.matmul(D2.T, D2))
        theta_hat_lambda = np.matmul(np.linalg.inv(HTH + lambda_val * HTDTdDdH), np.matmul(HT, z))
        ztrend = np.matmul(H, theta_hat_lambda)

        # Compute the stationary component

        z_stat = z - ztrend



        ## Stress Index ##

        # Count most common RR occurance and obtain amplitude of most frequent RR
        count_RR = Counter(z_stat) #change to RR_interval?? to z_stat Or Total_RR
        most_common_RRvalue = count_RR.most_common(1)[0][0]
        most_common_RRcount = count_RR.most_common(1)[0][1]

        print("Most common RR value:", most_common_RRvalue)
        print("Number of occurrences:", most_common_RRcount)
        AMo = round((round((most_common_RRcount / int(len(z_stat))), 3) * 100), 1)
        print("Amplitude of most frequent RR :", AMo)

        # Calculate Mode and difference between Max and Min of RR list/distribution

        mode = statistics.median(z_stat)
        diff = max(z_stat) - min(z_stat)

        # Stress Index formula:

        SI = AMo / ((2 * mode) * diff)

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

'''
if end_seconds <= time_seconds:
            end_index = row_index
            print(end_index)
            break

    if end_index is None:
        end_index = row_index
        


  HT = np.transpose(H)
            HTH = np.dot(HT, H)
            HTDTdDdH = np.dot(HT, np.dot(D2.T, D2))
            #theta_hat_lambda = np.matmul(np.linalg.inv(HTH + lambda_val * HTDTdDdH), np.matmul(HT, z))
            theta_hat_lambda = np.linalg.solve(HTH + lambda_val ** 2 * np.dot(HTDTdDdH, H), np.dot(HT, z)) ##Is considered more numerically stable to find the solution than inverting the matrices - which seem to have a determinant = 0
            ztrend = np.dot(H, theta_hat_lambda)
            z_stat = z - ztrend
            return z_stat
'''
