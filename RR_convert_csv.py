# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 15:15:46 2023

@author: lm9172
"""

import csv
import os
import re
import glob

# Set the path to the directory containing the files
directory = "C:\\Users\\lm9172\\OneDrive - Anglia Ruskin University\\HR_data\\"

for j in range(1,20):
    for i in range(1,16):
    
        fpath = directory + "PT-{:02} S{:02}.csv".format(j, i) 
    #for file in glob.glob(fpath):
        #print(fpath)
        if os.path.isfile(fpath) == False:
            continue
        
        else:
            filep = os.path.splitext(fpath)[0]
            filen = os.path.basename(filep)
            Newfile = os.path.dirname(fpath) + "\\" + filen + "_RR.csv"
            Nfile = "C:\\Users\\lm9172\\OneDrive - Anglia Ruskin University\\HR_data\\" + filen + "_RR.csv"
            #print(Nfile)
            if os.path.exists(Nfile):
                print("\n Skipping " + os.path.basename(fpath) + " \n because " + os.path.basename(Nfile) + " already exists")
            # Open the csv file for reading
                continue
            
            else:
                with open(fpath, 'r') as csv_file:
                    reader = csv.reader(csv_file)
                    
                    # Create a new csv file for writing
                    with open(Nfile, 'w') as new_file:
                        writer = csv.writer(new_file)
                        
                        # Iterate over the rows in the csv file
                        for row_index, row in enumerate(reader):
                            # Delete the 7th column
                            if len(row) > 6:
                                del row[6]
                            
                            # Insert a new column
                            row.insert(6, '')
                            
                            # Write "RR (ms)" in the new column of row 3
                            if row_index == 2:
                                row[6] = "RR (ms)"
                            
                            # Write the formula "=60000/Cx" in the new column of row 4 and subsequently
                            elif row_index > 2:
                                try:
                                    if int(row[2]) == 0:
                                        row[2] = prev_value
                                        
                                    prev_value = int(row[2])
                                    row[6] = 60000 / int(row[2])
                                except ValueError:
                                    pass
                            
                            # Write the modified row to the new csv file
                            writer.writerow(row)
                print("\n Converted BPM to RR intervals: " + os.path.basename(Nfile))
