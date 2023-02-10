# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 15:35:57 2023

@author: lm9172
"""
## CAREFUL with file names that have the same PT & Session number 
## yet are different parts or recordings.
## This code will remove the first of that series it encounters
## and rename it. Thus, you must manually rename appropriately that file.

import os
import re

# Set the path to the directory containing the files
directory = "C:\\Users\\lm9172\\OneDrive - Anglia Ruskin University\\HR_data"

# Use a regular expression to match the desired file name format
file_format = re.compile(r"PT-[0-9][0-9] S[0-9][0-9].csv")
print(directory)

# Use os.listdir() to get a list of all the files in the directory
for file in os.listdir(directory):
            print(file)

            numbers = re.findall(r"\d+", file)
           
            
        # Build the new file name
            new_file_name = f"PT-{numbers[0].zfill(2)} S{numbers[1].zfill(2)}.csv"
            print(new_file_name)
            # Construct the file paths
            old_file_path = os.path.join(directory, file)
            new_file_path = os.path.join(directory, new_file_name)
            
            
            # Rename the file
            if os.path.exists(new_file_path):
                #print("Filename already exists, adding (1) in front")
                #new_file_path =  directory + "\\(1)" + new_file_name
                #os.rename(old_file_path, new_file_path)
                #print(new_file_path)
                continue
            else:
            
                os.rename(old_file_path, new_file_path)
            
            print("finished renaming " + new_file_name)