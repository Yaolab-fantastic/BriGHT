import os
import csv

input_folder = "path/to/input_folder"
output_folder = "path/to/output_folder"

csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

for file_name in csv_files:
    input_file = os.path.join(input_folder, file_name)
    output_file = os.path.join(output_folder, file_name)

    with open(input_file, 'r') as infile:
        reader = csv.reader(infile)
        data = [row for row in reader]

    data = [row[1:] for row in data[1:]]

    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(data)

print("Done.")
