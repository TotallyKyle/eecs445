import csv
with open('USD_JPY_sample.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        print row
