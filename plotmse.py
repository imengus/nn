#!/home/ilkin/anaconda3/bin/python3.11

import matplotlib.pyplot as plt
import numpy as np
import csv

def main ():
    with open('mse_history.csv', 'r') as f:
        reader = csv.reader(f, delimiter='\n')
        data = list(reader)
    arr = np.array([float(i[0]) if i!='' else 0 for i in data ])
    plt.plot(arr, 'k')
    # plt.xaxis("off")
    plt.show()

if __name__ == "__main__":
    main()