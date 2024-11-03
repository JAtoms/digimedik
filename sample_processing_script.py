import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def ProcessData(csv_dict):
    df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                   columns=['a', 'b', 'c'])
    return df

def PlotImage(df):
    img = df.plot("a", use_index=True)
    return img

def main(arg):
    df = ProcessData(arg)
    img = PlotImage(df)
    return img

if __name__ == "__main__":
    main(sys.argv[0])