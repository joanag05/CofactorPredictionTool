import sys
import os
import pandas as pd
import re
from brendapy import BrendaParser
from propythia.protein.sequence import ReadSequence
sys.path.append(r"/home/jgoncalves/cofactor_prediction_tool/src")
from cofactor_prediction_tool.build_dataset import BuildDataset


if __name__ == "__main__":
    data_path = '/home/jgoncalves/cofactor_prediction_tool/data/Final'
    generator = BuildDataset(data_path)
    generator.generate_final_dataset()


