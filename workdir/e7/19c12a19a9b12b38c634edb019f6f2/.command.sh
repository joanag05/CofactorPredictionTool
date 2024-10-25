#!/usr/bin/env python
from cofactor_prediction_tool.preprocessing import Preprocessing
pr = Preprocessing()
import os
print(os.getcwd())
pr.read_fasta('iML1515.fasta')
pr.remove_duplicates_and_short_sequences()
pr.read_sequences()
pr.write_tsv('preprocessed_data.tsv')
