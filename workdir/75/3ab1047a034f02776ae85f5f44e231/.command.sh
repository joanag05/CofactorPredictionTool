#!/usr/bin/env python
from cofactor_prediction_tool.preprocessing import EmbeddingsESM2
embeddings = EmbeddingsESM2(folder_path='embeddings.tsv', data_path='preprocessed_data.tsv')
embeddings.compute_esm2_embeddings()
