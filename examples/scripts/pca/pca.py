import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os


esm = pd.read_csv('/home/jgoncalves/cofactor_prediction_tool/data/ml_dl_data/ESM2/esm.tsv', sep='\t')


if 'ProteinID' not in esm.columns:
    raise KeyError("The 'ProteinID' column is missing in the ESM data.")

reviewed_path = '/home/jgoncalves/cofactor_prediction_tool/data/reviewed_ids.csv'
unreviewed_path = '/home/jgoncalves/cofactor_prediction_tool/data/unreviewed_ids.csv'

if os.path.exists(reviewed_path):
    reviewed_df = pd.read_csv(reviewed_path, sep=',')
    reviewed_df['Reviewed'] = True
    if 'Accession' in reviewed_df.columns:
        reviewed_df.rename(columns={'Accession': 'ProteinID'}, inplace=True)
    if 'ProteinID' not in reviewed_df.columns:
        raise KeyError(f"The 'ProteinID' column is missing in the reviewed data at {reviewed_path}.")
else:
    print(f"Reviewed file not found at {reviewed_path}. Creating an empty DataFrame for reviewed data.")
    reviewed_df = pd.DataFrame(columns=['ProteinID', 'Reviewed'])
    reviewed_df['Reviewed'] = True

# Load unreviewed data
if os.path.exists(unreviewed_path):
    unreviewed_df = pd.read_csv(unreviewed_path, sep=',')
    unreviewed_df['Reviewed'] = False
    if 'Accession' in unreviewed_df.columns:
        unreviewed_df.rename(columns={'Accession': 'ProteinID'}, inplace=True)
    if 'ProteinID' not in unreviewed_df.columns:
        raise KeyError(f"The 'ProteinID' column is missing in the unreviewed data at {unreviewed_path}.")
else:
    print(f"Unreviewed file not found at {unreviewed_path}. Creating an empty DataFrame for unreviewed data.")
    unreviewed_df = pd.DataFrame(columns=['ProteinID', 'Reviewed'])
    unreviewed_df['Reviewed'] = False

# Concatenate reviewed and unreviewed DataFrames
status_df = pd.concat([reviewed_df, unreviewed_df])

# Merge the status back into the main dataset
merged_df = esm.merge(status_df, on='ProteinID', how='left')

# Extract embedding columns
embedding_columns = [col for col in merged_df.columns if col.startswith('embedding_')]
merged_df['Embeddings'] = merged_df[embedding_columns].values.tolist()

# List of cofactors
cofactors = ['SAM', 'NAD', 'NADP', 'FAD', 'CoA', 'THF', 'FMN', 'Menaquinone', 'GSH', 'Ubiquinone', 'Plastoquinone', 'Ferredoxin', 'Ferricytochrome']

# Define a consistent color palette for the 'Reviewed' status
palette = {True: 'blue', False: 'red'}

# Group by cofactor and perform PCA for each group
for cofactor in cofactors:
    group = merged_df[merged_df[cofactor] == 1]
    if group.empty:
        print(f"No data for cofactor: {cofactor}")
        continue

    embeddings = group['Embeddings'].tolist()
    labels = group['Reviewed'].tolist()

    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)

    # Create a DataFrame for the PCA results
    pca_df = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
    pca_df['Reviewed'] = labels

    # Plot the PCA results
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Reviewed', data=pca_df, palette=palette, hue_order=[True, False])
    plt.title(f'PCA of Protein Embeddings for Cofactor: {cofactor}')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title='Reviewed Status')

    # Save the plot
    output_path = f'/home/jgoncalves/cofactor_prediction_tool/pca/pca_{cofactor}.png'
    plt.savefig(output_path)
    plt.close()
    print(f"Saved PCA plot for cofactor {cofactor} at {output_path}")