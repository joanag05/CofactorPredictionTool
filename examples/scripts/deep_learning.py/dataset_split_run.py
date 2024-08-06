import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_and_save_dataset(file_path, output_path, test_size=0.2, val_size=0.15, random_state=42):

    df = pd.read_csv(file_path, sep='\t', index_col=0)
     
    cofactors = ['NAD', 'NADP', 'FAD', 'SAM', 'CoA', 'THF', 'FMN', 'Menaquinone', 'GSH', 'Ubiquinone', 'Plastoquinone', 'Ferredoxin', 'Ferricytochrome']

    cofactors_in_dataset = [cofactor for cofactor in cofactors if cofactor in df.columns]
    

    X = df.drop(columns=cofactors_in_dataset)
    y = df[cofactors_in_dataset]
    

    def multilabel_train_test_split(X, y, test_size=0.2, val_size=0.10, random_state=42):
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size, random_state=random_state)
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    X_train, X_val, X_test, y_train, y_val, y_test = multilabel_train_test_split(X, y, test_size=test_size, val_size=val_size, random_state=random_state)
    
    with pd.HDFStore(output_path, mode='w') as store:
        store.put('X_train', X_train)
        store.put('X_val', X_val)
        store.put('X_test', X_test)
        store.put('y_train', y_train)
        store.put('y_val', y_val)
        store.put('y_test', y_test)
    
    print("Dataset prepared and saved successfully.")

file_path = '/home/jgoncalves/cofactor_prediction_tool/data/Final/Flash/final_dataset_protflash_embeddings.tsv'
output_path = '/home/jgoncalves/cofactor_prediction_tool/data/Final/Flash/flashprot.h5'
prepare_and_save_dataset(file_path, output_path, test_size=0.2, val_size=0.15, random_state=42)
