
process preprocessing{
    input:
    path inputFile 
    output:
    path ('preprocessed_data.tsv')
    script:
    """
    echo "Running preprocessing"
    echo "Input file: $inputFile"
    echo "$workDir"
    """

    if (inputFile.name.endsWith('.fasta')) {
        """
        #!/usr/bin/env python
        from cofactor_prediction_tool.preprocessing import Preprocessing
        pr = Preprocessing()
        import os
        print(os.getcwd())
        pr.read_fasta('$inputFile')
        pr.remove_duplicates_and_short_sequences()
        pr.read_sequences()
        pr.write_tsv('preprocessed_data.tsv')

        """


    } else if (inputFile.name.endsWith('.tsv')) {
        """
        #!/usr/bin/env python
        from cofactor_prediction_tool.preprocessing import Preprocessing
        pr = Preprocessing()
        pr.read_tsv('$inputFile')
        pr.remove_duplicates_and_short_sequences()
        pr.read_sequences()
        pr.write_tsv('preprocessed_data.tsv')
        """
    }

}

process compute_embeddings {
    input:
    path preprocessedFile
    output:
    path ('embeddings.tsv')
    script:
    """
    #!/usr/bin/env python
    from cofactor_prediction_tool.preprocessing import EmbeddingsESM2
    embeddings = EmbeddingsESM2(folder_path='embeddings.tsv', data_path='$preprocessedFile')
    embeddings.compute_esm2_embeddings()
    """
}


process predict{ 
    input:
    path embeddingsFile
    val modelName
    output:
    path ('predictions.tsv')
    path ('predictions_proba.tsv')
    script:
    """
    #!/usr/bin/env python
    import shutil
    import pandas as pd
    from cofactor_prediction_tool.deep_learning.cnn_model import CNNModel
    from cofactor_prediction_tool.deep_learning.cnn_model import predict, predict_proba, load_model
    model = load_model('$modelName')
    embeddings_df = pd.read_csv('$embeddingsFile', index_col=0, sep='\t')
    labels = ['NAD', 'NADP', 'FAD', 'SAM', 'CoA', 'THF', 'FMN', 'Menaquinone', 'GSH', 'Ubiquinone', 'Plastoquinone', 'Ferredoxin', 'Ferricytochrome']
    predictions = predict(model, embeddings_df, labels=labels)
    prediction_proba = predict_proba(model, embeddings_df, labels=labels,batch_size=32)
    predictions.to_csv("predictions.tsv", sep='\t')
    prediction_proba.to_csv("predictions_proba.tsv", sep='\t')
    shutil.copy("predictions.tsv", '$workDir/predictions.tsv')
    shutil.copy("predictions_proba.tsv", '$workDir/predictions_proba.tsv')

    """

}

process gsmmodel {
    input:
    path configFile
    path predictionsFile
    path modelFile
    script:
    """
    #!/usr/bin/env python
    import shutil
    from cofactor_prediction_tool.metabolic_modeling.gsmmodel import update_model
    update_model('$configFile', predictions = '$predictionsFile', model_path = '$modelFile')
    shutil.copy("updated_model.xml", '$workDir/updated_model.xml')
    shutil.copy("report.json", '$workDir/report.json')

    """
}


workflow  { 
    fasta_channel = params.fasta ? file(params.fasta) : null
    tsv_channel = params.tsv ? file(params.tsv) : null
    model_channel = params.model ? file(params.model) : EmptyChannel()
    input_channel = Channel.from(fasta_channel, tsv_channel).flatten().filter{ it != null }
    preprocessed_channel = preprocessing(input_channel)
    embeddings_channel = compute_embeddings(preprocessed_channel)
    (predictions, predictions_proba) = predict(embeddings_channel, 'cnn')
    gsmmodel(params.config, predictions, model_channel)
}

