import numpy as np
from preprocess import DataPreprocess
from load_dataset import BuildDataset
from global_args import PAD_INDEX, SOS_INDEX, EOS_INDEX, UNK_INDEX, toks_and_inds, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from embedding import forward, build, input_dim, output_dim

if __name__ == "__main__": 
    #import data
    from data import train_data, valid_data, test_data 

    """
        data preprocessing
    """
    #preprocess_obj = DataPreprocess()
    #train_data, valid_data, test_data = preprocess_obj.clear_dataset(train_data, valid_data, test_data)

    """
        build vocabulary
    """
    datasets = [train_data, valid_data, test_data]
    buildDataset_obj = BuildDataset()
    for dataset in datasets:
        vocabs = buildDataset_obj.build_vocab(dataset, toks_and_inds, min_freq=1)
    
    train_data_batches = buildDataset_obj.add_tokens(train_data, batch_size=2)
    train_source, train_target = buildDataset_obj.build_dataset(train_data_batches, vocabs)

    """
        embedding layer
    """

    w, v, m, v_hat, m_hat = build(input_dim, output_dim)
    embedding_nparray = forward(train_source, w)