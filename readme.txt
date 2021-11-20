---------README---------

#Files

## ADL_DCASE_DATA.zip

This is the zip file of the DCASE 2016 dataset. Unzipped it has the following file format:
-- Development/
---- audio/
---- labels.csv
-- Evaluation/
---- audio/
---- labels.csv

Each "audio/" directory contains all of the wav files for that split. "labels.csv" contains the labels for each audio sample inside "audio/". You should exclusively use the data in "Development/" for training. The data in "Evaluation/" is exclusively for evaluating your model. Do not train your model on the data in "Evaluation/".


## dataset.py
This is a PyTorch Dataset implemention for the DCASE 2016 dataset. This code creates spectrograms from the .wav files provided and splits them into shorter sequences. The DCASE class requires a path to your dataset and the length of your audio clips in seconds. You should use this class in conjuction with a PyTorch DataLoader. You can see examples of how to use a DataLoader in your lab code.

This dataset class will return tensors of the shape [batches, num_clips, height, width]. Most CNN models will expect data in the form [batches, channels, height, width]. In this case there is an additional dimension (num_clips) as a result of the sequence splitting described in the paper. In order to resolve this, we suggest you combine the number of clips into the batch dimension using torch.view(). You can then retrieve the correct dimensions by reshaping your data after passing it through the model. The DCASE class has a function, get_num_clips, which you can call. This function will return the number of clips each spectrogram is split into (determined by the clip length). You may find this useful when reshaping your tensors. 

## VisualiseSpec.ipynb
This is a jupyter notebook that allows you to visualise a spectrogram. The spectrogram producing code is the same as in dataset.py, and the current parameters match those in the paper. You can adjust these parameters if you wish. 






