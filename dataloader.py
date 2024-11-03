import torch
import os
import numpy as np
class TrainDataset(torch.utils.data.Dataset):

    # For this homework, we give you full flexibility to design your data set class.
    # Hint: The data from HW1 is very similar to this HW

    def __init__(self, root, phonemes = PHONEMES, partition= "train-clean-100", limit=None):
        '''
        Initializes the dataset.

        INPUTS: What inputs do you need here?
        '''

        # Load the directory and all files in them

        self.mfcc_dir = os.path.join(root, partition, 'mfcc')
        self.transcript_dir = os.path.join(root, partition, 'transcript')

        self.mfcc_names = sorted(os.listdir(self.mfcc_dir))
        self.transcript_names = sorted(os.listdir(self.transcript_dir))

        self.PHONEMES = phonemes

        assert len(self.mfcc_names) == len(self.transcript_names)

        # length of dataset is the number of transcripts
        self.length = len(self.transcript_names)

        # HOW CAN WE REPRESENT PHONEMES? CAN WE CREATE A MAPPING FOR THEM?
        # HINT: TENSORS CANNOT STORE NON-NUMERICAL VALUES OR STRINGS
        self.mfccs, self.transcripts = [], []
        
        if limit is None:
            limit = len(self.mfcc_names)
        # Iterate through mfccs and transcripts
        for i in range(limit):
        #   Load a single mfcc
            mfcc        = np.load(os.path.join(self.mfcc_dir, self.mfcc_names[i]))
        #   Do Cepstral Normalization of mfcc (explained in writeup)
            mfcc        = (mfcc - np.mean(mfcc, axis=0)) / (np.std(mfcc, axis=0) + 1e-8)
        #   Load the corresponding transcript
            transcript  = np.load(os.path.join(self.transcript_dir, self.transcript_names[i]))
        #   Remove [SOS] and [EOS] from the transcript
            transcript  = transcript[1:-1]
            # Note that SOS will always be in the starting and EOS at end, as the name suggests.
        #   Append each mfcc to self.mfcc, transcript to self.transcript
            self.mfccs.append(mfcc)
            self.transcripts.append(transcript)

        # The available phonemes in the transcript are of string data type
        # But the neural network cannot predict strings as such.
        # Hence, we map these phonemes to integers

        # Map the phonemes to their corresponding list indexes in self.phonemes
        # CREATE AN ARRAY OF ALL FEATUERS AND LABELS
        # Use Cepstral Normalization of mfcc
        self.transcripts = [[self.PHONEMES.index(phoneme) for phoneme in transcript] for transcript in self.transcripts]
        assert len(self.mfccs) == len(self.transcripts)
        
        '''
        You may decide to do this in __getitem__ if you wish.
        However, doing this here will make the __init__ function take the load of
        loading the data, and shift it away from training.
        '''


    def __len__(self):

        return self.length

    def __getitem__(self, ind):
        '''
        RETURN THE MFCC COEFFICIENTS AND ITS CORRESPONDING LABELS

        If you didn't do the loading and processing of the data in __init__,
        do that here.

        Once done, return a tuple of features and labels.
        '''
        mfcc        = torch.tensor(self.mfccs[ind], dtype=torch.float32)
        transcript  = torch.tensor(self.transcripts[ind], dtype=torch.int32)
        return mfcc, transcript


    def collate_fn(self,batch):
        '''
        1.  Extract the features and labels from 'batch'
        2.  We will additionally need to pad both features and labels,
            look at pytorch's docs for pad_sequence
        3.  This is a good place to perform transforms, if you so wish.
            Performing them on batches will speed the process up a bit.
        4.  Return batch of features, labels, lenghts of features,
            and lengths of labels.
        '''
        # batch of input mfcc coefficients and output phonemes
        batch_mfcc, batch_transcript = zip(*batch)

        # HINT: CHECK OUT -> pad_sequence (imported above)
        # Also be sure to check the input format (batch_first)
        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True)
        lengths_mfcc = np.array([mfcc.shape[0] for mfcc in batch_mfcc])

        batch_transcript_pad = pad_sequence(batch_transcript, batch_first=True)
        lengths_transcript =  np.array([transcript.shape[0] for transcript in batch_transcript])

        # You may apply some transformation, Time and Frequency masking, here in the collate function;
        # Food for thought -> Why are we applying the transformation here and not in the __getitem__?
        #                  -> Would we apply transformation on the validation set as well?
        #                  -> Is the order of axes / dimensions as expected for the transform functions?

        # Return the following values: padded features, padded labels, actual length of features, actual length of the labels
        return batch_mfcc_pad, batch_transcript_pad, torch.tensor(lengths_mfcc), torch.tensor(lengths_transcript)


class TrainDataset(torch.utils.data.Dataset):

    # For this homework, we give you full flexibility to design your data set class.
    # Hint: The data from HW1 is very similar to this HW

    def __init__(self, root, phonemes = PHONEMES, partition= "train-clean-100", limit=None):
        '''
        Initializes the dataset.

        INPUTS: What inputs do you need here?
        '''

        # Load the directory and all files in them

        self.mfcc_dir = os.path.join(root, partition, 'mfcc')
        self.transcript_dir = os.path.join(root, partition, 'transcript')

        self.mfcc_names = sorted(os.listdir(self.mfcc_dir))
        self.transcript_names = sorted(os.listdir(self.transcript_dir))

        self.PHONEMES = phonemes

        assert len(self.mfcc_names) == len(self.transcript_names)

        # length of dataset is the number of transcripts
        self.length = len(self.transcript_names)

        # HOW CAN WE REPRESENT PHONEMES? CAN WE CREATE A MAPPING FOR THEM?
        # HINT: TENSORS CANNOT STORE NON-NUMERICAL VALUES OR STRINGS
        self.mfccs, self.transcripts = [], []
        
        if limit is None:
            limit = len(self.mfcc_names)
        # Iterate through mfccs and transcripts
        for i in range(limit):
        #   Load a single mfcc
            mfcc        = np.load(os.path.join(self.mfcc_dir, self.mfcc_names[i]))
        #   Do Cepstral Normalization of mfcc (explained in writeup)
            mfcc        = (mfcc - np.mean(mfcc, axis=0)) / (np.std(mfcc, axis=0) + 1e-8)
        #   Load the corresponding transcript
            transcript  = np.load(os.path.join(self.transcript_dir, self.transcript_names[i]))
        #   Remove [SOS] and [EOS] from the transcript
            transcript  = transcript[1:-1]
            # Note that SOS will always be in the starting and EOS at end, as the name suggests.
        #   Append each mfcc to self.mfcc, transcript to self.transcript
            self.mfccs.append(mfcc)
            self.transcripts.append(transcript)

        # The available phonemes in the transcript are of string data type
        # But the neural network cannot predict strings as such.
        # Hence, we map these phonemes to integers

        # Map the phonemes to their corresponding list indexes in self.phonemes
        # CREATE AN ARRAY OF ALL FEATUERS AND LABELS
        # Use Cepstral Normalization of mfcc
        self.transcripts = [[self.PHONEMES.index(phoneme) for phoneme in transcript] for transcript in self.transcripts]
        assert len(self.mfccs) == len(self.transcripts)
        
        '''
        You may decide to do this in __getitem__ if you wish.
        However, doing this here will make the __init__ function take the load of
        loading the data, and shift it away from training.
        '''


    def __len__(self):

        return self.length

    def __getitem__(self, ind):
        '''
        RETURN THE MFCC COEFFICIENTS AND ITS CORRESPONDING LABELS

        If you didn't do the loading and processing of the data in __init__,
        do that here.

        Once done, return a tuple of features and labels.
        '''
        mfcc        = torch.tensor(self.mfccs[ind], dtype=torch.float32)
        transcript  = torch.tensor(self.transcripts[ind], dtype=torch.int32)
        return mfcc, transcript


    def collate_fn(self,batch):
        '''
        1.  Extract the features and labels from 'batch'
        2.  We will additionally need to pad both features and labels,
            look at pytorch's docs for pad_sequence
        3.  This is a good place to perform transforms, if you so wish.
            Performing them on batches will speed the process up a bit.
        4.  Return batch of features, labels, lenghts of features,
            and lengths of labels.
        '''
        # batch of input mfcc coefficients and output phonemes
        batch_mfcc, batch_transcript = zip(*batch)

        # HINT: CHECK OUT -> pad_sequence (imported above)
        # Also be sure to check the input format (batch_first)
        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True)
        lengths_mfcc = np.array([mfcc.shape[0] for mfcc in batch_mfcc])

        batch_transcript_pad = pad_sequence(batch_transcript, batch_first=True)
        lengths_transcript =  np.array([transcript.shape[0] for transcript in batch_transcript])

        # You may apply some transformation, Time and Frequency masking, here in the collate function;
        # Food for thought -> Why are we applying the transformation here and not in the __getitem__?
        #                  -> Would we apply transformation on the validation set as well?
        #                  -> Is the order of axes / dimensions as expected for the transform functions?

        # Return the following values: padded features, padded labels, actual length of features, actual length of the labels
        return batch_mfcc_pad, batch_transcript_pad, torch.tensor(lengths_mfcc), torch.tensor(lengths_transcript)
    

class TrainDataset(torch.utils.data.Dataset):

    # For this homework, we give you full flexibility to design your data set class.
    # Hint: The data from HW1 is very similar to this HW

    def __init__(self, root, phonemes = PHONEMES, partition= "train-clean-100", limit=None):
        '''
        Initializes the dataset.

        INPUTS: What inputs do you need here?
        '''

        # Load the directory and all files in them

        self.mfcc_dir = os.path.join(root, partition, 'mfcc')
        self.transcript_dir = os.path.join(root, partition, 'transcript')

        self.mfcc_names = sorted(os.listdir(self.mfcc_dir))
        self.transcript_names = sorted(os.listdir(self.transcript_dir))

        self.PHONEMES = phonemes

        assert len(self.mfcc_names) == len(self.transcript_names)

        # length of dataset is the number of transcripts
        self.length = len(self.transcript_names)

        # HOW CAN WE REPRESENT PHONEMES? CAN WE CREATE A MAPPING FOR THEM?
        # HINT: TENSORS CANNOT STORE NON-NUMERICAL VALUES OR STRINGS
        self.mfccs, self.transcripts = [], []
        
        if limit is None:
            limit = len(self.mfcc_names)
        # Iterate through mfccs and transcripts
        for i in range(limit):
        #   Load a single mfcc
            mfcc        = np.load(os.path.join(self.mfcc_dir, self.mfcc_names[i]))
        #   Do Cepstral Normalization of mfcc (explained in writeup)
            mfcc        = (mfcc - np.mean(mfcc, axis=0)) / (np.std(mfcc, axis=0) + 1e-8)
        #   Load the corresponding transcript
            transcript  = np.load(os.path.join(self.transcript_dir, self.transcript_names[i]))
        #   Remove [SOS] and [EOS] from the transcript
            transcript  = transcript[1:-1]
            # Note that SOS will always be in the starting and EOS at end, as the name suggests.
        #   Append each mfcc to self.mfcc, transcript to self.transcript
            self.mfccs.append(mfcc)
            self.transcripts.append(transcript)

        # The available phonemes in the transcript are of string data type
        # But the neural network cannot predict strings as such.
        # Hence, we map these phonemes to integers

        # Map the phonemes to their corresponding list indexes in self.phonemes
        # CREATE AN ARRAY OF ALL FEATUERS AND LABELS
        # Use Cepstral Normalization of mfcc
        self.transcripts = [[self.PHONEMES.index(phoneme) for phoneme in transcript] for transcript in self.transcripts]
        assert len(self.mfccs) == len(self.transcripts)
        
        '''
        You may decide to do this in __getitem__ if you wish.
        However, doing this here will make the __init__ function take the load of
        loading the data, and shift it away from training.
        '''


    def __len__(self):

        return self.length

    def __getitem__(self, ind):
        '''
        RETURN THE MFCC COEFFICIENTS AND ITS CORRESPONDING LABELS

        If you didn't do the loading and processing of the data in __init__,
        do that here.

        Once done, return a tuple of features and labels.
        '''
        mfcc        = torch.tensor(self.mfccs[ind], dtype=torch.float32)
        transcript  = torch.tensor(self.transcripts[ind], dtype=torch.int32)
        return mfcc, transcript


    def collate_fn(self,batch):
        '''
        1.  Extract the features and labels from 'batch'
        2.  We will additionally need to pad both features and labels,
            look at pytorch's docs for pad_sequence
        3.  This is a good place to perform transforms, if you so wish.
            Performing them on batches will speed the process up a bit.
        4.  Return batch of features, labels, lenghts of features,
            and lengths of labels.
        '''
        # batch of input mfcc coefficients and output phonemes
        batch_mfcc, batch_transcript = zip(*batch)

        # HINT: CHECK OUT -> pad_sequence (imported above)
        # Also be sure to check the input format (batch_first)
        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True)
        lengths_mfcc = np.array([mfcc.shape[0] for mfcc in batch_mfcc])

        batch_transcript_pad = pad_sequence(batch_transcript, batch_first=True)
        lengths_transcript =  np.array([transcript.shape[0] for transcript in batch_transcript])

        # You may apply some transformation, Time and Frequency masking, here in the collate function;
        # Food for thought -> Why are we applying the transformation here and not in the __getitem__?
        #                  -> Would we apply transformation on the validation set as well?
        #                  -> Is the order of axes / dimensions as expected for the transform functions?

        # Return the following values: padded features, padded labels, actual length of features, actual length of the labels
        return batch_mfcc_pad, batch_transcript_pad, torch.tensor(lengths_mfcc), torch.tensor(lengths_transcript)