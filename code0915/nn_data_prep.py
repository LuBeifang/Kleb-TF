import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import pyfastx
from tqdm.auto import tqdm




def tf_fa_path_parser(p):
    return p.split('/')[-1].split('.')[0]

def peak_fa_path_parser(p):
    return p.split('/')[-1].split('.')[0]


class TFDataset(Dataset):
    def __init__(self, tf_fasta_paths, peak_fasta_paths, binding_peaks_csv, all_peaks_csv, whole_genome_fa_path, neg_sampling_rate=1):
        # tf_fasta_paths is the path for DNA sequences of all TFs, each TF is a file
        # peak_fasta_paths is the path for DNA sequences of all binding peaks, each TF has a file containing multiple binding regions
        # binding_peaks_csv is a csv file containing the binding information of all TFs, each row is a binding region
        self.neg_sampling_rate = neg_sampling_rate
        self.binding_peaks_df = self._read_binding_csv(binding_peaks_csv)
        self.all_binding_peaks_df = self._read_binding_csv(all_peaks_csv)
        print('Finish reading binding peaks csv')

        tf_seqs_df = self._get_tf_seqs(tf_fasta_paths)
        pos_peak_seqs = self._get_pos_peak_seqs(peak_fasta_paths, binding_peaks_csv)
        print('Finish reading TF and binding DNA sequences')

        self.peak_with_seqs_df = pd.merge(tf_seqs_df, pos_peak_seqs, on='tf_name', validate='one_to_many')
        # assert len(self.peak_with_seqs_df) == len(self.binding_peaks_df)
        print('Finish reading positive peak sequences')

        neg_peak_seqs = self._get_neg_peak_seqs(tf_fasta_paths, whole_genome_fa_path, binding_peaks_csv, all_peaks_csv)
        self.neg_peak_with_seqs_df = pd.merge(tf_seqs_df, neg_peak_seqs, on='tf_name', validate='one_to_many')
        # assert len(self.neg_peak_with_seqs_df) == int(self.neg_sampling_rate * len(self.binding_peaks_df))
        print('Finish creating negative peak sequences')

    def _get_pos_peak_seqs(self, peak_fasta_paths, binding_peaks_csv):
        tf_names = []  # one to many
        peak_seqs = []
        peak_fc = []
        df = pd.read_csv(binding_peaks_csv, sep=',', header=None, index_col=False, usecols=[0, 2, 3, 8], names=['tf_name', 'binding_start', 'binding_end', 'binding_fc'])

        for p in tqdm(peak_fasta_paths):
            current_tf_name = peak_fa_path_parser(p)
            fa = pyfastx.Fasta(p, build_index=False)
            for _, seq in fa:
                peak_seqs.append(seq)
                tf_names.append(current_tf_name)

            # Retrive the present TF's binding_fc in binding_peaks_df
            for _, row in df[df['tf_name'] == current_tf_name].iterrows():
                peak_fc.append(row['binding_fc'])


        # peak_fc = [x / 7 for x in peak_fc]

        assert len(peak_seqs) == len(tf_names) == len(peak_fc)
        peak_df = pd.DataFrame({'tf_name': tf_names, 'peak_seq': peak_seqs, 'peak_fc': peak_fc})
        return peak_df


    def _get_neg_peak_seqs(self, tf_fasta_paths, whole_genome_path,binding_peaks_csv, all_peaks_csv):
        """generate negative samples, excluding all binding sites"""
        # df = pd.read_csv(binding_peaks_csv, sep=',', header=None, index_col=False, usecols=[0, 2, 3, 8], names=['tf_name', 'binding_start', 'binding_end', 'binding_fc'])

        df_all = pd.read_csv(all_peaks_csv, sep=',', header=None, index_col=False, usecols=[0, 2, 3, 8], names=['tf_name', 'binding_start', 'binding_end', 'binding_fc'])

        peak_seq_lengths = self.peak_with_seqs_df['peak_seq'].apply(len).values

        whole_genome = pyfastx.Fasta(whole_genome_path)
        whole_genome = ''.join([item.seq for item in whole_genome])

        tf_pos_counts = self.peak_with_seqs_df['tf_name'].value_counts()

        tf_names = []
        peak_seqs = []

        for p in tqdm(tf_pos_counts.index, desc="Processing TFs"):
            tf_name = p
            pos_count = tf_pos_counts[tf_name]
            neg_count = int(pos_count * self.neg_sampling_rate)


            # Retrive the present TF's binding region in binding_peaks_df
            tf_binding_regions = [
            (row['binding_start'], row['binding_end'])
            for _, row in df_all[df_all['tf_name'] == tf_name].iterrows()
        ]

            generated = 0
            max_attempts = neg_count * 10
            attempts = 0

            pbar = tqdm(
            total=neg_count,
            desc=f"Generating negatives for {tf_name}",
            leave=True,
        )

            while generated < neg_count and attempts < max_attempts:
                attempts += 1

                current_peak_len = np.random.choice(peak_seq_lengths)

                random_start = np.random.randint(0, len(whole_genome) - current_peak_len + 1)
                random_end = random_start + current_peak_len

                overlaps = False
                for binding_start, binding_end in tf_binding_regions:
                    if not (random_end <= binding_start or random_start >= binding_end):
                        overlaps = True
                        break

                if not overlaps:
                    seq = whole_genome[random_start:random_end]

                    peak_seqs.append(seq)
                    tf_names.append(tf_name)

                    generated += 1
                    pbar.update(1)
            pbar.close()

            if generated < neg_count:
                print(f"warning: only {generated}/{neg_count} negs were generated for {tf_name}")

        # Store the generated sequences
        peak_df = pd.DataFrame({
            'tf_name': tf_names,
            'peak_seq': peak_seqs,
            'peak_fc': np.zeros(len(peak_seqs))
        })
        # assert len(peak_df) == int(self.neg_sampling_rate * len(self.binding_peaks_df))
        return peak_df


    def _get_tf_seqs(self, tf_fasta_paths):
        tf_seqs = []
        tf_names = []
        for p in tqdm(tf_fasta_paths):
            fa = pyfastx.Fasta(p, build_index=False)
            seqs = [item[1] for item in fa]
            assert len(seqs) == 1
            seq = seqs[0]
            tf_seqs.append(seq)
            tf_names.append(tf_fa_path_parser(p))
        tf_df = pd.DataFrame({'tf_name': tf_names, 'tf_seq': tf_seqs})
        return tf_df

    def _read_binding_csv(self, binding_peaks_csv):
        df = pd.read_csv(binding_peaks_csv, sep=',', header=None, index_col=False, usecols=[0, 2, 3, 8], names=['tf_name', 'binding_start', 'binding_end', 'binding_fc'])
        df['binding_start'] = df['binding_start'] - 1  # Convert to 0-base
        # print(df)
        return df

    def __len__(self):
        pos_len = len(self.binding_peaks_df)
        neg_len = int(self.neg_sampling_rate * pos_len)
        return pos_len + neg_len

    # def _normalization(self, peak_fc):
    #     range = np.max(peak_fc) - 0
    #     return (peak_fc - 0) / range



    def __getitem__(self, idx):
        if idx < len(self.binding_peaks_df):
            tf_seq = self.peak_with_seqs_df.iloc[idx]['tf_seq']
            peak_seq = self.peak_with_seqs_df.iloc[idx]['peak_seq']
            peak_fc = self.peak_with_seqs_df.iloc[idx]['peak_fc']
            label = 1
        else:
            neg_idx = idx - len(self.binding_peaks_df)
            tf_seq = self.neg_peak_with_seqs_df.iloc[neg_idx]['tf_seq']
            peak_seq = self.neg_peak_with_seqs_df.iloc[neg_idx]['peak_seq']
            peak_fc = self.neg_peak_with_seqs_df.iloc[neg_idx]['peak_fc']
            label = 0
        return tf_seq, peak_seq, peak_fc, torch.tensor(label, dtype=torch.float32)

