
from random import random,sample,shuffle
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import pyfastx
from tqdm.auto import tqdm
import glob
import pyfastx
import os


seed = 42
torch.manual_seed(seed)          
np.random.seed(seed)             
np.random.seed(seed)

def tf_fa_path_parser(p):
    return p.split('/')[-1].split('.')[0]

def peak_fa_path_parser(p):
    return p.split('/')[-1].split('.')[0]
def partial_shuffle(seq: str, shuffle_frac: float) -> str:
    """
    Partially shuffle a DNA sequence.

    Parameters
    ----------
    seq : str
        Input DNA sequence (e.g. 'ACGT...').
    shuffle_frac : float
        Fraction of positions to be shuffled in-place (0.0 ~ 1.0).
        - 0.0  -> no change
        - 0.2  -> roughly 20% of positions will be permuted among themselves
        - 1.0  -> full shuffle of all positions

    Returns
    -------
    shuffled_seq : str
        Sequence of same length where only a subset of positions
        have been randomly permuted.
    """
    if not 0.0 <= shuffle_frac <= 1.0:
        raise ValueError("shuffle_frac must be between 0.0 and 1.0")

    L = len(seq)
    if L == 0 or shuffle_frac == 0.0:
        return seq

    k = max(1, int(round(L * shuffle_frac))) if shuffle_frac > 0 else 0
    k = min(k, L)  
    positions = sorted(sample(range(L), k=k))

    chars = [seq[i] for i in positions]

    shuffle(chars)

    seq_list = list(seq)
    for idx, pos in enumerate(positions):
        seq_list[pos] = chars[idx]

    return ''.join(seq_list)

class TFDataset(Dataset):
    def __init__(self, tf_fasta_paths, peak_fasta_paths, binding_peaks_csv, all_peaks_csv, whole_genome_fa_path, gff_path, neg_sampling_rate=1, ifshuffle_neg=False,shuffle_fraction=0.4):
        # tf_fasta_paths is the path for DNA sequences of all TFs, each TF is a file
        # peak_fasta_paths is the path for DNA sequences of all binding peaks, each TF has a file containing multiple binding regions
        # binding_peaks_csv is a csv file containing the binding information of all TFs, each row is a binding region
        self.neg_sampling_rate = neg_sampling_rate
        self.binding_peaks_df = self._read_binding_csv(binding_peaks_csv)
        self.all_binding_peaks_df = self._read_binding_csv(all_peaks_csv)
        self.shuffle_fraction=shuffle_fraction
        print('Finish reading binding peaks csv')

        self.tss_positions = self._read_tss(gff_path)
        print('Finish reading TSS positions')

        tf_seqs_df = self._get_tf_seqs(tf_fasta_paths)
        self.pos_peak_seqs = self._get_pos_peak_seqs(peak_fasta_paths, binding_peaks_csv)
        print('Finish reading TF and binding DNA sequences')

        self.peak_with_seqs_df = pd.merge(tf_seqs_df, self.pos_peak_seqs, on='tf_name', validate='one_to_many')
        assert len(self.peak_with_seqs_df) == len(self.binding_peaks_df)
        print('Finish reading positive peak sequences')

        if ifshuffle_neg:
            tmp_neg = [
            self._dna_permutations(peak_fasta_paths, binding_peaks_csv) for _ in range(neg_sampling_rate)
            ]
            neg_peak_seqs = pd.concat(tmp_neg, ignore_index=True)
            self.neg_peak_with_seqs_df = pd.merge(tf_seqs_df, neg_peak_seqs, on='tf_name', validate='one_to_many')
        else:
            neg_peak_seqs_tss = self._get_neg_peak_seqs_tss(tf_fasta_paths, whole_genome_fa_path, binding_peaks_csv, all_peaks_csv)
            neg_peak_seqs_all = self._get_neg_peak_seqs_whole(tf_fasta_paths, whole_genome_fa_path, binding_peaks_csv, all_peaks_csv)
            #shuffled_neg_peak_seqs = self._get_shuffled_neg_peak_seqs()
            shuffled_neg_peak_seqs= self._get_all_noverlap_neg_peak_seqs()
            #neg_peak_seqs=pd.concat([shuffled_neg_peak_seqs,neg_peak_seqs_all])
            total_neg=np.round(self.neg_sampling_rate*len(self.binding_peaks_df)/3).astype(int)
            neg_peak_seqs_tss=neg_peak_seqs_tss.sample(n=total_neg,random_state=42)
            neg_peak_seqs_all=neg_peak_seqs_all.sample(n=total_neg,random_state=42)
            shuffled_neg_peak_seqs=shuffled_neg_peak_seqs.sample(n=total_neg,random_state=42)
            neg_peak_seqs=pd.concat([neg_peak_seqs_tss,neg_peak_seqs_all,shuffled_neg_peak_seqs])
            self.neg_peak_with_seqs_df = pd.merge(tf_seqs_df, neg_peak_seqs, on='tf_name', validate='one_to_many')
        

        print(self.neg_peak_with_seqs_df.shape, neg_peak_seqs.shape)
        #assert len(self.neg_peak_with_seqs_df) == int(self.neg_sampling_rate * len(self.binding_peaks_df))
        print('Finish creating negative peak sequences',f"dimension of negative data is {self.neg_peak_with_seqs_df.shape}, the biniding peak size is {self.binding_peaks_df.shape}")


    
    def _get_neg_peak_seqs_whole(self, tf_fasta_paths, whole_genome_path,binding_peaks_csv, all_peaks_csv):
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
        assert len(peak_df) == int(self.neg_sampling_rate * len(self.binding_peaks_df))
        return peak_df
    def _get_neg_peak_seqs_tss(self, tf_fasta_paths, whole_genome_path,binding_peaks_csv, all_peaks_csv):
        """generate negative samples from TSS promoter regions, excluding all binding sites"""
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


            # Retrieve the present TF's binding regions from all_binding_peaks_df
            tf_binding_regions = self.all_binding_peaks_df[self.all_binding_peaks_df['tf_name'] == tf_name][['binding_start', 'binding_end']].values.tolist()

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

                # Randomly select a TSS
                


                tss_item = self.tss_positions[np.random.choice(len(self.tss_positions))]

                tss, strand = tss_item

                if strand == '+':
                    prom_start = max(0, tss - 300)
                    prom_end = min(len(whole_genome), tss + 200)
                elif strand == '-':
                    prom_start = max(0, tss - 200)
                    prom_end = min(len(whole_genome), tss + 300)
                else:
                    continue

                if prom_end - prom_start < current_peak_len:
                    continue

                random_start = np.random.randint(prom_start, prom_end - current_peak_len + 1)
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
        assert len(peak_df) == int(self.neg_sampling_rate * len(self.binding_peaks_df))
        return peak_df

    def _get_pos_peak_seqs(self, peak_fasta_paths, binding_peaks_csv):
        tf_names = []  # one to many
        peak_seqs = []
        peak_fc = []
        binding_starts = []
        binding_ends = []
        df = pd.read_csv(binding_peaks_csv, sep=',', header=None, index_col=False, usecols=[0, 2, 3, 8], names=['tf_name', 'binding_start', 'binding_end', 'binding_fc'])

        for p in tqdm(peak_fasta_paths):
            current_tf_name = peak_fa_path_parser(p)
            fa = pyfastx.Fasta(p, build_index=False)
            seqs = [seq for name, seq in fa]

            tf_df = df[df['tf_name'] == current_tf_name].reset_index(drop=True)
            assert len(seqs) == len(tf_df)

            for i in range(len(seqs)):
                tf_names.append(current_tf_name)
                peak_seqs.append(seqs[i])
                peak_fc.append(tf_df.iloc[i]['binding_fc'])
                binding_starts.append(tf_df.iloc[i]['binding_start'])
                binding_ends.append(tf_df.iloc[i]['binding_end'])

        assert len(peak_seqs) == len(tf_names) == len(peak_fc) == len(binding_starts) == len(binding_ends)
        peak_df = pd.DataFrame({'tf_name': tf_names, 'peak_seq': peak_seqs, 'peak_fc': peak_fc, 'binding_start': binding_starts, 'binding_end': binding_ends})
        return peak_df
    
    def _get_all_noverlap_neg_peak_seqs(self):
        """Generate negative samples using ALL non-overlapping peaks from other TFs"""
        tf_pos_counts = self.peak_with_seqs_df['tf_name'].value_counts()
        
        all_tf_names = []
        all_peak_seqs = []
        
        for tf_name in tqdm(tf_pos_counts.index, desc="Processing TFs for non-overlap negatives"):
            # Get current TF's binding regions
            tf_binding_regions = self.all_binding_peaks_df[
                self.all_binding_peaks_df['tf_name'] == tf_name
            ][['binding_start', 'binding_end']].values.tolist()
            
            # Get positive peaks from other TFs
            other_tf_peaks = self.pos_peak_seqs[self.pos_peak_seqs['tf_name'] != tf_name]
            
            if len(other_tf_peaks) == 0:
                print(f"Warning: No other TF peaks available for {tf_name}")
                continue
            
            # Filter for non-overlapping peaks
            non_overlap_peaks = []
            
            for _, row in other_tf_peaks.iterrows():
                random_start = row['binding_start']
                random_end = row['binding_end']
                seq = row['peak_seq']
                fc = row.get('peak_fc', 0.0)  # Use FC if available, otherwise 0
                
                # Check overlap with current TF's binding regions
                overlaps = False
                for binding_start, binding_end in tf_binding_regions:
                    if not (random_end <= binding_start or random_start >= binding_end):
                        overlaps = True
                        break
                
                if not overlaps:
                    non_overlap_peaks.append({
                        'seq': seq,
                        'fc': fc
                    })
            
            # Add all non-overlapping peaks as negative samples
            for peak_data in non_overlap_peaks:
                all_tf_names.append(tf_name)
                all_peak_seqs.append(peak_data['seq'])
                
            print(f"TF {tf_name}: Found {len(non_overlap_peaks)} non-overlapping negative peaks from other TFs")
        
        # Create DataFrame with all collected negative samples
        neg_peak_df = pd.DataFrame({
            'tf_name': all_tf_names,
            'peak_seq': all_peak_seqs,
            'peak_fc': np.zeros(len(all_peak_seqs))
        })
        
        # # Optional: Limit the number of negatives per TF if needed
        # if hasattr(self, 'max_negatives_per_tf') and self.max_negatives_per_tf:
        #     print(f"Limiting to {self.max_negatives_per_tf} negatives per TF")
        #     neg_peak_df = neg_peak_df.groupby('tf_name').head(self.max_negatives_per_tf).reset_index(drop=True)
        
        return neg_peak_df

    def _dna_permutations(self, peak_fasta_paths, binding_peaks_csv):
        
        tf_names = []  # one to many
        peak_seqs = []
        peak_fc = []
        binding_starts = []
        binding_ends = []
        df = pd.read_csv(binding_peaks_csv, sep=',', header=None, index_col=False, usecols=[0, 2, 3, 8], names=['tf_name', 'binding_start', 'binding_end', 'binding_fc'])

        for p in tqdm(peak_fasta_paths):
            current_tf_name = peak_fa_path_parser(p)
            fa = pyfastx.Fasta(p, build_index=False)
            seqs = [seq for name, seq in fa]
            seqs = [partial_shuffle(seq, self.shuffle_fraction) for seq in seqs]
            tf_df = df[df['tf_name'] == current_tf_name].reset_index(drop=True)
            assert len(seqs) == len(tf_df)

            for i in range(len(seqs)):
                tf_names.append(current_tf_name)
                peak_seqs.append(seqs[i])
                peak_fc.append(tf_df.iloc[i]['binding_fc'])
                binding_starts.append(tf_df.iloc[i]['binding_start'])
                binding_ends.append(tf_df.iloc[i]['binding_end'])

        assert len(peak_seqs) == len(tf_names) == len(peak_fc) == len(binding_starts) == len(binding_ends)
        peak_df = pd.DataFrame({'tf_name': tf_names, 'peak_seq': peak_seqs, 'peak_fc': peak_fc, 'binding_start': binding_starts, 'binding_end': binding_ends})
        return peak_df    

    def _get_tf_seqs(self, tf_fasta_paths):
        tf_seqs = []
        tf_names = []
        for p in tqdm(tf_fasta_paths):
            fa = pyfastx.Fasta(p, build_index=False)
            seqs = [item[1] for item in fa]
            assert len(seqs) == 1
            seq = seqs[0]
            tf_seqs.append(seq.replace('\t', ''))
            tf_names.append(tf_fa_path_parser(p))
        tf_df = pd.DataFrame({'tf_name': tf_names, 'tf_seq': tf_seqs})
        return tf_df

    def _read_binding_csv(self, binding_peaks_csv):
        df = pd.read_csv(binding_peaks_csv, sep=',', header=None, index_col=False, usecols=[0, 2, 3, 8], names=['tf_name', 'binding_start', 'binding_end', 'binding_fc'])
        df['binding_start'] = df['binding_start'] - 1  # Convert to 0-base
        # print(df)
        return df

    def _read_tss(self, gff_path):
        df = pd.read_csv(gff_path, sep='\t', comment='#', header=None,
                         names=['seqid', 'source', 'type', 'start', 'end', 'score', 'strand', 'phase', 'attributes'])
        gene_df = df[df['type'] == 'gene']
        tss_positions = []
        for _, row in gene_df.iterrows():
            start = row['start'] - 1  # 1-based to 0-based
            end = row['end']  # 1-based inclusive, for slice end
            if row['strand'] == '+':
                tss = start
            elif row['strand'] == '-':
                tss = end - 1
            else:
                continue
            tss_positions.append((tss, row['strand']))
        return tss_positions

    def __len__(self):
        pos_len = len(self.binding_peaks_df)
        #neg_len = int(self.neg_sampling_rate *3* pos_len)
        neg_len = int(self.neg_sampling_rate * pos_len)
        return pos_len + neg_len




    def __getitem__(self, idx):
        if idx < len(self.peak_with_seqs_df):
            tf_name = self.peak_with_seqs_df.iloc[idx]['tf_name']
            tf_seq = self.peak_with_seqs_df.iloc[idx]['tf_seq']
            peak_seq = self.peak_with_seqs_df.iloc[idx]['peak_seq']
            peak_fc = self.peak_with_seqs_df.iloc[idx]['peak_fc']
            label = 1
        else:
            
            neg_idx = idx - len(self.peak_with_seqs_df)
            tf_name = self.neg_peak_with_seqs_df.iloc[neg_idx]['tf_name']
            tf_seq = self.neg_peak_with_seqs_df.iloc[neg_idx]['tf_seq']
            peak_seq = self.neg_peak_with_seqs_df.iloc[neg_idx]['peak_seq']
            peak_fc = self.neg_peak_with_seqs_df.iloc[neg_idx]['peak_fc']
            label = 0
        return tf_name, tf_seq, peak_seq, peak_fc, torch.tensor(label, dtype=torch.float32)

# if __name__ == '__main__':
#     tf_fasta_paths = glob.glob('/home/dylan/data/ml/PACHIP/trykleb/kp3/data/input/TF_fasta/*.fa')
#     tf_fasta_paths = [p for p in tf_fasta_paths if ':' not in p]
#     peak_fasta_paths = glob.glob('/home/dylan/data/ml/PACHIP/trykleb/kp3/data/input/peak_fasta_201bp_p3n30/*.fasta')
#     dataset = TFDataset(tf_fasta_paths, peak_fasta_paths, '/home/dylan/data/ml/PACHIP/trykleb/kp3/data/input/allpeak_p3n30.csv',
#                         '/home/dylan/data/ml/PACHIP/trykleb/kp3/data/input/allpeak_27TF.csv', '/home/dylan/data/ml/PACHIP/trykleb/kp3/data/hvkp4.fasta','/home/dylan/data/ml/PACHIP/trykleb/kp3/pred/HKU57.gff')
      
#     print(dataset[1])
#     print('done')