import sys
import os

# Add parent directory for imports from predictionhead/
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import pandas as pd
from BCBio import GFF
from torch.utils.data import Dataset
import pyfastx
import numpy as np
import torch
from tqdm.auto import tqdm
import argparse

from model import create_model
from model_wrapper import ModelWrapper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- 96 TF name -> ID mapping (sorted alphabetically, matches training) ----
SORTED_96_TF_NAMES = [
    'RS00500','RS00705','RS00760','RS00775','RS01665','RS01765','RS03250',
    'RS03255','RS03555','RS03725','RS03755','RS03890','RS04220','RS04780',
    'RS04810','RS05775','RS05825','RS05835','RS06015','RS06080','RS06165',
    'RS06825','RS07010','RS07305','RS07655','RS07795','RS07825','RS08105',
    'RS08330','RS09070','RS09095','RS09385','RS09805','RS09905','RS09940',
    'RS10135','RS10220','RS10370','RS10420','RS10575','RS10685','RS10755',
    'RS10980','RS11020','RS11395','RS12120','RS12175','RS12630','RS12945',
    'RS12950','RS13070','RS13215','RS13395','RS13415','RS14185','RS15210',
    'RS15265','RS15770','RS16240','RS16310','RS17175','RS17375','RS17425',
    'RS17565','RS17600','RS18110','RS18490','RS19350','RS19910','RS20195',
    'RS20415','RS20525','RS20810','RS20845','RS20865','RS20885','RS21105',
    'RS21310','RS21375','RS21390','RS21455','RS21505','RS21600','RS21785',
    'RS21815','RS22025','RS22030','RS22075','RS22355','RS23270','RS23320',
    'RS24705','RS25850','RS26015','RS26715','RS26760',
]
tf_name_to_id = {name: i for i, name in enumerate(SORTED_96_TF_NAMES)}


class PromoterDataset(Dataset):
    def __init__(self, tf_fasta_path, genome_fasta_path, gff_path, length=201):
        self.sequences = []
        self.labels = []
        tf = pyfastx.Fasta(tf_fasta_path)
        self.tf_seq = tf.longest.seq
        raw_name = tf.longest.name
        self.tf_name = raw_name.split()[0] if raw_name else 'unknown'
        self.tf_id = tf_name_to_id.get(self.tf_name, 0)

        whole_genome = pyfastx.Fasta(genome_fasta_path)
        whole_genome = ''.join([item.seq for item in whole_genome])

        for rec in GFF.parse(gff_path):
            for feature in rec.features:
                start = feature.location.start
                end = feature.location.end
                strand = feature.location.strand
                strand = '+' if strand == 1 else '-' if strand == -1 else '.'
                if 'gene' in feature.id:
                    if start > length and end > length:
                        promoter_seq = (
                            whole_genome[start-length:start]
                            if strand == '+'
                            else whole_genome[end:end+length]
                        )
                        self.sequences.append(promoter_seq)
                        self.labels.append(feature.id)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.tf_name, self.tf_seq, self.sequences[idx], self.labels[idx], self.tf_id


def my_collate_fn(batch):
    tf_names, tf_seqs, peak_seqs, labels, tf_ids = zip(*batch)
    return list(tf_names), list(tf_seqs), list(peak_seqs), list(labels), list(tf_ids)


def predict(model_wrapper, dataset):
    model_wrapper.attached_model.eval()
    model_wrapper.protein_model.eval()

    tf_name_list, tf_seqs_list = [], []
    peak_seqs_list, peaks_label_list, preds_list = [], [], []

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=my_collate_fn)

    for tf_names, tf_seqs, peak_seqs, peak_labels, tf_ids in tqdm(loader, leave=True, position=0):
        tf_name_list.extend(tf_names)
        tf_seqs_list.extend(tf_seqs)
        peak_seqs_list.extend(peak_seqs)
        peaks_label_list.extend(peak_labels)

        tf_embeddings, protein_mask = model_wrapper.get_protein_embeddings(tf_seqs)
        peak_embeddings = model_wrapper.get_dna_embeddings(peak_seqs)

        tf_id_tensor = torch.tensor(tf_ids, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model_wrapper.attached_model(
                tf_embeddings, peak_embeddings, tf_id_tensor, protein_mask=protein_mask
            )
        probs = 1.0 / (1.0 + np.exp(-logits.squeeze().cpu().numpy()))
        preds_list.append(probs)

    all_preds = np.concatenate(preds_list) if isinstance(preds_list[0], np.ndarray) else np.array(preds_list)

    return pd.DataFrame({
        "TF_name": tf_name_list,
        'TF_seqs': tf_seqs_list,
        'pesked_seqs': peak_seqs_list,
        'Labels': peaks_label_list,
        'Predictions': all_preds,
    })


def sequences_to_fasta(df, sequence_column, output_file):
    with open(output_file, 'w') as f:
        for i, seq in enumerate(df[sequence_column]):
            f.write(f">sequence_{i+1}\n")
            f.write(f"{seq}\n")
    print('Predicted sequences saved to', output_file)


def main(args):
    model_wrapper = ModelWrapper(
        protein_model_name="ESM_DBP",
        dna_encoding_method="onehot",
        dna_max_length=201,
        protein_max_length=903,
        num_tfs=96,
        tf_name_to_id=tf_name_to_id,
        model_creator=create_model,
        aux_weight=0.3,
    )
    model_wrapper.get_model_settings()
    model_wrapper.load_model(args.model)

    fasta_file = os.path.join(args.output, 'Predicted_sequences.fasta')

    promoterDataset = PromoterDataset(
        args.input_tf_fasta, args.genome, args.gff, length=201
    )

    if promoterDataset.tf_name not in tf_name_to_id:
        print(
            f"WARNING: TF '{promoterDataset.tf_name}' not in the 96-TF mapping. "
            f"Falling back to head 0. Predictions may be unreliable."
        )

    preds = predict(model_wrapper, promoterDataset)
    preds.to_csv(os.path.join(args.output, 'predictions_matrix.csv'), index=False)
    print(f'max Predictions {preds.Predictions.max()}')

    sequences_to_fasta(
        preds[preds.Predictions >= args.threshold], 'pesked_seqs', fasta_file
    )
    print(f'Matrix saved to {os.path.join(args.output, "predictions_matrix.csv")}')


parser = argparse.ArgumentParser(description='Predict TF and DNA binding')
parser.add_argument('-tf', '--input_tf_fasta', type=str, help='Directory path of input TF fasta file', default='')
parser.add_argument('-g', '--genome', type=str, help='Directory path of input KP isolates genome fasta file', default='')
parser.add_argument('-gff', type=str, help='Directory path of input KP isolates gene annotation gff file', default='')
parser.add_argument('-model', type=str, help='Path to the model file', default='best_model.pt')
parser.add_argument('-t', '--threshold', type=float, help='Prediction threshold', default=0.8)
parser.add_argument('-o', '--output', type=str, help='Output directory path', default='output')
args = parser.parse_args()


if __name__ == "__main__":
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    main(args)
