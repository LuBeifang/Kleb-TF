import pandas as pd
from BCBio import GFF
from torch.utils.data import Dataset
import pyfastx
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
import argparse
import subprocess
from models import AttachedModel



parser = argparse.ArgumentParser(description='Predict TF and DNA binding')
parser.add_argument('-tf','--input_tf_fasta', type=str, help='Directory path of input TF fasta file',default='')
parser.add_argument('-g', '--genome', type=str, help='Directory path of input KP isolates genome fasta file',default='')
parser.add_argument('-gff', type=str, help='Directory path of input KP isolates gene annotation gff file',default='')
# parser.add_argument('--model', type=str, help='Path to the model file',default='/media/lu/lussd/kp_tf/best_model.pt')
parser.add_argument('-o','--output', type=str, help='Output csv path',default='output.csv')
parser.add_argument('-device', type=str, help='Device name: cuda or cpu', default='cuda')
parser.add_argument('--motif', action='store_true', help='Run motif at the same time')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dna_tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-50m-multi-species", trust_remote_code=True)
dna_feature_extractor = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-50m-multi-species", trust_remote_code=True).to(device)
dna_max_length = dna_tokenizer.model_max_length

class PromoterDataset(Dataset):
    def __init__(self, tf_fasta_path, genome_fasta_path, gff_path, length=201):
        self.sequences = []
        self.labels = []  # Assuming there's a label or similar use-case
        whole_genome = pyfastx.Fasta(genome_fasta_path)
        whole_genome = ''.join([item.seq for item in whole_genome])

        # features_list = []
        for rec in GFF.parse(gff_path):
            for feature in rec.features:
                chrom = rec.id
                start = feature.location.start
                end = feature.location.end
                strand = feature.location.strand
                strand = '+' if strand == 1 else '-' if strand == -1 else '.'

                if 'gene' in feature.id:

                    if start > length and end > length:
                        promoter_seq = whole_genome[start-length:start] if strand == '+' else whole_genome[end:end+length]
                        self.sequences.append(promoter_seq)
                        self.labels.append(feature.id)  # Example placeholder

                # features_list.append({'chromosome': chrom, 'gene_name': name, 'start': start, 'end': end, 'strand': strand, 'promoter_seq': seq})

    # gff_df = pd.DataFrame(features_list)
    # gff_df = gff_df[gff_df['gene_name'].str.contains('gene')]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def compute_mean_embeddings_per_sequence(embeddings, attention_mask):
    # Add embed dimension axis
    attention_mask = torch.unsqueeze(attention_mask, dim=-1)

    # Compute mean embeddings per sequence
    mean_sequence_embeddings = torch.sum(attention_mask * embeddings, axis=-2) / torch.sum(attention_mask, axis=1)
    return mean_sequence_embeddings


def get_tf_embeddings(tf_fasta_path, device = 'cuda'):
    # embedding setting
    tf_tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-50m-3mer-multi-species", trust_remote_code=True)
    tf_feature_extractor = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-50m-3mer-multi-species", trust_remote_code=True).to(device)
    tf_max_length = tf_tokenizer.model_max_length

    tf_seqs = pyfastx.Fasta(tf_fasta_path, build_index=False)
    tf_seqs = [item[1] for item in tf_seqs]
    tf_token_ids = tf_tokenizer.batch_encode_plus(tf_seqs, return_tensors="pt", padding="max_length", max_length=tf_max_length)[
        "input_ids"].to(device)

    tf_attention_mask = tf_token_ids != tf_tokenizer.pad_token_id
    tf_llm_outputs = tf_feature_extractor(
        tf_token_ids,
        attention_mask=tf_attention_mask,
        encoder_attention_mask=tf_attention_mask,
        output_hidden_states=True
    )
    tf_embeddings = tf_llm_outputs['hidden_states'][-1].detach()
    tf_embeddings = compute_mean_embeddings_per_sequence(tf_embeddings, tf_attention_mask)

    return tf_embeddings

def get_pro_embeddings(promoter_seqs, device = 'cuda'):
    global dna_tokenizer, dna_feature_extractor, dna_max_length

    peak_seqs = promoter_seqs
    peak_token_ids = dna_tokenizer.batch_encode_plus(peak_seqs, return_tensors="pt", padding="max_length", max_length=dna_max_length)[
        "input_ids"].to(device)

    peak_attention_mask = peak_token_ids != dna_tokenizer.pad_token_id
    peak_llm_outputs = dna_feature_extractor(
        peak_token_ids,
        attention_mask=peak_attention_mask,
        encoder_attention_mask=peak_attention_mask,
        output_hidden_states=True
    )
    peak_embeddings = peak_llm_outputs['hidden_states'][-1].detach()
    peak_embeddings = compute_mean_embeddings_per_sequence(peak_embeddings, peak_attention_mask)

    return peak_embeddings


def predict_promoters_batch(model, promoterDataset, tf_embeddings, batch_size = 2):
    model.eval()
    gene_names = []
    sequences = []
    predictions = []
    pred_loader = torch.utils.data.DataLoader(promoterDataset, batch_size, shuffle=False)

    # count = 0
    # max_batches = 20

    for batch in tqdm(pred_loader, leave=True, position=0):
        pro_seqs, labels = batch
        peak_embeddings = get_pro_embeddings(pro_seqs)
        preds = model(tf_embeddings.repeat(batch_size,1), peak_embeddings)
        predictions.extend(preds.cpu().detach().numpy())# Assuming some form of regression/classification
        gene_names.extend(labels)
        sequences.extend(pro_seqs)

        # count += 1
        # if count >= max_batches:
        #     break
    predictions = [item[0] for item in predictions]
    final_predictions = pd.DataFrame({'Gene Name': gene_names,
                                      'Promoter sequence': sequences,
                                      'Prediction': predictions})

    return final_predictions

def run_meme(fasta_path, meme_output_dir):
    cmd = ["meme", fasta_path, "-dna", "-nostatus", "-time", "14400", "-mod", "zoops",
           "-nmotifs", "5", "-minw", "6", "-maxw", "20",  "-revcomp", "-markov_order", "0", "-oc", meme_output_dir]
    subprocess.run(cmd)

def save_sequences_to_fasta(sequences, fasta_path):
    with open(fasta_path, 'w') as fasta_file:
        for idx, seq in enumerate(sequences):
            fasta_file.write(f">Sequence_{idx}\n{seq}\n")


# if __name__ == "__main__":
#     tf_fasta_path = '/media/lu/lussd/kp_tf/pred/RS12175.fa'
#     tf_embeddings = get_tf_embeddings(tf_fasta_path)
#     genome_fasta_path = '/media/lu/lussd/kp_tf/pred/HKU57.fna'
#     gff_path = '/media/lu/lussd/kp_tf/pred/HKU57.gff'
#
#     model = AttachedModel()
#     model = model.to(device)
#     checkpoint = torch.load('/media/lu/lussd/kp_tf/best_model_96TFepoch20_0626.pt', weights_only=False)
#     model.load_state_dict(checkpoint['model_state_dict'])
#
#     promoterDataset = PromoterDataset(tf_fasta_path, genome_fasta_path, gff_path)
#
#     preds = predict_promoters_batch(model, promoterDataset, tf_embeddings)
#     preds.to_csv('/media/lu/lussd/kp_tf/pred/RS12175_HKU57_0626.csv',index=False)


def main(tf_fasta_path, genome_fasta_path, gff_path, device, output_path, motif):
    tf_embeddings = get_tf_embeddings(tf_fasta_path)
    torch.cuda.empty_cache()
    model = AttachedModel()
    model = model.to(device)
    checkpoint = torch.load('/media/lu/lussd/kp_tf/trained_models/best_model_96TF_test0824.pt', weights_only=False)
    # print(checkpoint.keys())
    model.load_state_dict(checkpoint['model_state_dict'])

    promoterDataset = PromoterDataset(tf_fasta_path, genome_fasta_path, gff_path)

    preds = predict_promoters_batch(model, promoterDataset, tf_embeddings)
    preds.to_csv(output_path,index=False)

    if args.motif:
        high_confidence_seqs = preds[preds['Prediction'] > 0.7]['Promoter sequence']
        save_sequences_to_fasta(high_confidence_seqs, output_path + "seqs.fasta")

        # Run MEME
        run_meme(output_path + "seqs.fasta", output_path + "_motif")


if __name__ == "__main__":
    main(args.input_tf_fasta, args.genome, args.gff, args.device, args.output, args.motif)





