import pandas as pd
from BCBio import GFF
from torch.utils.data import Dataset
import pyfastx
import os
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from model import create_attached_model
import argparse


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DNAToProteinConverter:
    """Convert DNA sequences to protein sequences"""
    
    def __init__(self):
        # Standard genetic code
        self.codon_table = {
            'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
            'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
            'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
            'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
            'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
            'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
            'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
            'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
            'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
            'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
            'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
        }
    
    def translate_dna_to_protein(self, dna_seq):
        """Translate DNA sequence to protein sequence"""
        dna_seq = dna_seq.upper().replace('U', 'T')  # Convert RNA to DNA if needed
        
        # Try all three reading frames and choose the longest ORF
        best_protein = ""
        
        for frame in range(3):
            protein = ""
            for i in range(frame, len(dna_seq) - 2, 3):
                codon = dna_seq[i:i+3]
                if len(codon) == 3:
                    amino_acid = self.codon_table.get(codon, 'X')  # X for unknown
                    if amino_acid == '*':  # Stop codon
                        break
                    protein += amino_acid
            
            # Keep the longest protein sequence
            if len(protein) > len(best_protein):
                best_protein = protein
        

        
        return best_protein if best_protein else "M"  # Return at least one amino acid

class DNAOneHotEncoder:
    """One-hot encoding for DNA sequences"""
    
    def __init__(self, max_length=1024):
        self.max_length = max_length
        self.nucleotide_map = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
        self.num_nucleotides = len(self.nucleotide_map)
    
    def encode_sequence(self, sequence):
        """Encode a single DNA sequence to one-hot"""
        sequence = sequence.upper()
        # Pad or truncate to max_length
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        else:
            sequence = sequence.ljust(self.max_length, 'N')
        
        # Create one-hot encoding
        encoded = np.zeros((self.max_length, self.num_nucleotides), dtype=np.float32)
        for i, nucleotide in enumerate(sequence):
            if nucleotide in self.nucleotide_map:
                encoded[i, self.nucleotide_map[nucleotide]] = 1.0
            else:
                # Unknown nucleotide, set N (index 4)
                encoded[i, 4] = 1.0
        
        return encoded
    
    def encode_batch(self, sequences):
        """Encode a batch of DNA sequences"""
        batch_size = len(sequences)
        encoded_batch = np.zeros((batch_size, self.max_length, self.num_nucleotides), dtype=np.float32)
        
        for i, seq in enumerate(sequences):
            encoded_batch[i] = self.encode_sequence(seq)
        
        return torch.tensor(encoded_batch)
    

    



class ModelWrapper:
    def __init__(self, protein_model_name="facebook/esm2_t6_8M_UR50D", dna_encoding_method="transformer", dna_max_length=None,amino_or_dna='dna',protein_max_length=1024):
        self.protein_model_name = protein_model_name
        self.protein_tokenizer = None
        self.protein_model = None
        self.protein_max_length = protein_max_length
        self.dna_tokenizer = None
        self.dna_feature_extractor = None
        self.dna_max_length = dna_max_length  
        self.attached_model = None
        self.optimizer = None
        self.loss_fn = None
        self.scheduler = None
        self.dna_converter = DNAToProteinConverter()
        self.dna_encoding_method = dna_encoding_method
        self.amino_or_dna=amino_or_dna
    def calculate_peak_max_length(self, dataset):
        """Calculate the maximum length of peak sequences in the dataset"""
        max_length = 0
        print("Calculating maximum peak sequence length...")
        
        # Sample a subset if dataset is very large to avoid long computation
        sample_size = min(1000, len(dataset))  # Sample up to 1000 sequences
        indices = np.random.choice(len(dataset), sample_size, replace=False)
        
        for i in tqdm(indices, desc="Scanning peak sequences"):
            try:
                _, peak_seq, _, _ = dataset[i]
                if isinstance(peak_seq, str):
                    max_length = max(max_length, len(peak_seq))
                elif isinstance(peak_seq, list):
                    max_length = max(max_length, len(peak_seq[0]) if peak_seq else 0)
            except Exception as e:
                print(f"Error processing sequence {i}: {e}")
                continue
        
        print(f"Maximum peak sequence length found: {max_length}")
        return max_length
    def convert_dna_to_protein_batch(self, dna_sequences):
        """Convert batch of DNA sequences to protein sequences"""
        protein_sequences = []
        for dna_seq in dna_sequences:
            if self.amino_or_dna == 'amino':
                protein_seq = dna_seq  # Already amino acid sequence
            else:
                protein_seq = self.dna_converter.translate_dna_to_protein(dna_seq)
            # Add spaces between amino acids for some tokenizers (like ProtBERT)
            if "prot_bert" in self.protein_tokenizer.name_or_path:
                protein_seq = ' '.join(protein_seq)
            protein_sequences.append(protein_seq)
        return protein_sequences
    def get_model_settings(self, dataset=None):
        # Calculate dna_max_length if not provided
        if self.dna_max_length is None and dataset is not None:
            self.dna_max_length = self.calculate_peak_max_length(dataset)
            # Add some padding for safety
            self.dna_max_length = min(self.dna_max_length + 50, 2048)  # Cap at reasonable limit
            print(f"Using dna_max_length: {self.dna_max_length}")
        elif self.dna_max_length is None:
            # Fallback to default if no dataset provided
            self.dna_max_length = 1024
            print(f"No dataset provided, using default dna_max_length: {self.dna_max_length}")

        # Load protein model
        if "esm" in self.protein_model_name:
            from transformers import EsmTokenizer, EsmModel
            self.protein_tokenizer = EsmTokenizer.from_pretrained(self.protein_model_name)
            self.protein_model = EsmModel.from_pretrained(self.protein_model_name).to(device)
        else:
            from transformers import AutoTokenizer, AutoModel
            self.protein_tokenizer = AutoTokenizer.from_pretrained(self.protein_model_name)
            self.protein_model = AutoModel.from_pretrained(self.protein_model_name).to(device)
        
        self.protein_max_length = min(self.protein_max_length, self.protein_tokenizer.model_max_length)

        # Load DNA processing method
        if self.dna_encoding_method == "transformer":
            from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
            self.dna_tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-50m-multi-species", trust_remote_code=True)
            self.dna_feature_extractor = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-50m-multi-species", trust_remote_code=True).to(device)
            self.dna_max_length = min(self.dna_max_length, self.dna_tokenizer.model_max_length)
            print(f"Transformer dna_max_length adjusted to: {self.dna_max_length}")
        elif self.dna_encoding_method == "onehot":
            self.dna_onehot_encoder = DNAOneHotEncoder(max_length=self.dna_max_length)
            self.dna_feature_extractor = None
            print(f"One-hot encoding using dna_max_length: {self.dna_max_length}")
        else:
            raise ValueError(f"Unknown DNA encoding method: {self.dna_encoding_method}")

        # Create appropriate model
        self.attached_model = create_attached_model(
            self.protein_model_name,
            dna_encoding_method=self.dna_encoding_method,
            dna_max_length=self.dna_max_length
        ).to(device)
        
        self.optimizer = torch.optim.AdamW(self.attached_model.parameters(), lr=1e-4, fused=True)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=7, min_lr=1e-6)
        self.loss_fn = torch.nn.BCELoss(reduction='none')
    
    def get_dna_embeddings(self, dna_sequences):
        """Get DNA embeddings using the appropriate method"""
        if self.dna_encoding_method == "transformer":
            # Use transformer-based encoding
            dna_token_ids = self.dna_tokenizer.batch_encode_plus(
                dna_sequences, 
                return_tensors="pt", 
                padding="max_length", 
                max_length=self.dna_max_length
            )["input_ids"].to(device)

            with torch.no_grad():
                attention_mask = dna_token_ids != self.dna_tokenizer.pad_token_id
                dna_llm_outputs = self.dna_feature_extractor(
                    dna_token_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                dna_embeddings = dna_llm_outputs['hidden_states'][-1].detach()
                #dna_embeddings = self.compute_mean_embeddings_per_sequence(dna_embeddings, attention_mask) #no mean no learn i think
        
        elif self.dna_encoding_method == "onehot":
            # Use one-hot encoding
            dna_embeddings = self.dna_onehot_encoder.encode_batch(dna_sequences).to(device)
            
        return dna_embeddings
    
    def get_protein_embeddings(self, protein_sequences):
        """Get protein embeddings using protein language model"""
        # Tokenize protein sequences
        protein_tokens = self.protein_tokenizer.batch_encode_plus(
            protein_sequences, 
            return_tensors="pt", 
            padding="max_length", 
            max_length=self.protein_max_length,
            truncation=True
        )
        
        input_ids = protein_tokens["input_ids"].to(device)
        attention_mask = protein_tokens["attention_mask"].to(device)
        
        with torch.no_grad():
            if "esm" in self.protein_tokenizer.name_or_path:
                # ESM models
                outputs = self.protein_model(input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state
            else:
                # Other protein models (ProtBERT, ProtT5, etc.)
                outputs = self.protein_model(input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state
        
        # Compute mean embeddings
        #mean_embeddings = self.compute_mean_embeddings_per_sequence(embeddings, attention_mask)
        return embeddings
    @torch.no_grad()
    def predict(self, dataset):
        self.attached_model.eval()
        self.protein_model.eval()
        tf_name_list, tf_seqs_list, peak_seqs_list, preds_list, peaks_label_list = [], [], [], [], []

        loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=my_collate_fn)

        for tf_name, tf_seqs, peak_seqs, peak_labels in tqdm(loader, leave=True, position=0):
            tf_name_list.extend(tf_name)       # 现在是 list，不是 tuple
            tf_seqs_list.extend(tf_seqs)
            peak_seqs_list.extend(peak_seqs)
            peaks_label_list.extend(peak_labels)

            # 后续：计算 embedding / 预测时，注意这些是 list[str]
            protein_sequences = self.convert_dna_to_protein_batch(tf_seqs)
            tf_embeddings = self.get_protein_embeddings(protein_sequences)
            peak_embeddings = self.get_dna_embeddings(peak_seqs)
            preds = self.attached_model(tf_embeddings, peak_embeddings)
            preds_list.append(preds.squeeze().cpu().numpy())

        all_preds = preds_list
        return  pd.DataFrame({
            "TF_name": tf_name_list,
            'TF_seqs': tf_seqs_list,
            'pesked_seqs': peak_seqs_list,
            'Labels': peaks_label_list,
            'Predictions': all_preds,
        })
    def load_model(self, model_path):
        checkpoint = torch.load(model_path, weights_only=False)
        self.attached_model.load_state_dict(checkpoint['model_state_dict'])
        self.attached_model.eval()

class PromoterDataset(Dataset):
    def __init__(self, tf_fasta_path, genome_fasta_path, gff_path, length=201):
        self.sequences = []
        self.labels = []
        # 如果每个样本都用同一个 TF，直接存字符串（最清晰）
        tf = pyfastx.Fasta(tf_fasta_path)
        self.tf_seq = tf.longest.seq        # 字符串
        self.tf_name = tf.longest.name      # 字符串

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
        # 直接返回字符串与标签字符串
        return self.tf_name, self.tf_seq, self.sequences[idx], self.labels[idx]

def my_collate_fn(batch):
    # batch: list of tuples -> unzip
    tf_names, tf_seqs, peak_seqs, labels = zip(*batch)  # 得到 tuple
    # 转成 list（便于后续 extend）
    return list(tf_names), list(tf_seqs), list(peak_seqs), list(labels)


def main(args):
    def sequences_to_fasta(df, sequence_column, output_file):
        """
        Convert sequences from pandas column to FASTA format
        """
        with open(output_file, 'w') as f:
            for i, seq in enumerate(df[sequence_column]):
                f.write(f">sequence_{i+1}\n")
                f.write(f"{seq}\n")
        print('Predicted sequences saved to', output_file)
    model_wrapper = ModelWrapper(
        protein_model_name="facebook/esm2_t6_8M_UR50D",
        dna_encoding_method='transformer',
        dna_max_length=40,
        protein_max_length=903
    )
    model_wrapper.get_model_settings()

    model_wrapper.load_model(args.model)

    tf_fasta_path = args.input_tf_fasta
    genome_fasta_path = args.genome
    gff_path = args.gff
    fasta_file=os.path.join(args.output,'Predicted_sequences.fasta')
    promoterDataset = PromoterDataset(tf_fasta_path, genome_fasta_path, gff_path,length=201)
    preds = model_wrapper.predict(promoterDataset)
    preds.to_csv(os.path.join(args.output, 'predictions_matrix.csv'),index=False)
    print(f'max Predictions {preds.Predictions.max()}')
    sequences_to_fasta(preds[preds.Predictions>=args.threshold], 'pesked_seqs', fasta_file)
    print(f'Matrix saved to {os.path.join(args.output, "predictions_matrix.csv")}')


parser = argparse.ArgumentParser(description='Predict TF and DNA binding')
parser.add_argument('-tf','--input_tf_fasta', type=str, help='Directory path of input TF fasta file',default='')
parser.add_argument('-g', '--genome', type=str, help='Directory path of input KP isolates genome fasta file',default='')
parser.add_argument('-gff', type=str, help='Directory path of input KP isolates gene annotation gff file',default='')
parser.add_argument('-model', type=str, help='Path to the model file',default='best_model.pt')
parser.add_argument('-t','--threshold', type=float, help='Prediction threshold',default=0.8)
parser.add_argument('-o','--output', type=str, help='Output directory path',default='output')
args = parser.parse_args()



if __name__ == "__main__":
    if os.path.exists(args.output)==False:
        os.makedirs(args.output)
    main(args)

