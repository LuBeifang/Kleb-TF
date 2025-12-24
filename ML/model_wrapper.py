import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel, EsmModel, EsmTokenizer
import torch
import glob
from tqdm.auto import tqdm
from torchmetrics.classification import BinaryAccuracy
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from scipy.stats import pearsonr
from nn_data_prep import TFDataset
from model import AttachedModel,create_attached_model
from captum.attr import InputXGradient 
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
    def __init__(self, protein_model_name="facebook/esm2_t6_8M_UR50D", dna_encoding_method="transformer", dna_max_length=None,namefile=None,protein_max_length=None,save_path=None,learning_rate=5e-6):
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
        self.namefile=namefile
        self.save_path=save_path
        self.learning_rate=learning_rate
    def calculate_peak_max_length(self, dataset):
        """Calculate the maximum length of peak sequences in the dataset"""
        max_length = 0
        print("Calculating maximum peak sequence length...")
        
        sample_size = min(1000, len(dataset)) 
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
            protein_seq = self.dna_converter.translate_dna_to_protein(dna_seq)
            if "prot_bert" in self.protein_tokenizer.name_or_path:
                protein_seq = ' '.join(protein_seq)
            protein_sequences.append(protein_seq)
        return protein_sequences
    def get_model_settings(self, dataset=None):
        # Calculate dna_max_length if not provided
        if self.dna_max_length is None and dataset is not None:
            self.dna_max_length = self.calculate_peak_max_length(dataset)
            # Add some padding for safety
            self.dna_max_length = min(self.dna_max_length, 2048)  
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
        
        self.protein_max_length = min(1024, self.protein_tokenizer.model_max_length,self.protein_max_length)

        # Load DNA processing method
        if self.dna_encoding_method == "transformer":
            from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
            self.dna_tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-50m-multi-species", trust_remote_code=True)
            self.dna_feature_extractor = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-50m-multi-species", trust_remote_code=True).to(device)
            # For transformer, respect the model's max length
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
        
        self.optimizer = torch.optim.AdamW(self.attached_model.parameters(), lr=self.learning_rate, fused=True)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
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
    
        elif self.dna_encoding_method == "onehot":
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
                outputs = self.protein_model(input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state
        return embeddings

    def train_batch(self, batch):
        self.attached_model.train()
        tf_name,tf_seqs, peak_seqs, peak_fcs, labels = batch
        
        # Convert TF DNA sequences to protein sequences
        protein_sequences = self.convert_dna_to_protein_batch(tf_seqs)
        
        # Get protein embeddings for TF sequences
        tf_embeddings = self.get_protein_embeddings(protein_sequences)
        
        # Get DNA embeddings for peak sequences
        peak_embeddings = self.get_dna_embeddings(peak_seqs)

        # Forward pass through attached model
        preds = self.attached_model(tf_embeddings, peak_embeddings)
        loss = self.loss_fn(preds.squeeze(), labels.squeeze().to(device))
        loss = torch.mean(loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), labels.mean()

    @torch.no_grad()
    def evaluate(self, val_loader):
        self.attached_model.eval()
        self.protein_model.eval()
        loss_list = []
        preds_list = []
        labels_list = []
        tf_seqs_list = []
        peak_seqs_list = []
        peak_fcs_list = []
        tf_name_list = []
        for batch in tqdm(val_loader, leave=True, position=0):
            tf_name, tf_seqs, peak_seqs, peak_fcs, labels = batch
            tf_name_list.extend(tf_name)
            tf_seqs_list.extend(tf_seqs)
            peak_seqs_list.extend(peak_seqs)
            peak_fcs_list.extend(peak_fcs)

            protein_sequences = self.convert_dna_to_protein_batch(tf_seqs)
            
            tf_embeddings = self.get_protein_embeddings(protein_sequences)

            peak_embeddings = self.get_dna_embeddings(peak_seqs)

            # Predictions
            preds = self.attached_model(tf_embeddings, peak_embeddings)
            preds_list.append(preds.squeeze().cpu().numpy())
            labels_list.append(labels.squeeze().cpu().numpy())

            loss = self.loss_fn(preds.squeeze(), labels.squeeze().to(device))
            loss = loss.mean()
            loss_list.append(loss.item())

        all_preds = np.concatenate(preds_list)
        all_labels = np.concatenate(labels_list)
        acc = BinaryAccuracy(threshold=0.8)(torch.tensor(all_preds), torch.tensor(all_labels))
        auc = roc_auc_score(all_labels, all_preds)
        auprc = average_precision_score(all_labels, all_preds)
        
        return (np.mean(np.array(loss_list)), acc.cpu().numpy(), auc, auprc, 
                all_labels.mean().astype(np.float64), all_preds, all_labels, tf_name_list,
                tf_seqs_list, peak_seqs_list, peak_fcs_list)
    def train(self, train_dataset, val_dataset, test_dataset, batch_size, epochs=35):

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
        )

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True)

        best_auc = 0
        best_cor = 0
        history = []
        for epoch in list(range(1, epochs+1)):
            loss_list = []
            label_list = []
            for batch in tqdm(train_loader, leave=True, position=0, desc=f'Epoch {epoch}'):
                loss, mean_label = self.train_batch(batch)
                loss_list.append(loss)
                label_list.append(mean_label)

            
            val_loss, val_acc, val_auc, val_auprc, val_label,test_preds, test_labels,tf_name_list, test_tf_seqs_list, test_peak_seqs_list, test_peak_fcs_list = self.evaluate(val_loader)
            # val_loss, val_label, val_cor, val_pvalue = self.evaluate(val_loader)

            self.scheduler.step(val_loss)
            current_lr = self.scheduler.get_last_lr()[0]  # 获取当前学习率

            print(f"Epoch {epoch}: Learning Rate = {current_lr}")
            print(f'  Train Loss: {np.mean(loss_list):.4f}, mean labels: {np.mean(label_list):.4f}')
            print(f'  Val Loss: {val_loss:.4f}, Val Labels: {val_label:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}, Val AUPRC: {val_auprc:.4f}')
            # print(f'  Val Loss: {val_loss:.4f}, Val Labels: {val_label:.4f}, Val cor: {val_cor:.4f}, val p-value: {val_pvalue:.4e}')

            history.append({
                'epoch': epoch,
                'learning_rate': current_lr ,
                'train_loss': np.mean(loss_list),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_auc': val_auc,
                'val_auprc':val_auprc,
                # 'val_cor': val_cor,
                # 'val_pvalue' : val_pvalue,
            })

            if val_auc > best_auc:
                best_auc = val_auc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.attached_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_auc': val_auc,
                },os.path.join(self.save_path, f'best_model.pt'))
                print(f'  The best model saved (AUC: {val_auc:.4f})')

            df_history = pd.DataFrame(history)
            df_history.to_csv(os.path.join(self.save_path, f'training_history_{self.namefile}.csv'), index=False)

        # TEST
        test_results = []
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)
        final_test_loss, final_test_acc, final_test_auc, final_test_auprc, final_test_label,test_preds, test_labels, tf_name_list, test_tf_seqs_list, test_peak_seqs_list, test_peak_fcs_list = self.evaluate(test_loader)
        # final_test_loss, final_test_label, final_test_cor, final_test_pvalue = self.evaluate(test_loader)
        print(f'\nFinal Test Performance:')
        print(f'  Loss: {final_test_loss:.4f}, label: {final_test_label:.4f}  Acc: {final_test_acc:.4f}, AUC: {final_test_auc:.4f}, AUPRC: {final_test_auprc:.4f}')
        # print(f'  Loss: {final_test_loss:.4f}, label: {final_test_label:.4f}  Cor: {final_test_cor:.4f}, p-value: {final_test_pvalue:.4e}')
        test_results.append({
            'loss': final_test_loss,
            'label': final_test_label,
            # 'pearson_r': final_test_cor,
            # 'p_value': final_test_pvalue,
            'acc': final_test_acc,
            'auc': final_test_auc,
            'auprc': final_test_auprc
            })
        test_df = pd.DataFrame(test_results)
        test_df.to_csv(os.path.join(self.save_path, f'test_results_{self.namefile}.csv'), index=False)
        


    def test(self, test_dataset):
        self.attached_model.eval()
        test_results = []
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle = False)
        final_test_loss, final_test_acc, final_test_auc, final_test_auprc, final_test_label, test_preds, test_labels,tf_name_list, test_tf_seqs_list, test_peak_seqs_list, test_peak_fcs_list = self.evaluate(test_loader)

        print(f'\nFinal Test Performance:')
        print(f'  Loss: {final_test_loss:.4f}, label: {final_test_label:.4f}  Acc: {final_test_acc:.4f}, AUC: {final_test_auc:.4f}, AUPRC: {final_test_auprc:.4f}')

        test_results.append({
            'loss': final_test_loss,
            'label': final_test_label,
            'acc': final_test_acc,
            'auc': final_test_auc,
            'auprc': final_test_auprc
            })
        test_df = pd.DataFrame(test_results)
        test_df.to_csv(os.path.join(self.save_path, f'test_results_{self.namefile}.csv'), index=False)

        test_matrix = pd.DataFrame({
            "TF_name": tf_name_list,
            'TF_seqs': test_tf_seqs_list,
            'pesked_seqs': test_peak_seqs_list,
            'peak_fcs': [i.numpy() for i in test_peak_fcs_list],
            'Labels': test_labels,
            'Predictions': test_preds,
        })
        test_matrix.to_csv(os.path.join(self.save_path, f'test_matrix_{self.namefile}.csv'), index=False)
        print('test matrix saved')
    def load_model(self, model_path):
        checkpoint = torch.load(model_path, weights_only=False)
        self.attached_model.load_state_dict(checkpoint['model_state_dict'])
        self.attached_model.eval()

        
    def compute_motif_weights(self, tf_seqs, peak_seqs, target=0, motif_window=12, normalize=True):
        protein_sequences = self.convert_dna_to_protein_batch(tf_seqs)

            # Get protein embeddings for TF sequences
        tf_embeddings = self.get_protein_embeddings(protein_sequences)
    
        # Get DNA embeddings for peak sequences
        peak_embeddings = self.dna_onehot_encoder.encode_sequence(peak_seqs)
        peak_embeddings = torch.from_numpy(peak_embeddings).float().unsqueeze(0).to(device)
            
        if self.dna_encoding_method != "onehot":
            raise ValueError("Motif weights computation is supported only for onehot encoding.")
        
        if peak_embeddings.shape[1] != 220:
            raise ValueError("Expected DNA sequence length of 220.")
        
        self.attached_model.eval()  # Set to eval mode
        peak_embeddings = peak_embeddings.clone().requires_grad_(True).to(device)
        tf_embeddings = tf_embeddings.to(device)
        # Forward pass (use sigmoid=False for logits)
        logits = self.attached_model(tf_embeddings, peak_embeddings, sigmoid=False)
        # Enformer-style Input x Gradient using Captum
        attr_method = InputXGradient(self.attached_model)
        attributions = attr_method.attribute(
            inputs=(tf_embeddings, peak_embeddings),
            target=target,
            additional_forward_args=False
        )
        peak_attr = attributions[1]  # (B, 220, 5)
        
        # Aggregate per base: sum over 5 channels (A/T/G/C/N)
        per_base_weights = peak_attr.abs().sum(dim=-1)  # (B, 220)
        
        if normalize:
            per_base_weights = per_base_weights / per_base_weights.max(dim=1, keepdim=True)[0]
        
        # Aggregate for motifs: sliding window average (to highlight motif-like regions)
        B, seq_len = per_base_weights.shape
        motif_weights = torch.zeros(B, seq_len - motif_window + 1, device=device)
        for i in range(seq_len - motif_window + 1):
            motif_weights[:, i] = per_base_weights[:, i:i+motif_window].mean(dim=1)
        
        # Find top motif regions (e.g., top 5 highest-weight windows per sample)
        top_motifs = []
        for b in range(B):
            top_indices = torch.topk(motif_weights[b], k=5).indices
            top_motifs.append({
                'sample_index': b,
                'top_regions': [(idx.item(), idx.item() + motif_window) for idx in top_indices],
                'scores': motif_weights[b, top_indices].tolist()
            })
        
        return {
            'per_base_weights': per_base_weights,
            'motif_weights': motif_weights,
            'top_motifs': top_motifs
        }
