import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel, EsmModel, EsmTokenizer
import torch
from tqdm.auto import tqdm
from torchmetrics.classification import BinaryAccuracy
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, f1_score
from model import create_model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DNAOneHotEncoder:    
    def __init__(self, max_length=1024):
        self.max_length = max_length
        self.nucleotide_map = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
        self.num_nucleotides = len(self.nucleotide_map)
    
    def encode_sequence(self, sequence):
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
        batch_size = len(sequences)
        encoded_batch = np.zeros((batch_size, self.max_length, self.num_nucleotides), dtype=np.float32)
        
        for i, seq in enumerate(sequences):
            encoded_batch[i] = self.encode_sequence(seq)
        
        return torch.tensor(encoded_batch)
    



class ModelWrapper:
    def __init__(self, protein_model_name="facebook/esm2_t6_8M_UR50D", dna_encoding_method="transformer", dna_max_length=None,namefile=None,protein_max_length=None,save_path=None,learning_rate=5e-6, model_creator=None, num_tfs=None, tf_name_to_id=None, aux_weight=0.5):
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
        self.dna_encoding_method = dna_encoding_method
        self.namefile=namefile
        self.save_path=save_path
        self.learning_rate=learning_rate
        self.model_creator = model_creator
        self.num_tfs = num_tfs
        self.tf_name_to_id = tf_name_to_id or {}
        self.aux_weight = aux_weight
        self._has_per_tf_heads = False
    def calculate_peak_max_length(self, dataset):
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
        if "facebook/esm2" in self.protein_model_name:
            from transformers import EsmTokenizer, EsmModel
            self.protein_tokenizer = EsmTokenizer.from_pretrained(self.protein_model_name)
            self.protein_model = EsmModel.from_pretrained(self.protein_model_name).to(device)
        elif self.protein_model_name == 'ESM_DBP':
            import esm
            model_path = "/home/dylan/data/ml/ESM-DBP/model/ESM-DBP.model"
            self.esm_model = esm.ESM2()
            self.esm_alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
            self.esm_batch_converter = self.esm_alphabet.get_batch_converter()
            self.esm_model = torch.nn.DataParallel(self.esm_model)
            self.esm_model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
            self.esm_model.to(device)
            self.esm_model.eval()
            self.protein_tokenizer = None
            self.protein_model = self.esm_model
            if self.protein_max_length is None:
                self.protein_max_length = 1024
        else:
            from transformers import AutoTokenizer, AutoModel
            self.protein_tokenizer = AutoTokenizer.from_pretrained(self.protein_model_name)
            self.protein_model = AutoModel.from_pretrained(self.protein_model_name).to(device)
        
        if self.protein_tokenizer is not None:
            self.protein_max_length = min(1024, self.protein_tokenizer.model_max_length, self.protein_max_length or 1024)
        elif self.protein_max_length is None:
            self.protein_max_length = 1024

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
        if self.model_creator is not None:
            creator = self.model_creator
        else:
            creator = create_model

        self.attached_model = creator(
            self.protein_model_name,
            dna_encoding_method=self.dna_encoding_method,
            dna_max_length=self.dna_max_length,
            num_tfs=self.num_tfs
        ).to(device)

        self._has_per_tf_heads = hasattr(self.attached_model, 'tf_specific_heads')
        self.optimizer = torch.optim.AdamW(self.attached_model.parameters(), lr=self.learning_rate, fused=True)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
        if self._has_per_tf_heads:
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = torch.nn.BCELoss()
    
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
        if self.protein_model_name in ('ESM_DBP', 'ESM_DBP_LORA'):
            data = [(f"seq_{i}", seq) for i, seq in enumerate(protein_sequences)]
            batch_labels, batch_strs, batch_tokens = self.esm_batch_converter(data)
            batch_tokens = batch_tokens.to(device)
            with torch.no_grad():
                results = self.esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
                token_representations = results["representations"][33]
                embeddings = token_representations[:, 1:-1, :]
            B, L, D = embeddings.shape
            if L < self.protein_max_length:
                pad = torch.zeros(B, self.protein_max_length - L, D, device=device)
                embeddings = torch.cat([embeddings, pad], dim=1)
                protein_mask = torch.zeros(B, self.protein_max_length, dtype=torch.bool, device=device)
                protein_mask[:, L:] = True
            elif L > self.protein_max_length:
                embeddings = embeddings[:, :self.protein_max_length, :]
                protein_mask = torch.zeros(B, self.protein_max_length, dtype=torch.bool, device=device)
            else:
                protein_mask = torch.zeros(B, self.protein_max_length, dtype=torch.bool, device=device)
            return embeddings, protein_mask

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
            protein_mask = (attention_mask == 0)
            return embeddings, protein_mask

    def train_batch(self, batch):
        self.attached_model.train()
        tf_name, tf_seqs, peak_seqs, peak_fcs, tf_id, labels = batch
        protein_sequences=tf_seqs
        tf_embeddings, protein_mask = self.get_protein_embeddings(protein_sequences)
        peak_embeddings = self.get_dna_embeddings(peak_seqs)

        if self._has_per_tf_heads:
            tf_id = tf_id.to(device)
            logits, shared_logits = self.attached_model(tf_embeddings, peak_embeddings, tf_id,
                                                        protein_mask=protein_mask, return_all=True) 
            primary_loss = self.loss_fn(logits.squeeze(), labels.squeeze().to(device))
            aux_loss = self.loss_fn(shared_logits.squeeze(), labels.squeeze().to(device))
            loss = primary_loss + self.aux_weight * aux_loss
            pri_val = primary_loss.item()
            aux_val = aux_loss.item()
        else:
            preds = self.attached_model(tf_embeddings, peak_embeddings, protein_mask=protein_mask)
            loss = self.loss_fn(preds.squeeze(), labels.squeeze().to(device))
            pri_val = loss.item()
            aux_val = None

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), labels.mean(), pri_val, aux_val

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
            tf_name, tf_seqs, peak_seqs, peak_fcs, tf_id, labels = batch
            tf_name_list.extend(tf_name)
            tf_seqs_list.extend(tf_seqs)
            peak_seqs_list.extend(peak_seqs)
            peak_fcs_list.extend(peak_fcs)

            tf_embeddings, protein_mask = self.get_protein_embeddings(tf_seqs)
            peak_embeddings = self.get_dna_embeddings(peak_seqs)

            if self._has_per_tf_heads:
                tf_id = tf_id.to(device)
                preds = self.attached_model(tf_embeddings, peak_embeddings, tf_id, protein_mask=protein_mask)
            else:
                preds = self.attached_model(tf_embeddings, peak_embeddings, protein_mask=protein_mask)
            preds_list.append(preds.squeeze().cpu().numpy())
            labels_list.append(labels.squeeze().cpu().numpy())

            loss = self.loss_fn(preds.squeeze(), labels.squeeze().to(device))
            loss_list.append(loss.item())

        all_preds = np.concatenate(preds_list)
        all_labels = np.concatenate(labels_list)
        if self._has_per_tf_heads:
            all_preds = 1.0 / (1.0 + np.exp(-all_preds))
        all_preds_binary = (all_preds >= 0.8).astype(int)
        acc = BinaryAccuracy(threshold=0.8)(torch.tensor(all_preds), torch.tensor(all_labels))
        auc = roc_auc_score(all_labels, all_preds)
        auprc = average_precision_score(all_labels, all_preds)
        mcc = matthews_corrcoef(all_labels, all_preds_binary)
        f1 = f1_score(all_labels, all_preds_binary)

        return (np.mean(np.array(loss_list)), acc.cpu().numpy(), auc, auprc, mcc, f1,
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
            pri_loss_list = []
            aux_loss_list = []
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar = tqdm(train_loader, leave=True, position=0, desc=f'Epoch {epoch}')
            for batch in pbar:
                loss, mean_label, pri_val, aux_val = self.train_batch(batch, epoch)
                loss_list.append(loss)
                label_list.append(mean_label)
                pri_loss_list.append(pri_val)
                if aux_val is not None:
                    aux_loss_list.append(aux_val)

                postfix = {'loss': f'{loss:.3f}', 'lr': f'{current_lr:.2e}'}
                if self._has_per_tf_heads:
                    postfix['pri'] = f'{pri_val:.3f}'
                    postfix['aux'] = f'{aux_val:.3f}'
                pbar.set_postfix(postfix)

            val_loss, val_acc, val_auc, val_auprc, val_mcc, val_f1, val_label,test_preds, test_labels,tf_name_list, test_tf_seqs_list, test_peak_seqs_list, test_peak_fcs_list = self.evaluate(val_loader)

            self.scheduler.step(val_loss)
            current_lr = self.scheduler.get_last_lr()[0]

            train_loss_avg = np.mean(loss_list)
            log_parts = [f'Train Loss: {train_loss_avg:.4f}']
            if self._has_per_tf_heads:
                log_parts.append(f'pri: {np.mean(pri_loss_list):.4f}  aux: {np.mean(aux_loss_list):.4f}')
            log_parts.append(f'mean labels: {np.mean(label_list):.4f}')
            print(f"Epoch {epoch}: Learning Rate = {current_lr}")
            train_log = '  '.join(log_parts)
            print(f'  {train_log}')
            print(f'  Val Loss: {val_loss:.4f}, Val Labels: {val_label:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}, Val AUPRC: {val_auprc:.4f}, Val MCC: {val_mcc:.4f}, Val F1: {val_f1:.4f}')
            # print(f'  Val Loss: {val_loss:.4f}, Val Labels: {val_label:.4f}, Val cor: {val_cor:.4f}, val p-value: {val_pvalue:.4e}')

            history_entry = {
                'epoch': epoch,
                'learning_rate': current_lr,
                'train_loss': np.mean(loss_list),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_auc': val_auc,
                'val_auprc': val_auprc,
                'val_mcc': val_mcc,
                'val_f1': val_f1,
            }
            if self._has_per_tf_heads:
                history_entry['train_primary'] = np.mean(pri_loss_list)
                history_entry['train_aux'] = np.mean(aux_loss_list)
            history.append(history_entry)

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
        final_test_loss, final_test_acc, final_test_auc, final_test_auprc, final_test_mcc, final_test_f1, final_test_label,test_preds, test_labels, tf_name_list, test_tf_seqs_list, test_peak_seqs_list, test_peak_fcs_list = self.evaluate(test_loader)
        print(f'\nFinal Test Performance:')
        print(f'  Loss: {final_test_loss:.4f}, label: {final_test_label:.4f}  Acc: {final_test_acc:.4f}, AUC: {final_test_auc:.4f}, AUPRC: {final_test_auprc:.4f}, MCC: {final_test_mcc:.4f}, F1: {final_test_f1:.4f}')
        test_results.append({
            'loss': final_test_loss,
            'label': final_test_label,
            'acc': final_test_acc,
            'auc': final_test_auc,
            'auprc': final_test_auprc,
            'mcc': final_test_mcc,
            'f1': final_test_f1,
            })
        test_df = pd.DataFrame(test_results)
        test_df.to_csv(os.path.join(self.save_path, f'test_results_{self.namefile}.csv'), index=False)
        


    def test(self, test_dataset):
        self.attached_model.eval()
        test_results = []
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle = False)
        final_test_loss, final_test_acc, final_test_auc, final_test_auprc, final_test_mcc, final_test_f1, final_test_label, test_preds, test_labels,tf_name_list, test_tf_seqs_list, test_peak_seqs_list, test_peak_fcs_list = self.evaluate(test_loader)

        print(f'\nFinal Test Performance:')
        print(f'  Loss: {final_test_loss:.4f}, label: {final_test_label:.4f}  Acc: {final_test_acc:.4f}, AUC: {final_test_auc:.4f}, AUPRC: {final_test_auprc:.4f}, MCC: {final_test_mcc:.4f}, F1: {final_test_f1:.4f}')

        test_results.append({
            'loss': final_test_loss,
            'label': final_test_label,
            'acc': final_test_acc,
            'auc': final_test_auc,
            'auprc': final_test_auprc,
            'mcc': final_test_mcc,
            'f1': final_test_f1,
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

    # @torch.no_grad()
    # def simple_predict(self,data_for_predict):
    #     self.attached_model.eval()
    #     preds_list = []
    #     tf_seqs_list = []
    #     peak_seqs_list = []
    #     tf_name_list = []

    #     for batch in tqdm(data_for_predict, leave=True, position=0):
    #         tf_name, tf_seqs, peak_seqs = batch
    #         tf_name_list.extend(tf_name)
    #         tf_seqs_list.extend(tf_seqs)
    #         peak_seqs_list.extend(peak_seqs)

    #         tf_embeddings, protein_mask = self.get_protein_embeddings(tf_seqs)

    #         peak_embeddings = self.get_dna_embeddings(peak_seqs)

    #         # Predictions
    #         if self._has_per_tf_heads:
    #             tf_id_list = [self.tf_name_to_id.get(name, 0) for name in tf_name]
    #             tf_id = torch.tensor(tf_id_list, dtype=torch.long, device=device)
    #             preds = self.attached_model(tf_embeddings, peak_embeddings, tf_id, protein_mask=protein_mask)
    #         else:
    #             preds = self.attached_model(tf_embeddings, peak_embeddings, protein_mask=protein_mask)
    #         preds_list.append(preds.squeeze().cpu().numpy())

    #     all_preds = np.concatenate(preds_list)
    #     if self._has_per_tf_heads:
    #         all_preds = 1.0 / (1.0 + np.exp(-all_preds))
    #     return all_preds
    