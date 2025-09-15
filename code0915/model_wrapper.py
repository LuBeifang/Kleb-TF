from enum import unique

import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import glob
from tqdm.auto import tqdm
from torchmetrics.classification import BinaryAccuracy
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score



from nn_data_prep import TFDataset
from models import AttachedModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = 1
np.random.seed(seed)
torch.manual_seed(seed)

class ModelWrapper:
    def __init__(self):
        self.tf_tokenizer = None
        self.tf_feature_extractor = None
        self.tf_max_length = None
        self.dna_tokenizer = None
        self.dna_feature_extractor = None
        self.dna_max_length = None
        self.attached_model = None
        self.optimizer = None
        self.loss_fn = None
        self.scheduler = None

    def get_model_settings(self):
        self.tf_tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-50m-3mer-multi-species", trust_remote_code=True)
        self.tf_feature_extractor = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-50m-3mer-multi-species", trust_remote_code=True).to(device)
        self.tf_max_length = self.tf_tokenizer.model_max_length

        self.dna_tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-50m-multi-species", trust_remote_code=True)
        self.dna_feature_extractor = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-50m-multi-species", trust_remote_code=True).to(device)
        self.dna_max_length = self.dna_tokenizer.model_max_length

        self.attached_model = AttachedModel().to(device)
        self.optimizer = torch.optim.AdamW(self.attached_model.parameters(), lr=1e-3, weight_decay=1e-5, fused=True)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

        # self.loss_fn = torch.nn.BCELoss(reduction='none')
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')


    def compute_mean_embeddings_per_sequence(self, embeddings, attention_mask):
        # Add embed dimension axis
        attention_mask = torch.unsqueeze(attention_mask, dim=-1)

        # Compute mean embeddings per sequence
        mean_sequence_embeddings = torch.sum(attention_mask * embeddings, axis=-2) / torch.sum(attention_mask, axis=1)
        return mean_sequence_embeddings



    def train_batch(self, batch):
        self.attached_model.train()
        tf_seqs, peak_seqs, peak_fcs, labels = batch
        tf_token_ids = self.tf_tokenizer.batch_encode_plus(tf_seqs, return_tensors="pt", padding="max_length", max_length=self.tf_max_length)[
            "input_ids"].to(device)
        peak_token_ids = self.dna_tokenizer.batch_encode_plus(peak_seqs, return_tensors="pt", padding="max_length", max_length=self.dna_max_length)[
            "input_ids"].to(device)


        with torch.no_grad():
            tf_attention_mask = tf_token_ids != self.tf_tokenizer.pad_token_id
            tf_llm_outputs = self.tf_feature_extractor(
                tf_token_ids,
                attention_mask=tf_attention_mask,
                encoder_attention_mask=tf_attention_mask,
                output_hidden_states=True
            )
            tf_embeddings = tf_llm_outputs['hidden_states'][-1].detach()
            tf_embeddings = self.compute_mean_embeddings_per_sequence(tf_embeddings, tf_attention_mask)
            peak_attention_mask = peak_token_ids != self.dna_tokenizer.pad_token_id
            peak_llm_outputs = self.dna_feature_extractor(
                peak_token_ids,
                attention_mask=peak_attention_mask,
                encoder_attention_mask=peak_attention_mask,
                output_hidden_states=True
            )
            peak_embeddings = peak_llm_outputs['hidden_states'][-1].detach()
            peak_embeddings = self.compute_mean_embeddings_per_sequence(peak_embeddings, peak_attention_mask)


        # print(tf_embeddings.size(), peak_embeddings.size())
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
        self.tf_feature_extractor.eval()
        self.dna_feature_extractor.eval()
        loss_list = []
        preds_list = []
        labels_list = []
        tf_seqs_list = []
        peak_seqs_list = []
        peak_fcs_list = []


        for batch in tqdm(val_loader, leave=True, position=0):
            tf_seqs, peak_seqs, peak_fcs, labels = batch
            tf_seqs_list.extend(tf_seqs)
            peak_seqs_list.extend(peak_seqs)
            peak_fcs_list.extend(peak_fcs)

            tf_token_ids = self.tf_tokenizer.batch_encode_plus(tf_seqs, return_tensors="pt", padding="max_length", max_length=self.tf_max_length)[
                "input_ids"].to(device)
            peak_token_ids = self.dna_tokenizer.batch_encode_plus(peak_seqs, return_tensors="pt", padding="max_length", max_length=self.dna_max_length)[
                "input_ids"].to(device)
            with torch.no_grad():
                tf_attention_mask = tf_token_ids != self.tf_tokenizer.pad_token_id
                tf_llm_outputs = self.tf_feature_extractor(
                    tf_token_ids,
                    attention_mask=tf_attention_mask,
                    encoder_attention_mask=tf_attention_mask,
                    output_hidden_states=True
                )
                tf_embeddings = tf_llm_outputs['hidden_states'][-1].detach()
                tf_embeddings = self.compute_mean_embeddings_per_sequence(tf_embeddings, tf_attention_mask)
                peak_attention_mask = peak_token_ids != self.dna_tokenizer.pad_token_id
                peak_llm_outputs = self.dna_feature_extractor(
                    peak_token_ids,
                    attention_mask=peak_attention_mask,
                    encoder_attention_mask=peak_attention_mask,
                    output_hidden_states=True
                )
                peak_embeddings = peak_llm_outputs['hidden_states'][-1].detach()
                peak_embeddings = self.compute_mean_embeddings_per_sequence(peak_embeddings, peak_attention_mask)


        # print(tf_embeddings.size(), peak_embeddings.size())

            preds = self.attached_model(tf_embeddings, peak_embeddings)


            preds_list.append(preds.squeeze().cpu().numpy())
            labels_list.append(labels.squeeze().cpu().numpy())


            loss = self.loss_fn(preds.squeeze(), labels.squeeze().to(device))
            loss = loss.mean()
            loss_list.append(loss.item())

        all_preds = np.concatenate(preds_list)
        all_labels = np.concatenate(labels_list)
        acc = BinaryAccuracy(threshold=0.7)(torch.tensor(all_preds), torch.tensor(all_labels))
        auc = roc_auc_score(all_labels, all_preds)
        auprc = average_precision_score(all_labels, all_preds)
        return np.mean(np.array(loss_list)), acc.cpu().numpy(), auc, auprc, all_labels.mean().astype(np.float64), all_preds, all_labels, tf_seqs_list, peak_seqs_list, peak_fcs_list


    def train(self, train_dataset, val_dataset, epochs=50):

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=16, shuffle=True,
        )

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True)

        best_auc = 0
        history = []

        for epoch in list(range(1, epochs+1)):
            loss_list = []
            label_list = []
            for batch in tqdm(train_loader, leave=True, position=0, desc=f'Epoch {epoch}'):
                loss, mean_label = self.train_batch(batch)
                loss_list.append(loss)
                label_list.append(mean_label)


            val_loss, val_acc, val_auc, val_auprc, val_label, val_preds, val_labels, val_tf_seqs_list, val_peak_seqs_list, val_peak_fcs_list = self.evaluate(val_loader)

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
            })


            if val_auc > best_auc:
                best_auc = val_auc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.attached_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_auc': val_auc,
                }, f'/media/lu/lussd/kp_tf/trained_models/best_model_{train_date}.pt')
                print(f'  The best model saved (AUC: {val_auc:.4f})')



        df_history = pd.DataFrame(history)
        df_history.to_csv(f'/media/lu/lussd/kp_tf/record/training_history_{train_date}.csv', index=False)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, weights_only=False)
        self.attached_model.load_state_dict(checkpoint['model_state_dict'])
        self.attached_model.eval()

    def test(self, test_dataset):
        self.attached_model.eval()
        test_results = []
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle = False)
        final_test_loss, final_test_acc, final_test_auc, final_test_auprc, final_test_label, test_preds, test_labels, test_tf_seqs_list, test_peak_seqs_list, test_peak_fcs_list = self.evaluate(test_loader)

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
        test_df.to_csv(f'/media/lu/lussd/kp_tf/record/test_results_{train_date}.csv', index=False)

        test_matrix = pd.DataFrame({
            'TF_seqs': test_tf_seqs_list,
            'peak_seqs': test_peak_seqs_list,
            'peak_fcs': test_peak_fcs_list,
            'Labels': test_labels,
            'Predictions': test_preds,
        })
        test_matrix.to_csv(f'/media/lu/lussd/kp_tf/record/test_matrix_{train_date}.csv', index=False)




if __name__ == '__main__':
    # # 27TFs
    # tf_fasta_paths = glob.glob('/media/lu/lussd/kp_tf/data/input/TF_fasta/*.fa')
    # tf_fasta_paths = [p for p in tf_fasta_paths if ':' not in p]
    # peak_fasta_paths = glob.glob('/media/lu/lussd/kp_tf/data/input/peak_fasta_201bp_p3n30/*.fasta')
    # dataset = TFDataset(tf_fasta_paths, peak_fasta_paths, '/media/lu/lussd/kp_tf/data/input/allpeak_p3n30.csv',
    #                     '/media/lu/lussd/kp_tf/data/input/allpeak_27TF.csv', '/media/lu/lussd/kp_tf/data/hvkp4.fasta')

    # 96 TFs
    tf_fasta_paths = glob.glob('/media/lu/lussd/kp_tf/data/Trail3/TF_fasta/*.fa')
    tf_fasta_paths = [p for p in tf_fasta_paths if ':' not in p]
    peak_fasta_paths = glob.glob('/media/lu/lussd/kp_tf/data/Trail3/peak_fasta/*.fasta')
    dataset = TFDataset(tf_fasta_paths, peak_fasta_paths, '/media/lu/lussd/kp_tf/data/Trail3/allpeak_96TF.csv',
                        '/media/lu/lussd/kp_tf/data/Trail3/allpeak_96TF_forneg.csv',
                        '/media/lu/lussd/kp_tf/data/hvkp4.fasta')


    train_date = "96TF_test0824"

    batch_size = 16
    total_size = len(dataset.peak_with_seqs_df) + len(dataset.neg_peak_with_seqs_df)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)

    train_size = train_size - (train_size % batch_size)
    val_size = val_size - (val_size % batch_size)
    test_size = total_size - train_size - val_size
    test_size = test_size - (test_size % batch_size)

    val_start = train_size
    test_start = train_size + val_size

    indices = np.random.permutation(total_size)

    train_indices = indices[:train_size]
    val_indices = indices[val_start:val_start + val_size]
    test_indices = indices[test_start:test_start + test_size]
    indices = np.random.permutation(len(dataset))
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    torch.save(val_indices, f'/media/lu/lussd/kp_tf/record/val_indices_{train_date}.pt')
    torch.save(test_indices, f'/media/lu/lussd/kp_tf/record/test_indices_{train_date}.pt')


    model_wrapper = ModelWrapper()
    model_wrapper.get_model_settings()

    # Train model
    model_wrapper.train(train_dataset, val_dataset)

    # Test model
    model_wrapper.load_model(f'/media/lu/lussd/kp_tf/trained_models/best_model_{train_date}.pt')
    model_wrapper.test(test_dataset)
