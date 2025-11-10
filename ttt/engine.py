import torch
import os 
from transformers import DataCollatorForWholeWordMask
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.core.base_engine import BaseEngine

class TTTEngine(BaseEngine):
    def __init__(self, net, ssh: BertSelfSupervisedHead, optimizer, scheduler, criterion, tokenizer,
                 train_loader, val_loader, test_loader: DataLoader, config,
                 id2label=None, device=None, results_dir=None, labels_list=None):
        self.net = net # rede BERT com cabeça de NER
        self.ssh = ssh # self-supervised head (ext + head)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.tokenizer = tokenizer
        self.device = device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.labels_list = labels_list
        self.config = config
        self.id2label = id2label
        self.results_dir = results_dir
        self.loaded_local_checkpoint = False
        self.epochs_run = 0
        self.collator = DataCollatorForWholeWordMask(tokenizer=self.tokenizer)
    
    def forward_one_epoch(self, train=True):
        if train:
            self.net.train()
            self.ssh.train()
            data_loader = self.train_loader
        else:
            self.net.eval()
            self.ssh.eval()
            data_loader = self.val_loader
        total_loss = 0.0
        with torch.set_grad_enabled(train):
            for batch_idx, (data, instance_ids) in enumerate(data_loader):
                    if train:
                        self.optimizer.zero_grad()
                    input_ids = data['input_ids'].to(self.device)
                    attention_mask = data['attention_mask'].to(self.device)
                    labels = data['labels'].to(self.device)
                    output_net = self.net(input_ids=input_ids, attention_mask=attention_mask)
                    logits = output_net.logits.view(-1, output_net.logits.size(-1))
                    labels_flat = labels.view(-1)
                    loss = self.criterion(logits, labels_flat)

                    if self.config.get('shared', None) is not None:
                        wlm_data = self.collator(data['input_ids'])
                        inputs_ssh, labels_ssh = wlm_data['input_ids'], wlm_data['labels']
                        inputs_ssh, labels_ssh = inputs_ssh.cuda(), labels_ssh.cuda()
                        outputs_ssh = self.ssh(inputs_ssh, attention_mask)
                        mlm_logits = outputs_ssh.view(-1, outputs_ssh.size(-1))
                        loss_ssh = self.criterion(mlm_logits, labels_ssh.view(-1))
                        loss += loss_ssh

                    if train:
                        loss.backward()
                        self.optimizer.step()
                    total_loss += loss.item()
        mean_loss = total_loss / batch_idx if batch_idx > 0 else 0.0
        return mean_loss

    def train(self):
        EPOCHS = self.config['num_epochs']
        PATIENCE = self.config['patience']
        print("Starting training...")
        train_loss, val_loss, lr_lst = [], [], []
        best_val_loss = float('inf')
        patience_counter = 0
        model_sv_path = os.path.join(self.results_dir, "best_model.pth")
        losses_sv_path = os.path.join(self.results_dir, "losses.csv")
        for epoch in range(1, EPOCHS + 1):
            # Train epoch
            epoch_avg_loss = self.forward_one_epoch(train=True)
            self.epochs_run = epoch
            train_loss.append(epoch_avg_loss)
            # Validation epoch
            val_avg_loss = self.forward_one_epoch(train=False)
            val_loss.append(val_avg_loss)
            lr_lst.append(self.optimizer.param_groups[0]['lr'])
            if self.scheduler is not None:
                self.scheduler.step()
            print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}".format(epoch, train_loss[-1], val_loss[-1], lr_lst[-1]))
            if val_avg_loss < best_val_loss:
                best_val_loss = val_avg_loss
                patience_counter = 0
                self.save_checkpoint(model_sv_path)
                print(f"Saved best model with val_loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping at epoch {epoch}, no improvement in {PATIENCE} epochs.")
                    break
        
        losses_df = pd.DataFrame({
            "epoch": list(range(1, len(train_loss) + 1)),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr_lst
        })
        losses_df.to_csv(losses_sv_path, index=False)

    def test_std(self):
        if not self.loaded_local_checkpoint:
            print("testing for NER without loading checkpoint")
        self.net.eval()
        true_sequences = []
        pred_sequences = []
        with torch.no_grad():
            for batch_idx, (data, instance_ids) in enumerate(self.test_loader):
                input_ids = data['input_ids'].to(self.device)
                attention_mask = data['attention_mask'].to(self.device)
                labels = data['labels']
                output_net = self.net(input_ids=input_ids, attention_mask=attention_mask)
                logits = output_net.logits
                predicted = logits.argmax(dim=-1).cpu()

                for i in range(predicted.shape[0]):
                    mask = labels[i] != -100
                    seq_pred = predicted[i][mask].cpu().tolist()
                    seq_true = labels[i][mask].cpu().tolist()
                    pred_labels = [self.id2label[label_id] for label_id in seq_pred]
                    true_labels = [self.id2label[label_id] for label_id in seq_true]
                    pred_sequences.append(pred_labels)
                    true_sequences.append(true_labels)
        self.compute_scores(pred_sequences, true_sequences)
    
    def test_ttt_std(self):
        if not self.loaded_local_checkpoint:
            print("test_ttt_std() without loading checkpoint")
        true_sequences = []
        pred_sequences = []
        # Step 1: Uma sample por vez
        ttt_loader = DataLoader(dataset=self.test_loader.dataset, batch_size=1, collate_fn=self.test_loader.collate_fn)
        # Step 2: Initialize with pre-trained weights (fresh start for each sample)
        current_theta_e = self.ssh.state_dict().copy()
        print("Starting TTT-STD testing...")
        for batch_idx, (data, instance_ids) in enumerate(tqdm(ttt_loader, desc="TTT-STD Testing")):
            # Step 3: Test-time training on self-supervised task
            optimizer = optim.SGD(self.ssh.parameters(), lr=0.001)  # Only update feature extractor
            input_ids = data['input_ids'].to(self.device)
            attention_mask = data['attention_mask'].to(self.device)
            labels = data['labels']
            self.ssh.train()
            for step in range(5):  # 10 gradient steps
                optimizer.zero_grad()
                wlm_data = self.collator(data['input_ids'])
                inputs_ssh, labels_ssh = wlm_data['input_ids'].cuda(), wlm_data['labels'].cuda()
                # Forward pass through self-supervised branch
                outputs_ssh = self.ssh(inputs_ssh, attention_mask) # Using current_theta_e
                mlm_logits = outputs_ssh.view(-1, outputs_ssh.size(-1))
                loss_ssh = self.criterion(mlm_logits, labels_ssh.view(-1))
                # Backward pass - only update feature extractor
                loss_ssh.backward()
                optimizer.step()
            # Step 4: Make prediction with updated feature extractor
            with torch.no_grad():
                output_net = self.net(input_ids=input_ids, attention_mask=attention_mask)
                logits = output_net.logits
                predicted = logits.argmax(dim=-1).cpu()
                for i in range(predicted.shape[0]):
                    mask = labels[i] != -100
                    seq_pred = predicted[i][mask].cpu().tolist()
                    seq_true = labels[i][mask].cpu().tolist()
                    pred_labels = [self.id2label[label_id] for label_id in seq_pred]
                    true_labels = [self.id2label[label_id] for label_id in seq_true]
                    pred_sequences.append(pred_labels)
                    true_sequences.append(true_labels)
            self.ssh.load_state_dict(current_theta_e)
        self.compute_scores(pred_sequences, true_sequences, "ttt_std")

    def compute_scores(self, pred_sequences, true_sequences, test_type=None):
        scores = self.compute_entity_level_scores(pred_sequences, true_sequences)
        summary_path = os.path.join(self.results_dir, ("summary.txt" if test_type is None else f"summary_{test_type}.txt"))
        with open(summary_path, "w") as f:
            f.write("CONFIGURATION:\n")
            f.write(json.dumps(self.config, indent=2))
            f.write(f"\nTEST METRICS:\n")
            for metric_name, metric_value in scores.items():
                f.write(f"{metric_name}: {metric_value}\n")

    def load_checkpoint(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.results_dir, "best_model.pth")
        state = torch.load(checkpoint_path, map_location=self.device)
        self.net.load_state_dict(state['net'])
        self.ssh.head.load_state_dict(state['head'])
        self.loaded_local_checkpoint = True
        print(f"Loaded best model from {checkpoint_path}")
    
    def save_checkpoint(self, checkpoint_path=None):
        state = {'net': self.net.state_dict(), 'head': self.ssh.head.state_dict(),
            'optimizer': self.optimizer.state_dict()}
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.results_dir, "state.pth")
        torch.save(state, checkpoint_path)
        
