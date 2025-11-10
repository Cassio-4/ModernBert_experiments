from abc import ABC, abstractmethod

class BaseEngine(ABC):
    """Lean engine that only handles training/evaluation, checkpointing, and logging."""

    def __init__(self, config, model, train_loader, val_loader, test_loader, logger):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.logger = logger
        self.device = config.get("device", "cpu")
    
    def train(self):
        """Train the model for config.epochs number of epochs with early stopping."""
        EPOCHS = self.config['num_epochs']
        PATIENCE = self.config['patience']
        print("Starting training...")
        train_loss, val_loss, lr_lst = [], [], []
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, EPOCHS + 1):
            # Train epoch
            epoch_avg_loss = self._forward_one_epoch(train=True)
            self.epochs_run = epoch
            train_loss.append(epoch_avg_loss)
            # Validation epoch
            val_avg_loss = self._forward_one_epoch(train=False)
            val_loss.append(val_avg_loss)
            lr_lst.append(self.optimizer.param_groups[0]['lr'])
            self.logger.log_epoch(epoch, train_loss[-1], val_loss[-1], lr_lst[-1])
            if self.scheduler is not None:
                self.scheduler.step()
            if val_avg_loss < best_val_loss:
                best_val_loss = val_avg_loss
                patience_counter = 0
                self._save_checkpoint()
                print(f"Saved best model with val_loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping at epoch {epoch}, no improvement in {PATIENCE} epochs.")
                    break
    
    @abstractmethod
    def _forward_one_epoch(self, train: bool = True):
        """Forward one epoch of training or validation.

        Args:
            train (bool, optional): Whether to train or validate. Defaults to True.
        Returns:
            mean_loss (float): Mean loss over the epoch.
        """
        pass

    @abstractmethod
    def _save_checkpoint(self):
        """Save the current model checkpoint."""
        pass
