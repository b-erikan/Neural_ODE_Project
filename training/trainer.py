import torch
from pathlib import Path

class Trainer:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.scheduler_mode=self.config["scheduler"]["mode"]
        self.lr = self.config["lr"]
        self.num_epochs=config["num_epochs"]
        self.scheduler_patience = self.config["scheduler"]["patience"]
        
        self.scheduler_factor = self.config["scheduler"]["factor"]
        self.early_stopping_patience = self.config["early_stopping"]["patience"]
        self.early_stopping_min_delta = self.config["early_stopping"]["min_delta"]

        #For early stopping.
        self.epochs_without_improvement = 0

        self.best_val_loss = float("inf")
        self.checkpoint_path = Path(self.config["checkpoint_path"])

        self.loss_function = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=self.scheduler_mode,
            factor=self.scheduler_factor,
            patience=self.scheduler_patience
        )
    def train_step(self, batch, t):
        self.optimizer.zero_grad() #Reset grad for every batch.
        x0 = batch["x0"].to(self.device)
        x_true = batch["x"].to(self.device)
        t = t.to(self.device)
        x_pred = self.model.integrate(x0, t) 
        #ODE solver output is (T, B, D) but dataset target is (B, T, D) so we need to reshape the tensor.
        x_pred = x_pred.permute(1, 0, 2)
        #loss = self.loss_function(x_pred, x_true)

        #UPDATE: Normalized loss for better optimization.

        state_std = self.model.state_std.view(1, 1, -1)
        loss = self.loss_function((x_pred - x_true) / state_std,torch.zeros_like(x_true),)
        loss.backward()

        #Validation loss is oscillating a lot. I'll try gradient clipping here to prevent overcorrection.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=1.0)

        self.optimizer.step() #BP happens here.
        return loss.item()

    def train_epoch(self, train_loader, t):

        self.model.train() #Tell PyTorch that we're in training mode. Won't change much for our simple model but better to stick to the standards.
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            loss = self.train_step(batch, t)
            total_loss += loss
            num_batches += 1

        average_loss = total_loss / num_batches

        return average_loss

    def validate(self, val_loader, t):
        self.model.eval() #Model in evaluation mode. Just for developing habits as before.

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad(): #No grad = We dont need derivatives for this operation since we won't be doing BP here.
            for batch in val_loader:
                x0 = batch["x0"].to(self.device) #Get initial condition
                x_true = batch["x"].to(self.device) #Get ground truth
                t = t.to(self.device)
                x_pred = self.model.integrate(x0, t) #Rollout the network with the initial condition for prediction.
                x_pred = x_pred.permute(1, 0, 2) #Prediction reshaped.
                #loss = self.loss_function(x_pred, x_true) #Compare prediction to ground truth.

                #UPDATE: Normalized loss for better optimization.
                state_std = self.model.state_std.view(1, 1, -1)
                loss = self.loss_function((x_pred - x_true) / state_std,torch.zeros_like(x_true),)

                total_loss += loss.item()
                num_batches += 1

        average_loss = total_loss / num_batches

        return average_loss
    
    def save_checkpoint(self, val_loss):

        if val_loss < self.best_val_loss:

            self.best_val_loss = val_loss
            self.checkpoint_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )
            torch.save(
                self.model.state_dict(),
                self.checkpoint_path
            )

            print(f"New best model saved! Validation Loss: {val_loss:.6f}")
    
    # def early_stopping(self, val_loss):
    #     #TODO: Integrate this to the checkpoint saving.
    #     if val_loss < self.best_val_loss - self.min_delta:

    #         self.epochs_without_improvement = 0
    #         self.best_val_loss = val_loss
    #         return False
    #     else:
    #         self.epochs_without_improvement += 1
    #         if self.epochs_without_improvement >= self.early_stopping_patience:
    #             return True

    #         return False

    def train(self, train_loader, val_loader, train_time, val_time):

        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(
                train_loader=train_loader,
                t=train_time
            )
            val_loss = self.validate(
                val_loader=val_loader,
                t=val_time
            )
            #Update scheduler
            self.scheduler.step(val_loss)

            #current_lr = self.optimizer.param_groups[0]["lr"]

            # Save best model
            self.save_checkpoint(val_loss)
            #TODO: Add a print_interval parameter to the config and print accordingly instead of printing every epoch.
            print(
                f"Epoch [{epoch + 1}/{self.num_epochs}] "
                f"Train Loss: {train_loss:.6f} "
                f"Val Loss: {val_loss:.6f}"
        )