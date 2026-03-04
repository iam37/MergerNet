import wandb

from ignite.engine import (
    Events,
    create_supervised_trainer,
    create_supervised_evaluator,
)
from ignite.metrics import Loss, Accuracy, Precision, ConfusionMatrix, Recall, Fbeta
from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.contrib.handlers import CosineAnnealingScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from ignite.handlers import EarlyStopping

def create_trainer(model, optimizer, criterion, loaders, device, use_scheduler=True):
    """Set up Ignite trainer and evaluator."""
    trainer = create_supervised_trainer(
        model, optimizer, criterion, device=device
    )

    if use_scheduler:
        #torch_lr_scheduler = CosineAnnealingLR(optimizer, T_max=40)
        #cheduler = ReduceLROnPlateau(optimizer, mode="min", patience = 8, threshold=1e-4, min_lr=1e-7)
        scheduler = CosineAnnealingLR(optimizer, T_max = 15)
        #scheduler = LRScheduler(torch_lr_scheduler)
        #warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=5)
        #cosine_scheduler = CosineAnnealingLR(optimizer, T_max=25)

        #torch_lr_scheduler = SequentialLR(
        #    optimizer, 
        #    schedulers=[warmup_scheduler, cosine_scheduler],
        #    milestones=[5])
        #scheduler = LRScheduler(torch_lr_scheduler)

    metrics = {
        "accuracy": Accuracy(),
        "precision": Precision(average="weighted"),
        "recall": Recall(average="weighted"),
        "loss": Loss(criterion),
        "cm": ConfusionMatrix(num_classes=wandb.config["num_classes"], output_transform=lambda x: x),
        "f1": Fbeta(beta=1)
    }

    evaluator = create_supervised_evaluator(
        model, metrics=metrics, device=device
    )

    # Function to log metrics to wandb
    def log_metrics(trainer, loader, log_prefix=""):
        logging.info(f"Logging metrics for {log_prefix}")

        # Reset evaluator state before running evaluation
        evaluator.state.metrics = {}

        evaluator.run(loader)
        metrics = evaluator.state.metrics
        log_dict = {f"{log_prefix}{k}": v for k, v in metrics.items() if k != "cm"}

        # Handle the confusion matrix separately
        cm = metrics["cm"].cpu().numpy()
        class_names = [str(i) for i in range(wandb.config["num_classes"])]

        # Calculate true and predicted labels from the confusion matrix
        y_true, y_pred = [], []
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                y_true.extend([i] * int(cm[i, j]))
                y_pred.extend([j] * int(cm[i, j]))

        cm_plot = wandb.plot.confusion_matrix(probs=None,
                                              y_true=y_true,
                                              preds=y_pred,
                                              class_names=class_names)

        # Log other metrics and the confusion matrix plot
        log_dict[f"{log_prefix}confusion_matrix"] = cm_plot
        wandb.log(log_dict)

    def get_current_lr(optimizer):
        return optimizer.param_groups[0]['lr']

    # Define training hooks
    #if use_scheduler:
    #    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    @trainer.on(Events.STARTED)
    def log_results_start(trainer):
        logging.info("Log results started.")
        for L, loader in loaders.items():
            log_metrics(trainer, loader, log_prefix=f"{L}_")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_devel_results(trainer):
        for L, loader in loaders.items():
            log_metrics(trainer, loader, log_prefix=f"{L}_")
        wandb.log({"lr": get_current_lr(optimizer)})
        
    
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def run_validation(engine):
        evaluator.run(loaders['devel'])
        
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_and_schedule(trainer):
        # Run validation
        evaluator.run(loaders['devel'])
        
        # Get validation loss
        val_loss = evaluator.state.metrics['loss']
        
        
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Log all metrics
        for L, loader in loaders.items():
            log_metrics(trainer, loader, log_prefix=f"{L}_")
        
        # Log LR and epoch info
        current_lr = get_current_lr(optimizer)
        wandb.log({
            "lr": current_lr,
            "epoch": trainer.state.epoch,
            "devel_loss": val_loss
        })
        logging.info(f"Epoch {trainer.state.epoch}: Val Loss = {val_loss:.4f}, LR = {current_lr:.6e}")
    #if early_stopping_bool:
    #    early_stopping = EarlyStopping(
    #        patience=early_stopping_parameter,
    #        score_function=lambda engine: -engine.state.metrics['loss'],
    #        trainer=trainer
    #    )
    #    evaluator.add_event_handler(Events.COMPLETED, early_stopping)
    
    # Log when early stopping triggers
    #@trainer.on(Events.TERMINATE)
    #def log_early_stop(engine):
    #    logging.info(f"Early stopping triggered at epoch {engine.state.epoch}")

    @trainer.on(Events.ITERATION_COMPLETED)
    def clip_gradients(engine):
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)

    @trainer.on(Events.COMPLETED)
    def log_results_end(trainer):
        for L, loader in loaders.items():
            log_metrics(trainer, loader, log_prefix=f"{L}_")

        logging.info("Terminating run explicitly.")
        trainer.terminate()

    return trainer, evaluator


def create_transfer_learner(model, optimizer, criterion, loaders, device):
    """Method to create a transfer learner trainer."""

    # Initialize a stack that contains all frozen layers.
    frozen_layer_stack = []

    # Initial freezing of the layers.
    logging.info("Freezing non-FC layers for given model...")
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False
            frozen_layer_stack.append((name, param))

    # Create trainer
    trainer = create_trainer(
        model, optimizer, criterion, loaders, device
    )

    # Gradual unfreezing of layers based on epoch.
    @trainer.on(Events.EPOCH_COMPLETED)
    def unfreeze_layers(engine):
        epoch = engine.state.epoch

        # We unfreeze one entire layer at a time (O(1) complexity).
        wandb.log({"frozen_layers": len(frozen_layer_stack)})
        if frozen_layer_stack:
            top_name, top_param = frozen_layer_stack[-1]
            layer_name = top_name.split('.')[1]
            while frozen_layer_stack and frozen_layer_stack[-1][0].split('.')[1] == layer_name:
                name, param = frozen_layer_stack.pop()
                param.requires_grad = True
                logging.info(f"Epoch[{epoch}]: layer {name} is now trainable.")
        else:
            # All layers unfrozen already!
            logging.info(f"Epoch[{epoch}]: all layers trainable.")

    return trainer
