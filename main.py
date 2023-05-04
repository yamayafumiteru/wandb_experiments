import wandb
import random


class WandBTrainer:
    def __init__(self):
        self.wandb_project = "my-awesome-project"
        self.learning_rate = 0.02
        self.architecture = "CNN"
        self.dataset = "CIFAR-100"
        self.epochs = 10

        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project=self.wandb_project,
            # track hyperparameters and run metadata
            config={
                "learning_rate": self.learning_rate,
                "architecture": self.architecture,
                "dataset": self.dataset,
                "epochs": self.epochs,
            },
        )

    def train(self):
        # simulate training
        epochs = self.epochs
        offset = random.random() / 5
        for epoch in range(2, epochs):
            acc = 1 - 2**-epoch - random.random() / epoch - offset
            loss = 2**-epoch + random.random() / epoch + offset

            # log metrics to wandb
            wandb.log({"acc": acc, "loss": loss})

        accuracy = 0.9
        # 通知の送信
        wandb.alert(title="WandBからの通知", text=f"今の正解率は {accuracy} です。")

        wandb.finish()
