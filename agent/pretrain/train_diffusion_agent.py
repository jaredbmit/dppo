"""
Pre-training diffusion policy

"""

import logging
import wandb
import numpy as np

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.pretrain.train_agent import PreTrainAgent, batch_to_device


class TrainDiffusionAgent(PreTrainAgent):

    def __init__(self, cfg):
        super().__init__(cfg)

    def run(self):

        timer = Timer()
        self.epoch = 1
        cnt_batch = 0
        for _ in range(self.n_epochs):

            # train
            loss_train_epoch = []
            aux_train_epoch = {}
            for batch_train in self.dataloader_train:
                if self.dataset_train.device == "cpu":
                    batch_train = batch_to_device(batch_train, self.device)

                self.model.train()
                x, cond = batch_train.actions, batch_train.conditions
                if self.goal_conditioner is not None:
                    self.goal_conditioner.train()
                    cond = self.goal_conditioner(x, cond)
                loss_train, aux_train = self.model.loss(x, cond)
                loss_train.backward()
                loss_train_epoch.append(loss_train.item())
                for k, v in aux_train.items():
                    aux_train_epoch.setdefault(k, []).append(v.item())

                self.optimizer.step()
                self.optimizer.zero_grad()

                # update ema
                if cnt_batch % self.update_ema_freq == 0:
                    self.step_ema()
                cnt_batch += 1
            loss_train = np.mean(loss_train_epoch)
            aux_train_mean = {k: np.mean(v) for k, v in aux_train_epoch.items()}

            # validate
            loss_val_epoch = []
            if self.dataloader_val is not None and self.epoch % self.val_freq == 0:
                self.model.eval()
                for batch_val in self.dataloader_val:
                    if self.dataset_val.device == "cpu":
                        batch_val = batch_to_device(batch_val, self.device)
                    x_val, cond_val = batch_val.actions, batch_val.conditions
                    if self.goal_conditioner is not None:
                        self.goal_conditioner.eval()
                        cond_val = self.goal_conditioner(x_val, cond_val)
                    loss_val, _ = self.model.loss(x_val, cond_val)
                    loss_val_epoch.append(loss_val.item())
                self.model.train()
            loss_val = np.mean(loss_val_epoch) if len(loss_val_epoch) > 0 else None

            # update lr
            self.lr_scheduler.step()

            # save model
            if self.epoch % self.save_model_freq == 0 or self.epoch == self.n_epochs:
                self.save_model()

            # log loss
            if self.epoch % self.log_freq == 0:
                log.info(
                    f"{self.epoch}: train loss {loss_train:8.4f} | t:{timer():8.4f}"
                )
                if self.use_wandb:
                    if loss_val is not None:
                        wandb.log(
                            {"loss - val": loss_val}, step=self.epoch, commit=False
                        )
                    wandb.log(
                        {
                            "loss - train": loss_train,
                            **{f"loss - {k}": v for k, v in aux_train_mean.items()},
                        },
                        step=self.epoch,
                        commit=True,
                    )

            # count
            self.epoch += 1
