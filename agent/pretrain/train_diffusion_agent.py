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
        step_losses, step_aux = [], {}  # running window for dense (per-step) logging
        if self.load_epoch:
            self.load(self.load_epoch)
            self.epoch = self.load_epoch + 1
            cnt_batch = self.load_epoch * len(self.dataloader_train)
        for _ in range(self.n_epochs - self.load_epoch):

            # train
            loss_train_epoch = []
            aux_train_epoch = {}
            for batch_train in self.dataloader_train:
                if self.dataset_train.device == "cpu":
                    batch_train = batch_to_device(batch_train)

                self.model.train()
                x, cond = batch_train.actions, batch_train.conditions
                if self.goal_conditioner is not None:
                    self.goal_conditioner.train()
                    cond = self.goal_conditioner(x, cond)
                loss_train, aux_train = self.model.loss(x, cond)
                loss_train.backward()
                loss_train_epoch.append(loss_train.item())
                step_losses.append(loss_train.item())
                for k, v in aux_train.items():
                    aux_train_epoch.setdefault(k, []).append(v.item())
                    step_aux.setdefault(k, []).append(v.item())

                self.optimizer.step()
                self.optimizer.zero_grad()

                # update ema
                if cnt_batch % self.update_ema_freq == 0:
                    self.step_ema()
                cnt_batch += 1

                # dense logging: every log_step_freq gradient steps, keyed on the
                # global step (so wandb has one consistent step axis in this mode)
                if self.log_step_freq > 0 and cnt_batch % self.log_step_freq == 0:
                    loss_step = np.mean(step_losses)
                    aux_step = {k: np.mean(v) for k, v in step_aux.items()}
                    log.info(
                        f"step {cnt_batch} (ep {self.epoch}): "
                        f"train loss {loss_step:8.4f} | t:{timer():8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "loss - train": loss_step,
                                **{f"loss - {k}": v for k, v in aux_step.items()},
                                "epoch": self.epoch,
                            },
                            step=cnt_batch,
                            commit=True,
                        )
                    step_losses, step_aux = [], {}
            loss_train = np.mean(loss_train_epoch)
            aux_train_mean = {k: np.mean(v) for k, v in aux_train_epoch.items()}

            # validate
            loss_val_epoch = []
            if self.dataloader_val is not None and self.epoch % self.val_freq == 0:
                self.model.eval()
                for batch_val in self.dataloader_val:
                    if self.dataset_val.device == "cpu":
                        batch_val = batch_to_device(batch_val)
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
                # Per-epoch wandb logging keyed on epoch. When dense per-step
                # logging is active we already log train/aux on the global-step
                # axis, so here we only add the (epoch-granular) val loss, keyed
                # on the current global step to keep one monotonic wandb axis.
                if self.use_wandb and self.log_step_freq == 0:
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
                elif self.use_wandb and loss_val is not None:
                    wandb.log({"loss - val": loss_val}, step=cnt_batch, commit=True)

            # count
            self.epoch += 1
