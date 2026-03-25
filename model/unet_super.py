import abc
from argparse import ArgumentParser
import pytorch_lightning as pl
import torch
import numpy as np
import os

from losses.FocalLosses import FocalLoss, Cyclical_FocalLoss, DiceLoss


class UnetSuper(pl.LightningModule):
    """UnetSuper is a basic implementation of the LightningModule without any ANN modules
    It is a parent class which should not be used directly
    """
    def __init__(self, hparams, **kwargs):
        super(UnetSuper, self).__init__()

        self.num_classes = kwargs["num_classes"]
        self.metric = iou_fnc
        self.save_hyperparameters(hparams)
        self.args = kwargs

        if kwargs["flat_weights"]:
            self.weights = [1, 1, 1, 1, 1, 1, 1]
        else:
            self.weights = [0.5, 1, 1, 1, 1, 1, 1]

        #if kwargs["loss"] == "FocalLoss":
        #    self.criterion = FocalLoss(apply_nonlin=None, alpha=self.weights, gamma=2.0)
        #else:
        #    self.criterion = Cyclical_FocalLoss()
        self.criterion = FocalLoss(apply_nonlin=None, alpha=self.weights, gamma=2.0)

        self.dice_criterion = DiceLoss()
        self.dice_criterion.cuda()

        self.criterion.cuda()
        self._to_console = False
        self._val_outputs = []
        #self._train_outputs = []
        self._test_metrics_per_image = []
        self._test_metrics_per_image.append(["id", "iou_class_0", "iou_class_1", "iou_class_2", "iou_class_3", "iou_class_4",
                                  "iou_class_5", "iou_class_6", "dice_class_0", "dice_class_1", "dice_class_2", "dice_class_3",
                                  "dice_class_4", "dice_class_5", "dice_class_6", "mean_iou", "mean_dice", "foreground_iou", "foreground_dice"])


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 16)')
        parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
        parser.add_argument('--gamma-factor', type=float, default=2.0, help='gamma factor (default: 2.0)')
        parser.add_argument('--weight-decay', type=float, default=1e-5, help='weight decay (default: 0.0002)')
        parser.add_argument('--epsilon', type=float, default=1e-16, help='epsilon (default: 1e-16)')
        parser.add_argument('--models', type=str, default="Unet", help='the wanted model')
        parser.add_argument('--training-batch-size', type=int, default=1, help='Input batch size for training')
        parser.add_argument('--test-batch-size', type=int, default=1, help='Input batch size for testing')
        parser.add_argument('--dropout-val', type=float, default=0, help='dropout_value for layers')
        parser.add_argument('--flat-weights', type=bool, default=False, help='set all weights to 0.01')
        parser.add_argument('--loss', type=str, default="FocalLoss")
        return parser


    @abc.abstractmethod
    def forward(self, x):
        """
        Implemented in the child class, defines the forward pass of the model
        """
        pass


    def loss(self, logits, labels):
        """
        Initializes the loss function

        :return: output - Initialized cross entropy loss function
        """
        labels = labels.long()
        #focal_loss = self.criterion(logits, labels)
        #dice_loss = self.dice_criterion(logits, labels)

        return self.criterion(logits, labels)


    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        prob_mask = self.forward(x)

        loss = self.criterion(prob_mask, y.type(torch.long), self.current_epoch)

        # log loss (Lightning will average per epoch)
        self.log("train_avg_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # log IoU (per batch → averaged automatically)
        iter_iou, iter_count = iou_fnc(torch.argmax(prob_mask, dim=1).float(), y, self.args['num_classes'])

        for i in range(self.args['num_classes']):
            self.log(f"train_iou_{i}",
                torch.tensor(iter_iou[i], device=self.device),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        return loss


    def validation_step(self, test_batch, batch_idx):
        """
        Predicts on the test dataset to compute the current performance of the models.
        :param test_batch: Batch data
        :param batch_idx: Batch indices
        :return: output - Validation performance
        """

        output = {}
        x, y = test_batch
        prob_mask = self.forward(x)

        loss = self.criterion(prob_mask, y.type(torch.long), self.current_epoch)
        self.log("val_avg_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        preds = torch.argmax(prob_mask, dim=1).float()
        iter_iou, iter_count = iou_fnc(preds, y, self.args['num_classes'])
        iter_iou_tensor = torch.tensor(iter_iou, device=self.device)

        for c in range(self.args['num_classes']):
            self.log(f"val_iou_{c}", iter_iou_tensor[c], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        for i in range(self.args['num_classes']):
            output['val_iou_' + str(i)] = torch.tensor(iter_iou[i])
            output['val_iou_cnt_' + str(i)] = torch.tensor(iter_count[i])

        output['val_loss'] = loss

        self._val_outputs.append(output)

        #mean_iou = iter_iou_tensor.mean()
        #self.log("val_mean_iou", mean_iou, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return output


    def on_validation_epoch_end(self):
        outputs = self._val_outputs
        val_avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean().item()

        val_iou_sum = torch.zeros(self.args['num_classes'])
        val_iou_cnt_sum = torch.zeros(self.args['num_classes'])

        for i in range(self.args['num_classes']):
            val_iou_sum[i] = torch.stack([x['val_iou_' + str(i)] for x in outputs]).sum()
            val_iou_cnt_sum[i] = torch.stack([x['val_iou_cnt_' + str(i)] for x in outputs]).sum()

        iou_scores = val_iou_sum / (val_iou_cnt_sum + 1e-10)
        iou_mean = iou_scores[~torch.isnan(iou_scores)].mean().item()

        self.log('val_avg_loss', val_avg_loss, sync_dist=True, on_step=False, on_epoch=True)
        self.log('val_mean_iou', iou_mean, sync_dist=True, on_step=False, on_epoch=True)

        for c in range(self.args['num_classes']):
            if val_iou_cnt_sum[c] == 0.0:
                iou_scores[c] = 0
            self.log(f'val_iou_{c}', iou_scores[c].item(), sync_dist=True, on_step=False, on_epoch=True)

        if self._to_console:
            print(f'Validation Epoch {self.current_epoch} ------------------------')
            print(f'Loss: {val_avg_loss:.6f}, Mean IoU: {iou_mean:.6f}')
            for c in range(self.args['num_classes']):
                print(f'class {c} IoU: {iou_scores[c].item():.6f}')

        self._val_outputs.clear()


    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        prob_mask = self.forward(x)

        loss = self.criterion(prob_mask, y.type(torch.long), self.current_epoch)

        # log test loss
        self.log("test_avg_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        preds = torch.argmax(prob_mask, dim=1)
        iter_iou, iter_count = iou_fnc(preds, y, self.args['num_classes'])

        for i in range(self.args['num_classes']):
            self.log(
                f"test_iou_{i}",
                torch.tensor(iter_iou[i], device=self.device),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        for i in range(x.shape[0]):
            pred_img = preds[i]  # [H, W]
            true_img = y[i]      # [H, W]

            # compute perclass IoU
            iou_per_class, _ = iou_fnc(pred_img, true_img, n_classes=self.args['num_classes'])


            # compute perclass Dice
            dice_per_class = dice_fnc(pred_img, true_img, n_classes=self.args['num_classes'])

            # log per-image, perclass metrics, and mean metrics
            row = [f"{batch_idx}_{i}"]
            row.extend(iou_per_class.tolist())
            row.extend(dice_per_class.tolist())
            row.append(np.mean(iou_per_class))
            row.append(np.mean(dice_per_class))
            row.append(foreground_iou(pred_img, true_img))
            row.append(foreground_dice(pred_img, true_img))
            row = np.array(row)
            self._test_metrics_per_image.append(row)

        return loss


    def on_test_epoch_end(self):
        out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "mlruns", "test_metrics_per_image")
        os.makedirs(out_dir, exist_ok=True)

        with open (f'{out_dir}/test_metrics_per_image_{self.args["models"]}_lr-{self.args["lr"]}_wd-{self.args["weight_decay"]}_dropout-{self.args["dropout_val"]}_epoch-{self.args["epochs"]}_batchS-{self.args["test_batch_size"]}.csv', 'w') as f:
            for row in self._test_metrics_per_image:
                f.write(','.join(map(str, row)) + '\n')


    def prepare_data(self):
        """
        Prepares the data for training and prediction
        """
        return {}


    def configure_optimizers(self):
        """
        Initializes the optimizer and learning rate scheduler

        :return: output - Initialized optimizer and scheduler
        """

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.args['lr'])
        self.scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-6, ),
            'monitor': 'val_avg_loss', }

        return [self.optimizer], [self.scheduler]


def iou_fnc(pred, target, n_classes=7):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    count = np.zeros(n_classes)

    for cls in range(0, n_classes):
        pred_inds = pred == cls
        target_inds = target == cls

        intersection = (pred_inds & target_inds).sum().float()
        preds, targets = pred_inds.sum().float(), target_inds.sum().float()
        union = preds + targets - intersection

        if preds.item() == 0 and targets.item() == 0:
            ious.append(1.0)
        else:
            count[cls] += 1
            ious.append(float(intersection) / float(max(union, 1)))

    return np.array(ious), count


def foreground_iou(pred, target):
    pred = pred.view(-1)
    target = target.view(-1)

    pred_fg = (pred != 0).float()
    target_fg = (target != 0).float()

    intersection = (pred_fg * target_fg).sum()
    preds = pred_fg.sum()
    targets = target_fg.sum()
    union = preds + targets - intersection

    if preds.item() == 0 and targets.item() == 0:
        iou = 1.0
    else:
        iou = float(intersection) / float(max(union, 1))

    return iou


def dice_fnc(pred, target, n_classes=7):
    dices = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(0, n_classes):
        pred_inds = (pred == cls).float()
        target_inds = (target == cls).float()

        intersection = (pred_inds * target_inds).sum()
        preds, targets = pred_inds.sum(), target_inds.sum()

        if preds.item() == 0 and targets.item() == 0:
            dices.append(1.0)
        else:
            dices.append(float(2 * intersection) / float(max((preds + targets), 1)))

    return np.array(dices)


def foreground_dice(pred, target):
    pred = pred.view(-1)
    target = target.view(-1)

    pred_fg = (pred != 0).float()
    target_fg = (target != 0).float()

    intersection = (pred_fg * target_fg).sum()
    preds = pred_fg.sum()
    targets = target_fg.sum()

    if preds.item() == 0 and targets.item() == 0:
        dice = 1.0
    else:
        dice = float(2 * intersection) / float(max((preds + targets), 1))

    return dice
