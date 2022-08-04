from pathlib import Path

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torchio as tio
import wandb
from torch.utils.data import DataLoader

import metrics
from unet3d import Unet3d


class SegmentationModel3d(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()

        self._hparams = hparams

        self.model = Unet3d(
            in_classes=1,
            out_classes=1,
            num_blocks=hparams['blocks'],
            p_dropout=hparams['p_dropout'],
            initial_features=hparams['initial_features'])

        if hparams['loss'] == 'tversky':
            self.loss = metrics.TverskyLoss(hparams['alpha'])
        if hparams['loss'] == 'dice':
            self.loss = metrics.DiceLoss()

        self.sensitivity = metrics.Sensitivity()
        self.specificity = metrics.Specificity()
        self.accuracy = metrics.Accuracy()
        self.jaccard = metrics.Jaccard()
        self.stddev = metrics.StdDev()
        self.bias = metrics.Bias()

        data_path = Path(self.hparams['data_path'])
        files = list(data_path.glob('*.*'))
        self.files_orig = sorted(
            list(filter(lambda file: 'orig.nii.gz' in str(file), files)))
        self.files_masks = sorted(
            list(filter(lambda file: 'masks.nii.gz' in str(file), files)))
        
        self.files_orig.pop(106)
        self.files_orig.pop(106)
        self.files_masks.pop(106)
        self.files_masks.pop(106)

    def configure_optimizers(self):
        '''
            Setup the Adam Optimizer - initalized with the hyperparamters learning rate
        '''
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self._hparams['learning_rate'])
        return optimizer

    def forward(self, subject):
        '''
            Make a prediction with the model, given an subject
        '''
        # build sampler and aggregator
        sampler = tio.inference.GridSampler(
            subject, patch_size=self.hparams['patch_size'], patch_overlap=4)
        aggregator = tio.inference.GridAggregator(sampler)
        patch_loader = torch.utils.data.DataLoader(
            sampler, batch_size=self.hparams['batch_size'])
        for patch in patch_loader:
            locations = patch[tio.LOCATION]
            x = patch['t1'][tio.DATA]
            patch_prediction = self.model(x.to(self.device))
            aggregator.add_batch(patch_prediction, locations)

        prediction = aggregator.get_output_tensor()
        return prediction

    
    def split_batch(self, batch):
        '''
            Split the batch into the two tensors x and y.
            x is the data 
            y is the label
        '''
        x = batch['t1'][tio.DATA]
        y = batch['label'][tio.DATA]
        return x, y

    def metrics(self, prediction, y):
        '''
            Given a prediction and a label map calculate different metrics
            Metrics:
                Loss - Specified by the hyperparameters - either Tversky or Dice Loss
                Sensitivity - What percentage of the positive predictions were correct (tp)/(fp + tp)
                Specifitiy - What percentage of the negative predictions were correct (tn)/(fn + tn)
                Accuracy - What percentage of the predictions were correct (tn+tp)/(tn+fn+tp+fn)
        '''

        loss = self.loss(prediction, y)
        sensitivity = self.sensitivity(prediction, y)
        specificity = self.specificity(prediction, y)
        accuracy = self.accuracy(prediction, y)
        jaccard = self.jaccard(prediction, y)
        std_dev = self.stddev(prediction, y)
        bias = self.bias(prediction, y)
        return {
            'loss': loss,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'accuracy': accuracy,
            'jaccard': jaccard,
            'stddev': std_dev,
            'bias': bias
        }

    def log_metrics(self, level, metrics):
        '''
            Logg a dictionary of metrics to weights and biases.
            Level should be 'train', 'validation' or 'test'
        '''
        for key in metrics:
            self.log(f'{level}_{key}', metrics[key])

    def log_prediction(self, level, x, prediction, y):

        def cm_to_inch(value):
            return value/2.54

        plt.figure(figsize=(cm_to_inch(40), cm_to_inch(30)))

        for i in range(3):
            plt.subplot(2, 3, i+1)
            plt.title(f"Label in axis {i}")
            plt.xticks([])
            plt.yticks([])
            plt.imshow(x[0].sum(dim=i).cpu().numpy(), cmap='gray')
            plt.imshow(y[0].sum(dim=i).cpu().numpy(), cmap='Reds', alpha=0.5)

            plt.subplot(2, 3, 3+i+1)
            plt.title(f"Prediction in axis {i}")
            plt.xticks([])
            plt.yticks([])
            plt.imshow(x[0].sum(dim=i).cpu().numpy(), cmap='gray')
            plt.imshow(y[0].sum(dim=i).cpu().numpy(), cmap='Blues', alpha=0.5)
            plt.imshow(prediction[0].sum(
                dim=i).cpu().numpy(), cmap='Reds', alpha=0.5)

        wandb.log({f'{level}_sample': plt})

    def training_step(self, batch, batch_idx):
        x, y = self.split_batch(batch)

        x = x.float()
        prediction = self.model(x, use_dropout=True)

        metrics = self.metrics(prediction, y)

        if metrics['loss'].isnan():
            print(f'\n\nX: {x.isnan().any()}, shape: {x.shape}')
            print(
                f'Prediction: {prediction.isnan().any()}, shape: {prediction.shape}')
            print(f'Y: {y.isnan().any()}, shape: {y.shape}')

        self.log_metrics('train', metrics)

        if batch_idx == 0 and self.current_epoch % self.hparams['log_image_every_n'] == 0:
            x = x.detach()
            y = y.detach()
            prediction = prediction.detach()
            for i in range(len(x)):
                self.log_prediction('train', x[i], prediction[i], y[i])

        return metrics

    def validation_step(self, batch, batch_idx):

        with torch.no_grad():
            # build subject
            batch['t1'] = tio.Image(tensor=batch['t1']['data'][0])
            batch['label'] = tio.LabelMap(tensor=batch['label']['data'][0])
            subject = tio.Subject(batch)

            prediction = self(subject)

            x, y = self.split_batch(batch)
            x = x.cpu()
            y = y.cpu()
            metrics = self.metrics(prediction, y)
            self.log_metrics('validation', metrics)
            self.log_prediction('validation', x, prediction, y)
            return {
                f'validation_{key}': metrics[key]
                for key in metrics.keys()
            }

    def validation_epoch_end(self, outputs):
        avg = {f'avg_{key}': 0 for key in outputs[0].keys()}

        for out in outputs:
            for key in out.keys():
                avg[f'avg_{key}'] += out[key]

        for key in avg.keys():
            avg[key] /= len(outputs)
            self.log(key, avg[key])

    def test_step(self, batch, batch_idx):

        with torch.no_grad():
            # build subject
            batch['t1'] = tio.Image(tensor=batch['t1']['data'][0])
            batch['label'] = tio.LabelMap(tensor=batch['label']['data'][0])
            subject = tio.Subject(batch)

            prediction = self(subject)

            x, y = self.split_batch(batch)
            x = x.cpu()
            y = y.cpu()
            metrics = self.metrics(prediction, y)
            self.log_metrics('test', metrics)
            self.log_prediction('test', x, prediction, y)
            return metrics['loss']

    def get_dataset(self, start, stop, transform):

        subjects = [
            tio.Subject(
                t1=tio.ScalarImage(orig),
                label=tio.LabelMap(mask),
            )
            for orig, mask in zip(self.files_orig[start:stop], self.files_masks[start:stop])
        ]
        return tio.SubjectsDataset(subjects, transform=transform)

    def prepare_data(self):
        preprocessing = tio.RescaleIntensity(out_min_max=(0, 1))

        augmentation = tio.OneOf({
            tio.RandomAffine(): self.hparams['p_affine'],
            tio.RandomElasticDeformation(
                max_displacement=5.5,
                num_control_points=5
            ): self.hparams['p_elastic'],
        },
            p=self.hparams['p_affine_or_elastic']
        )

        train_transform = tio.Compose([preprocessing, augmentation])

        self.train_dataset = self.get_dataset(0, 100, train_transform)
        self.val_dataset = self.get_dataset(100, 107, preprocessing)

    def train_dataloader(self):
        patch_size = self.hparams['patch_size']
        samples_per_volume = self.hparams['samples_per_volume']
        queue_length = samples_per_volume*self.hparams['queue_length']

        sampler = tio.data.LabelSampler(patch_size, 'label', label_probabilities={
            0: self.hparams['random_sample_ratio'],
            1: 1,})

        patches_queue = tio.Queue(
            self.train_dataset,
            queue_length,
            samples_per_volume,
            sampler,
            shuffle_subjects=True,
            num_workers=0,
        )

        patches_loader = DataLoader(
            patches_queue,
            batch_size=self.hparams['batch_size'],
            num_workers=0,  # this must be 0
        )

        return patches_loader

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1)
