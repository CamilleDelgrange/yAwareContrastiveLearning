import os
import torch
from torch.nn import DataParallel
from tqdm import tqdm
import logging
import umap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap


class yAwareCLModel:

    def __init__(self, net, loss, loader_train, loader_val, config, scheduler=None):
        """

        Parameters
        ----------
        net: subclass of nn.Module
        loss: callable fn with args (y_pred, y_true)
        loader_train, loader_val: pytorch DataLoaders for training/validation
        config: Config object with hyperparameters
        scheduler (optional)
        """
        super().__init__()
        self.logger = logging.getLogger("yAwareCL")
        self.loss = loss
        self.model = net
        self.optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = scheduler
        self.loader = loader_train
        self.loader_val = loader_val
        self.device = torch.device("cuda" if config.cuda else "cpu")
        print(self.device)
        if config.cuda and not torch.cuda.is_available():
            raise ValueError("No GPU found: set cuda=False parameter.")
        self.config = config
        self.metrics = {}

        if hasattr(config, 'pretrained_path') and config.pretrained_path is not None:
            print("pretrained model")
            self.load_model(config.pretrained_path)

        #self.model = DataParallel(self.model).to(self.device)
        self.model = self.model.to(self.device)


    def pretraining(self):
        print(self.loss)
        print(self.optimizer)

        for epoch in range(self.config.nb_epochs):

            ## Training step
            self.model.train()
            nb_batch = len(self.loader)
            training_loss = 0
            pbar = tqdm(total=nb_batch, desc="Training")
            for (inputs, labels, _) in self.loader:
                pbar.update()
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                z_i = self.model(inputs[:, 0, :])
                z_j = self.model(inputs[:, 1, :])
                batch_loss, logits, target = self.loss(z_i, z_j, labels)
                batch_loss.backward()
                self.optimizer.step()
                training_loss += float(batch_loss) / nb_batch
            pbar.close()

            ## Validation step
            nb_batch = len(self.loader_val)
            pbar = tqdm(total=nb_batch, desc="Validation")
            val_loss = 0
            val_values = {}
            embeddings = []
            ages_list = []
            diagnoses_list = []
            with torch.no_grad():
                self.model.eval()
                for (inputs, labels, diagnosis) in self.loader_val:
                    pbar.update()
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    z_i = self.model(inputs[:, 0, :])
                    z_j = self.model(inputs[:, 1, :])
                    batch_loss, logits, target = self.loss(z_i, z_j, labels)
                    val_loss += float(batch_loss) / nb_batch
                    for name, metric in self.metrics.items():
                        if name not in val_values:
                            val_values[name] = 0
                        val_values[name] += metric(logits, target) / nb_batch
                    embeddings.append(z_i.detach().cpu().numpy())
                    ages_list.append(labels.detach().cpu().numpy())
                    diagnoses_list.append(diagnosis.detach().cpu().numpy())
            pbar.close()

            metrics = "\t".join(["Validation {}: {:.4f}".format(m, v) for (m, v) in val_values.items()])
            print("Epoch [{}/{}] Training loss = {:.4f}\t Validation loss = {:.4f}\t".format(
                epoch+1, self.config.nb_epochs, training_loss, val_loss)+metrics, flush=True)

            if self.scheduler is not None:
                self.scheduler.step()

            if (epoch % self.config.nb_epochs_per_saving == 0 or epoch == self.config.nb_epochs - 1) and epoch > 0:
                torch.save({
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict()},
                    os.path.join(self.config.checkpoint_dir, "{name}_epoch_{epoch}.pth".
                                 format(name="y-Aware_Contrastive_MRI", epoch=epoch)))
        embeddings = np.concatenate(embeddings, axis=0)
        ages_list = np.concatenate(ages_list, axis=0)
        diagnoses_list = np.concatenate(diagnoses_list, axis=0)
        self.plot_umap(embeddings, ages_list, diagnoses_list, title=f'Validation UMAP Epoch Pre-training {epoch+1}', output_path=os.path.join(self.config.checkpoint_dir, f'validation_umap_last_epoch_pretraining_.png'))

    def plot_umap(self, embeddings, ages, diagnoses, title="UMAP", output_path=None):
        reducer = umap.UMAP()
        umap_embeddings = reducer.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 7))

        # Adaptive size for age
        age_sizes = (ages - np.min(ages)) / (np.max(ages) - np.min(ages)) * 100 + 10  # Normalize ages and scale

        # Distinct colors for control and stroke
        control_color = 'blue'
        stroke_color = 'red'
        colors = np.array([control_color if diagnosis == 0 else stroke_color for diagnosis in diagnoses])
        
        # Plot UMAP with diagnoses
        scatter = plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=colors, s=age_sizes, alpha=0.7)
        
        # Create custom legends
        handles = [Line2D([0], [0], marker='o', color='w', label='Control', markersize=10, markerfacecolor=control_color),
                Line2D([0], [0], marker='o', color='w', label='Stroke', markersize=10, markerfacecolor=stroke_color)]

        age_handles = [Line2D([0], [0], marker='o', color='w', label=f'{age}', markersize=5, markerfacecolor='gray')
                    for age in sorted(np.unique(ages))]

        legend1 = plt.legend(handles=handles, title="Diagnosis")
        plt.gca().add_artist(legend1)
        plt.legend(handles=age_handles, title="Age", loc='upper right')

        plt.title(title)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)

        plt.show() 


    def fine_tuning(self):
        print(self.loss)
        print(self.optimizer)

        for epoch in range(self.config.nb_epochs):
            ## Training step
            self.model.train()
            nb_batch = len(self.loader)
            training_loss = 0.0
            pbar = tqdm(total=nb_batch, desc="Training")
            for (inputs, labels, diagnosis) in self.loader:
                pbar.update()
                inputs = inputs.to(self.device)
                diagnosis = diagnosis.to(self.device)
                self.optimizer.zero_grad()
                y = self.model(inputs)
                batch_loss = self.loss(y,diagnosis)
                batch_loss.backward()
                self.optimizer.step()
                print(training_loss)
                print(batch_loss)
                training_loss += batch_loss.item() / nb_batch
                print(training_loss)
            pbar.close()

            ## Validation step
            nb_batch = len(self.loader_val)
            pbar = tqdm(total=nb_batch, desc="Validation")
            val_loss = 0.0
            embeddings = []
            diagnosis_list = []
            ages_list = []
            with torch.no_grad():
                self.model.eval()
                for (inputs, labels, diagnosis) in self.loader_val:
                    pbar.update()
                    inputs = inputs.to(self.device)
                    diagnosis = diagnosis.to(self.device)
                    y = self.model(inputs)
                    batch_loss = self.loss(y, diagnosis)
                    val_loss += batch_loss.item() / nb_batch

                    embeddings.append(y.detach().cpu().numpy())
                    diagnosis_list.append(diagnosis.detach().cpu().numpy())
                    ages_list.append(labels.detach().cpu().numpy())
            pbar.close()

            print("Epoch [{}/{}] Training loss = {:.4f}\t Validation loss = {:.4f}\t".format(
                epoch+1, self.config.nb_epochs, training_loss, val_loss), flush=True)

            if self.scheduler is not None:
                self.scheduler.step()

        embeddings = np.concatenate(embeddings, axis=0)
        ages_list = np.concatenate(ages_list, axis=0)
        diagnosis_list = np.concatenate(diagnosis_list, axis=0)
        self.plot_umap(embeddings, ages_list, diagnosis_list, title=f'Fine-tuning validation UMAP Last Epoch', output_path=os.path.join(self.config.checkpoint_dir, f'validation_umap_last_epoch_finetuning_.png'))


    def load_model(self, path):
        checkpoint = None
        try:
            checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        except BaseException as e:
            self.logger.error('Impossible to load the checkpoint: %s' % str(e))
        if checkpoint is not None:
            try:
                if hasattr(checkpoint, "state_dict"):
                    unexpected = self.model.load_state_dict(checkpoint.state_dict())
                    #print('Model loading info: {}'.format(unexpected))
                    self.logger.info('Model loading info: {}'.format(unexpected))
                elif isinstance(checkpoint, dict):
                    if "model" in checkpoint:
                        unexpected = self.model.load_state_dict(checkpoint["model"], strict=False)
                        self.logger.info('Model loading info: {}'.format(unexpected))
                        #print('Model loading info: {}'.format(unexpected))
                else:
                    unexpected = self.model.load_state_dict(checkpoint)
                    #print('Model loading info: {}'.format(unexpected))
                    self.logger.info('Model loading info: {}'.format(unexpected))
            except BaseException as e:
                raise ValueError('Error while loading the model\'s weights: %s' % str(e))





