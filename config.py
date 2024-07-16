
PRETRAINING = 0
FINE_TUNING = 1

class Config:

    def __init__(self, mode):
        assert mode in {PRETRAINING, FINE_TUNING}, "Unknown mode: %i"%mode

        self.mode = mode

        if self.mode == PRETRAINING:
            self.batch_size = 8 #64
            self.nb_epochs_per_saving = 1
            self.pin_mem = False
            self.num_cpu_workers = 8
            self.nb_epochs = 2
            self.cuda = False
            # Optimizer
            self.lr = 1e-4
            self.weight_decay = 5e-5
            # Hyperparameters for our y-Aware InfoNCE Loss
            self.sigma = 5 # depends on the meta-data at hand
            self.temperature = 0.1
            self.tf = "all_tf"
            self.model = "DenseNet"


            # Paths to the data
            self.data_train = "synthetic_dataset/data.npy"
            self.label_train = "synthetic_dataset/metadata.csv"

            self.data_val = "synthetic_dataset/data.npy"
            self.label_val = "synthetic_dataset/metadata.csv"
            self.input_size = (1, 121, 145, 121)
            self.label_name = "age"

            self.checkpoint_dir = "synthetic_dataset/checkpoints/"

        elif self.mode == FINE_TUNING:
            ## We assume a classification task here
            self.batch_size = 8
            self.nb_epochs_per_saving = 1
            self.pin_mem = False
            self.num_cpu_workers = 0
            self.nb_epochs = 2
            self.cuda = False
            self.tf = "cutout"
            # Paths to the data
            self.data_train = "synthetic_dataset/data.npy"
            self.label_train = "synthetic_dataset/metadata.csv"

            self.data_val = "synthetic_dataset/data.npy"
            self.label_val = "synthetic_dataset/metadata.csv"
            self.input_size = (1, 121, 145, 121)
            self.label_name = "age"
            # Optimizer
            self.lr = 1e-4
            self.weight_decay = 5e-5

            self.pretrained_path = "C:\\Users\\camil\\Documents\\GitHub\\yAwareContrastiveLearning\\pretrained_model\\DenseNet_HCP_IXI_window-0.25_0_epoch_30.pth"
            self.num_classes = 2
            self.model = "DenseNet"
            self.checkpoint_dir = "synthetic_dataset/checkpoints/"
