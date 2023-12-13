import pickle
import numpy as np
import tensorflow as tf


class LabelledDatasetLoader(object):
    def __init__(self, dataset, batch_size, return_tensor=True, shuffle=True, pack_i=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.return_tensor = return_tensor
        
        self.shuffle = shuffle
        
        self.dataset_a = self._load_pkl(f"{dataset}_a_{pack_i}.pkl")
        self.dataset_x = self._load_pkl(f"{dataset}_x_{pack_i}.pkl")
        self.dataset_mask = self._load_pkl(f"{dataset}_mask_{pack_i}.pkl")
        self.dataset_label = self._load_pkl(f"{dataset}_label_{pack_i}.pkl")
        self.dataset_name = self._load_pkl(f"{dataset}_name_{pack_i}.pkl")
        self.dataset_smiles = self._load_pkl(f"{dataset}_smiles_{pack_i}.pkl")
        
        self.dataset_size = len(self.dataset_name)
        print(f"Dataset <{dataset}>_<{pack_i}> has been loaded. Size: <{self.dataset_size}>")
        
        self.train_idx, self.train_perm, self.train_size = None, None, 0
        self.test_idx, self.test_perm, self.test_size = None, None, 0
    
    @staticmethod
    def _load_pkl(file_pkl):
        pkl = open(file_pkl, 'rb')
        _ = pickle.load(pkl)
        pkl.close()
        return _
    
    def split_data(self, train_idx, test_idx):
        self.train_idx = train_idx
        self.train_size = len(train_idx)
        
        self.test_size = len(test_idx) // 2
        self.test_idx = test_idx[:self.test_size]
        self.valid_idx = test_idx[self.test_size:]
        self.valid_size = len(self.valid_idx)
        
    def get_valid_set(self):
        self.valid_perm = self.valid_idx
        self.batch_i = 0
        while self.batch_i * self.batch_size < self.valid_size:
            label, name = [], []
            x_real, a_real, mask_real = [], [], []
            x_fake, a_fake, mask_fake = [], [], []
            smiles = []
            low = self.batch_i * self.batch_size
            high = min((self.batch_i + 1) * self.batch_size, self.valid_size)
            for u in range(low, high):
                
                i = self.valid_perm[u]
                label.append(self.dataset_label[i])
                name.append(self.dataset_name[i])
                
                x_real.append(self.dataset_x[i][0])
                a_real.append(self.dataset_a[i][0])
                mask_real.append(self.dataset_mask[i][0])
                
                fake_j = np.random.randint(1, len(self.dataset_x[i]))
                x_fake.append(self.dataset_x[i][fake_j])
                a_fake.append(self.dataset_a[i][fake_j])
                mask_fake.append(self.dataset_mask[i][fake_j])
                
                smi_j = np.random.randint(len(self.dataset_smiles[i]))
                smiles.append(self.dataset_smiles[i][smi_j])
                
            label = np.array(label).astype(np.float32)
            smiles = np.array(smiles).astype(np.float32)
            if len(label.shape) == 1:
                label = label[:, np.newaxis]
            if self.return_tensor:
                label = tf.convert_to_tensor(label)
                x_real = tf.convert_to_tensor(x_real)
                a_real = tf.convert_to_tensor(a_real)
                mask_real = tf.convert_to_tensor(mask_real)
                
                x_fake = tf.convert_to_tensor(x_fake)
                a_fake = tf.convert_to_tensor(a_fake)
                mask_fake = tf.convert_to_tensor(mask_fake)
                
                smiles = tf.convert_to_tensor(smiles)
            else:
                x_real = np.vstack(x_real)
                a_real = np.vstack(a_real)
                mask_real = np.vstack(mask_real)
                
                x_fake = np.vstack(x_fake)
                a_fake = np.vstack(a_fake)
                mask_fake = np.vstack(mask_fake)
                
            real = (a_real, x_real, mask_real)
            fake = (a_fake, x_fake, mask_fake)
            self.batch_i += 1
            yield label, real, fake, smiles, name
    
    def get_train_set(self):
        if self.shuffle:
            self.train_perm = np.random.permutation(self.train_idx)
        else:
            self.train_perm = self.train_idx
        
        self.batch_i = 0
        while self.batch_i * self.batch_size < self.train_size:
            label, name = [], []
            x_real, a_real, mask_real = [], [], []
            x_fake, a_fake, mask_fake = [], [], []
            smiles = []
            low = self.batch_i * self.batch_size
            high = min((self.batch_i + 1) * self.batch_size, self.train_size)
            for u in range(low, high):
                
                i = self.train_perm[u]
                label.append(self.dataset_label[i])
                name.append(self.dataset_name[i])
                
                x_real.append(self.dataset_x[i][0])
                a_real.append(self.dataset_a[i][0])
                mask_real.append(self.dataset_mask[i][0])
                
                fake_j = np.random.randint(1, len(self.dataset_x[i]))
                x_fake.append(self.dataset_x[i][fake_j])
                a_fake.append(self.dataset_a[i][fake_j])
                mask_fake.append(self.dataset_mask[i][fake_j])
                
                smi_j = np.random.randint(len(self.dataset_smiles[i]))
                smiles.append(self.dataset_smiles[i][smi_j])
                
            label = np.array(label).astype(np.float32)
            smiles = np.array(smiles).astype(np.float32)
            if len(label.shape) == 1:
                label = label[:, np.newaxis]
            if self.return_tensor:
                label = tf.convert_to_tensor(label)
                x_real = tf.convert_to_tensor(x_real)
                a_real = tf.convert_to_tensor(a_real)
                mask_real = tf.convert_to_tensor(mask_real)
                
                x_fake = tf.convert_to_tensor(x_fake)
                a_fake = tf.convert_to_tensor(a_fake)
                mask_fake = tf.convert_to_tensor(mask_fake)
                
                smiles = tf.convert_to_tensor(smiles)
            else:
                x_real = np.vstack(x_real)
                a_real = np.vstack(a_real)
                mask_real = np.vstack(mask_real)
                
                x_fake = np.vstack(x_fake)
                a_fake = np.vstack(a_fake)
                mask_fake = np.vstack(mask_fake)
                
            real = (a_real, x_real, mask_real)
            fake = (a_fake, x_fake, mask_fake)
            self.batch_i += 1
            yield label, real, fake, smiles, name
        
    def get_test_set(self):
        self.test_perm = self.test_idx
        self.batch_i = 0
        while self.batch_i * self.batch_size < self.test_size:
            label, name = [], []
            x_real, a_real, mask_real = [], [], []
            x_fake, a_fake, mask_fake = [], [], []
            smiles = []
            low = self.batch_i * self.batch_size
            high = min((self.batch_i + 1) * self.batch_size, self.test_size)
            for u in range(low, high):
                
                i = self.test_perm[u]
                label.append(self.dataset_label[i])
                name.append(self.dataset_name[i])
                
                x_real.append(self.dataset_x[i][0])
                a_real.append(self.dataset_a[i][0])
                mask_real.append(self.dataset_mask[i][0])
                
                fake_j = np.random.randint(1, len(self.dataset_x[i]))
                x_fake.append(self.dataset_x[i][fake_j])
                a_fake.append(self.dataset_a[i][fake_j])
                mask_fake.append(self.dataset_mask[i][fake_j])
                
                smi_j = np.random.randint(len(self.dataset_smiles[i]))
                smiles.append(self.dataset_smiles[i][smi_j])
            
            label = np.array(label).astype(np.float32)
            smiles = np.array(smiles).astype(np.float32)
            if len(label.shape) == 1:
                label = label[:, np.newaxis]
            if self.return_tensor:
                label = tf.convert_to_tensor(label)
                x_real = tf.convert_to_tensor(x_real)
                a_real = tf.convert_to_tensor(a_real)
                mask_real = tf.convert_to_tensor(mask_real)
                
                x_fake = tf.convert_to_tensor(x_fake)
                a_fake = tf.convert_to_tensor(a_fake)
                mask_fake = tf.convert_to_tensor(mask_fake)
            else:
                x_real = np.vstack(x_real)
                a_real = np.vstack(a_real)
                mask_real = np.vstack(mask_real)
                
                x_fake = np.vstack(x_fake)
                a_fake = np.vstack(a_fake)
                mask_fake = np.vstack(mask_fake)
                
                smiles = tf.convert_to_tensor(smiles)
                
            real = (a_real, x_real, mask_real)
            fake = (a_fake, x_fake, mask_fake)
            self.batch_i += 1
            yield label, real, fake, smiles, name
        
    def get_partially_labelled_train_set(self, ratio, seed=123):
        
        _perm = np.random.RandomState(seed).permutation(self.train_idx)
        self.labelled_size = int(self.train_size * ratio)
        self.unlabelled_size = self.train_size - self.labelled_size
        
        self.labelled_idx = _perm[ :self.labelled_size]
        self.unlabelled_idx = _perm[self.labelled_size: ]
        self.labelled_batch_size = int(self.batch_size * ratio)
        self.unlabelled_batch_size = self.batch_size - self.labelled_batch_size
        
        if self.shuffle:
            self.labelled_perm = np.random.permutation(self.labelled_idx)
            self.unlabelled_perm = np.random.permutation(self.unlabelled_idx)
        else:
            self.labelled_perm = self.labelled_idx
            self.unlabelled_perm = self.unlabelled_idx
        
        self.batch_i = 0
        
        print(f"Labelled Size: {self.labelled_size}\t"\
              f"Labelled Batch: {self.labelled_batch_size}\t"\
              f"Unlabelled Size: {self.unlabelled_size}\t"\
              f"Unlabelled Batch: {self.unlabelled_batch_size}\t"\
        )
        
        while (
            self.batch_i * self.labelled_batch_size < self.labelled_size
            and self.batch_i * self.unlabelled_batch_size < self.unlabelled_size
            ):
            # print("WHILE IS RUNNING...")
            label, name = [], []
            x_real, a_real, mask_real = [], [], []
            x_fake, a_fake, mask_fake = [], [], []
            smiles = []
            low = self.batch_i * self.labelled_batch_size
            high = min((self.batch_i + 1) * self.labelled_batch_size, self.labelled_size)
            for u in range(low, high):
                
                i = self.labelled_perm[u]
                label.append(self.dataset_label[i])
                name.append(self.dataset_name[i])
                
                x_real.append(self.dataset_x[i][0])
                a_real.append(self.dataset_a[i][0])
                mask_real.append(self.dataset_mask[i][0])
                
                fake_j = np.random.randint(1, len(self.dataset_x[i]))
                x_fake.append(self.dataset_x[i][fake_j])
                a_fake.append(self.dataset_a[i][fake_j])
                mask_fake.append(self.dataset_mask[i][fake_j])
                
                smi_j = np.random.randint(len(self.dataset_smiles[i]))
                smiles.append(self.dataset_smiles[i][smi_j])
            
            label = np.array(label).astype(np.float32)
            smiles = np.array(smiles).astype(np.float32)
            if len(label.shape) == 1:
                label = label[:, np.newaxis]
            if self.return_tensor:
                label = tf.convert_to_tensor(label)
                x_real = tf.convert_to_tensor(x_real)
                a_real = tf.convert_to_tensor(a_real)
                mask_real = tf.convert_to_tensor(mask_real)
                
                x_fake = tf.convert_to_tensor(x_fake)
                a_fake = tf.convert_to_tensor(a_fake)
                mask_fake = tf.convert_to_tensor(mask_fake)
                
                smiles = tf.convert_to_tensor(smiles)
            else:
                x_real = np.vstack(x_real)
                a_real = np.vstack(a_real)
                mask_real = np.vstack(mask_real)
                
                x_fake = np.vstack(x_fake)
                a_fake = np.vstack(a_fake)
                mask_fake = np.vstack(mask_fake)
                
            real = (a_real, x_real, mask_real)
            fake = (a_fake, x_fake, mask_fake)
            labelled_data = (label, real, fake, smiles, name)
            
            label, name = [], []
            x_real, a_real, mask_real = [], [], []
            x_fake, a_fake, mask_fake = [], [], []
            smiles = []
            low = self.batch_i * self.unlabelled_batch_size
            high = min((self.batch_i + 1) * self.unlabelled_batch_size, self.unlabelled_size)
            for u in range(low, high):
                
                i = self.unlabelled_perm[u]
                label.append(self.dataset_label[i])
                name.append(self.dataset_name[i])
                
                x_real.append(self.dataset_x[i][0])
                a_real.append(self.dataset_a[i][0])
                mask_real.append(self.dataset_mask[i][0])
                
                fake_j = np.random.randint(1, len(self.dataset_x[i]))
                x_fake.append(self.dataset_x[i][fake_j])
                a_fake.append(self.dataset_a[i][fake_j])
                mask_fake.append(self.dataset_mask[i][fake_j])
                
                smi_j = np.random.randint(len(self.dataset_smiles[i]))
                smiles.append(self.dataset_smiles[i][smi_j])
            
            label = np.array(label).astype(np.float32)
            smiles = np.array(smiles).astype(np.float32)
            if len(label.shape) == 1:
                label = label[:, np.newaxis]
            if self.return_tensor:
                label = tf.convert_to_tensor(label)
                x_real = tf.convert_to_tensor(x_real)
                a_real = tf.convert_to_tensor(a_real)
                mask_real = tf.convert_to_tensor(mask_real)
                
                x_fake = tf.convert_to_tensor(x_fake)
                a_fake = tf.convert_to_tensor(a_fake)
                mask_fake = tf.convert_to_tensor(mask_fake)
                
                smiles = tf.convert_to_tensor(smiles)
            else:
                x_real = np.vstack(x_real)
                a_real = np.vstack(a_real)
                mask_real = np.vstack(mask_real)
                
                x_fake = np.vstack(x_fake)
                a_fake = np.vstack(a_fake)
                mask_fake = np.vstack(mask_fake)
                
            real = (a_real, x_real, mask_real)
            fake = (a_fake, x_fake, mask_fake)
            unlabelled_data = (label, real, fake, smiles, name)
            
            self.batch_i += 1
            yield unlabelled_data, labelled_data





