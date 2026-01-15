import copy
import os
import csv
from utils.conf import base_path, log_path
from utils.util import create_if_not_exists
useless_args = ['pub_aug','public_len','public_dataset','structure', 'model', 'csv_log', 'device_id', 'seed',
                'tensorboard','conf_jobnum','conf_timestamp','conf_host']
import pickle
import datetime
class CsvWriter:
    def __init__(self, args, private_dataset):
        self.args = args
        self.private_dataset = private_dataset
        self.model_folder_path = self._model_folder_path()
        self.para_folder_path = self.model_folder_path  # Always use a single folder
        self._write_args()
        print(self.para_folder_path)

    def _model_folder_path(self):
        # Define a fixed folder path
        args = self.args
        data_path = log_path() + args.dataset
        create_if_not_exists(data_path)

        model_path = os.path.join(data_path, args.model)
        create_if_not_exists(model_path)

        # Always return the same folder
        return model_path

    def generate_filename(self, base_name):
        # Generate filenames inside the fixed folder
        params_to_include = ['dataset', 'model']
        param_strings = [f"{param}_{getattr(self.args, param)}" for param in params_to_include]
        filename = f"{base_name}_{'_'.join(param_strings)}.csv"
        return os.path.join(self.para_folder_path, filename)

    def write_acc(self, accs_dict, mean_acc_list):
        # Write accuracy files
        acc_path = os.path.join(self.para_folder_path, 'all_acc.csv')
        self._write_all_acc(accs_dict)

        mean_acc_path = os.path.join(self.para_folder_path, 'mean_acc.csv')
        self._write_mean_acc(mean_acc_list)

    def _write_args(self):
        # Overwrite args.csv in the same folder
        args = copy.deepcopy(vars(self.args))
        for cc in useless_args:
            if cc in args:
                del args[cc]

        args_path = os.path.join(self.para_folder_path, 'args.csv')
        with open(args_path, 'w') as tmp:
            writer = csv.DictWriter(tmp, fieldnames=args.keys())
            writer.writeheader()
            writer.writerow(args)

    def _write_mean_acc(self, acc_list):
        # Overwrite mean_acc.csv
        mean_path = self.generate_filename('mean_acc')
        with open(mean_path, 'w') as result_file:
            for epoch in range(self.args.communication_epoch):
                result_file.write(f'epoch_{epoch},')
            result_file.write('\n')
            result_file.write(','.join(map(str, acc_list)))
            result_file.write('\n')

    def _write_all_acc(self, all_acc_list):
        # Overwrite all_acc.csv
        all_path = self.generate_filename('all_acc')
        with open(all_path, 'w') as result_file:
            for epoch in range(self.args.communication_epoch):
                result_file.write(f'epoch_{epoch},')
            result_file.write('\n')
            for key, method_result in all_acc_list.items():
                result_file.write(','.join(map(str, method_result)))
                result_file.write('\n')

    def write_loss(self, loss_dict, loss_name):
        # Overwrite loss.pkl
        loss_path = os.path.join(self.para_folder_path, f'{loss_name}.pkl')
        with open(loss_path, 'wb') as f:
            pickle.dump(loss_dict, f)
