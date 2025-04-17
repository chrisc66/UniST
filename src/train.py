import torch
from torch.optim import AdamW
import random
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
import math
import time
import matplotlib.pyplot as plt
import json
import os
import copy


class TrainLoop:
    def __init__(self, args, writer, model, data, test_data, val_data, device, early_stop = 5):
        self.args = args
        self.writer = writer
        self.model = model
        self.data = data
        self.test_data = test_data
        self.val_data = val_data
        self.device = device
        self.lr_anneal_steps = args.lr_anneal_steps
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.opt = AdamW([p for p in self.model.parameters() if p.requires_grad==True], lr=args.lr, weight_decay=self.weight_decay)
        self.log_interval = args.log_interval
        self.best_rmse_random = 1e9
        self.warmup_steps=5
        self.min_lr = args.min_lr
        self.best_rmse = 1e9
        self.early_stop = early_stop
        
        self.mask_list = {'random':[0.5],'temporal':[0.5],'tube':[0.5],'block':[0.5]}


    def run_step(self, batch, step, mask_ratio, mask_strategy,index, name):
        self.opt.zero_grad()
        loss, num, loss_real, num2 = self.forward_backward(batch, step, mask_ratio, mask_strategy,index=index, name = name)

        self._anneal_lr()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.args.clip_grad)
        self.opt.step()
        return loss, num, loss_real, num2


    def extract_predictions(self, pred, target, mask, input_size, dataset_name):
        """
        Extract both masked and unmasked predictions
        Args:
            pred: model predictions [N, L, patch_size**2 * t_patch_size]
            target: ground truth [N, L, patch_size**2 * t_patch_size]
            mask: masking tensor [N, L]
            input_size: tuple of (T, H, W) for reshaping
            dataset_name: name of dataset for scaler
        """
        N = pred.shape[0]
        T, H, W = input_size
        p = self.model.args.patch_size
        u = self.model.args.t_patch_size

        T = T * 2       # double sequence length (hist == prediction)
        t = T // u      # temporal patches
        h = H           # height patches (already divided in forward_encoder)
        w = W           # width patches (already divided in forward_encoder)
        H_orig = H * p  # Convert back to pixel dimensions
        W_orig = W * p  # Convert back to pixel dimensions

        self.model.patch_info = (N, T, H_orig, W_orig, p, u, t, h, w)

        # Masked data
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, pred.shape[-1])
        pred_masked = pred[mask_expanded==1]
        target_masked = target[mask_expanded==1]

        # Unpatch data
        pred_patched = copy.deepcopy(pred).reshape(N, t * h * w, u * p * p)
        target_patched = copy.deepcopy(target).reshape(N, t * h * w, u * p * p)

        pred_unpatched = self.model.unpatchify(pred_patched).reshape(N, T, H_orig, W_orig)
        target_unpatched = self.model.unpatchify(target_patched).reshape(N, T, H_orig, W_orig)

        # Un-normalize data original scale
        pred_numpy = pred_unpatched.detach().cpu().numpy()
        target_numpy = target_unpatched.detach().cpu().numpy()

        pred_unnormalized = self.args.scaler[dataset_name].inverse_transform(pred_numpy.reshape(-1, 1)).reshape(pred_numpy.shape)
        target_unnormalized = self.args.scaler[dataset_name].inverse_transform(target_numpy.reshape(-1, 1)).reshape(target_numpy.shape)

        os.makedirs(f"{self.args.model_path}/plots", exist_ok=True)

        C = 0

        # 1. Masked Data
        plt.figure(figsize=(100, 30))

        plt.subplot(311)
        plt.plot(target_masked.detach().cpu().numpy(), label='Target')
        plt.title(f'Masked Target ({target_masked.shape})')
        plt.legend()
        plt.grid()

        plt.subplot(312)
        plt.plot(pred_masked.detach().cpu().numpy(), label='Prediction')
        plt.title(f'Masked Prediction ({pred_masked.shape})')
        plt.legend()
        plt.grid()

        plt.subplot(313)
        plt.plot(np.abs(target_masked.detach().cpu().numpy() - pred_masked.detach().cpu().numpy()), label='Absolute Error')
        plt.title('Masked Absolute Error')
        plt.legend()
        plt.grid()

        plt.suptitle('Masked Data Visualization')
        plt.tight_layout()
        plt.savefig(f"{self.args.model_path}/plots/1_masked.png")
        plt.close()

        # 1. Masked Data (Partial)
        LEN = 50
        plt.figure(figsize=(30, 10))

        plt.subplot(131)
        plt.bar(range(LEN), target_masked[:LEN].detach().cpu().numpy(), label='Target')
        plt.title(f'Masked Target ({target_masked[:LEN].shape})')
        plt.legend()
        plt.grid()

        plt.subplot(132)
        plt.bar(range(LEN), pred_masked[:LEN].detach().cpu().numpy(), label='Prediction')
        plt.title(f'Masked Prediction ({pred_masked[:LEN].shape})')
        plt.legend()
        plt.grid()

        plt.subplot(133)
        plt.bar(range(LEN), np.abs(target_masked[:LEN].detach().cpu().numpy() - pred_masked[:LEN].detach().cpu().numpy()), label='Absolute Error')
        plt.title('Masked Absolute Error')
        plt.legend()
        plt.grid()

        plt.suptitle('Masked Data Visualization')
        plt.tight_layout()
        plt.savefig(f"{self.args.model_path}/plots/1_masked_partial.png")
        plt.close()

        # 2 Patched Data
        plt.figure(figsize=(10, 100))

        plt.subplot(131)
        plt.imshow(target_patched[C].detach().cpu().numpy(), origin='lower', vmin=-1, vmax=1)
        plt.title(f'Target ({target_patched[C].shape})')
        plt.colorbar()

        plt.subplot(132)
        plt.imshow(pred_patched[C].detach().cpu().numpy(), origin='lower', vmin=-1, vmax=1)
        plt.title(f'Prediction ({pred_patched[C].shape})')
        plt.colorbar()

        plt.subplot(133)
        plt.imshow(np.abs(target_patched[C].detach().cpu().numpy() - pred_patched[C].detach().cpu().numpy()), origin='lower', vmin=-1, vmax=1)
        plt.title('Absolute Error')
        plt.colorbar()

        plt.suptitle('Patched Visualization')
        plt.tight_layout()
        plt.savefig(f"{self.args.model_path}/plots/2_patched.png")
        plt.close()

        # 3 Unpatched Data
        plt.figure(figsize=(50, 10))
        timesteps = range(0, T)
        for i, t_idx in enumerate(timesteps):
            plt.subplot(3, len(timesteps), i + 1)
            plt.imshow(target_unpatched[C, t_idx].detach().cpu().numpy(), vmin=-1, vmax=1)
            plt.gca().invert_xaxis()
            plt.title(f'Target t={t_idx}')
            plt.colorbar()

            plt.subplot(3, len(timesteps), i + 1 + len(timesteps))
            plt.imshow(pred_unpatched[C, t_idx].detach().cpu().numpy(), vmin=-1, vmax=1)
            plt.gca().invert_xaxis()
            plt.title(f'Prediction t={t_idx}')
            plt.colorbar()

            plt.subplot(3, len(timesteps), i + 1 + 2 * len(timesteps))
            plt.imshow(np.abs(pred_unpatched[C, t_idx].detach().cpu().numpy() - target_unpatched[C, t_idx].detach().cpu().numpy()), vmin=-1, vmax=1)
            plt.gca().invert_xaxis()
            plt.title(f'Absolute Error t={t_idx}')
            plt.colorbar()

        plt.suptitle('Unpatched Visualization')
        plt.tight_layout()
        plt.savefig(f"{self.args.model_path}/plots/3_unpatched.png")
        plt.close()

        # 4 Un-normalized Data
        plt.figure(figsize=(50, 10))
        timesteps = range(0, T)
        for i, t_idx in enumerate(timesteps):
            plt.subplot(3, len(timesteps), i + 1)
            plt.imshow(target_unnormalized[C, t_idx], vmin=0, vmax=22)
            plt.gca().invert_xaxis()
            plt.title(f'Target t={t_idx}')
            plt.colorbar()

            plt.subplot(3, len(timesteps), i + 1 + len(timesteps))
            plt.imshow(pred_unnormalized[C, t_idx], vmin=0, vmax=22)
            plt.gca().invert_xaxis()
            plt.title(f'Prediction t={t_idx}')
            plt.colorbar()

            plt.subplot(3, len(timesteps), i + 1 + 2 * len(timesteps))
            plt.imshow(np.abs(pred_unnormalized[C, t_idx] - target_unnormalized[C, t_idx]), vmin=0, vmax=22)
            plt.gca().invert_xaxis()
            plt.title(f'Absolute Error t={t_idx}')
            plt.colorbar()

        plt.suptitle('Un-normalized Visualization (Original Scale)')
        plt.tight_layout()
        plt.savefig(f"{self.args.model_path}/plots/4_unnormalized.png")
        plt.close()

        # 5. Compute accuracy masks for each threshold
        epsilon = 1
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0]
        threshold_labels = [f'{int(thresh * 100)}%' if thresh <= 1.0 else f'{int(thresh)}00%' for thresh in thresholds]
        N, T, H_orig, W_orig = pred_unnormalized.shape
        # For each timestep, create a grid showing where prediction matches target within threshold
        acc_per_timestep = {label: [] for label in threshold_labels}
        plt.figure(figsize=(160, 60))
        for t_idx in range(T // 2, T):
            for i, (thresh, label) in enumerate(zip(thresholds, threshold_labels)):
                mask_within = np.abs(pred_unnormalized[C, t_idx] - target_unnormalized[C, t_idx]) <= (np.abs(target_unnormalized[C, t_idx]) * thresh + epsilon)
                subplot_idx = (t_idx - T // 2) * len(thresholds) + i + 1
                plt.subplot(T - T // 2, len(thresholds), subplot_idx)
                plt.imshow(mask_within, cmap='Greens', vmin=0, vmax=1)
                for y in range(H_orig):
                    for x in range(W_orig):
                        pred_val = pred_unnormalized[C, t_idx, y, x]
                        target_val = target_unnormalized[C, t_idx, y, x]
                        plt.text(x, y, f'{pred_val:.1f}\n{target_val:.1f}',
                            ha='center', va='center', fontsize=6,
                            color='black' if mask_within[y, x] else 'gray',
                            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1'))
                # Compute accuracy for this threshold and timestep
                acc = np.sum(mask_within) / (H_orig * W_orig)
                acc_per_timestep[label].append(acc)
                plt.title(f'Timestep {t_idx} within {label}, accuracy {acc * 100:.2f}%')
                plt.axis('off')
        plt.suptitle(f'Prediction matches target within X%')
        plt.tight_layout()
        plt.savefig(f"{self.args.model_path}/plots/5_accuracy_within_thresholds_grid.png")
        plt.close()

        # 5. Plot accuracy within X percent for each timestep
        plt.figure(figsize=(16, 6))
        for label in threshold_labels:
            plt.plot(range(T // 2, T), acc_per_timestep[label], marker='o', label=f'Within {label}')
        plt.xlabel('Timestep')
        plt.ylabel('Accuracy (fraction within threshold)')
        plt.title('Accuracy within X% per Timestep')
        plt.ylim(0, 1)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"{self.args.model_path}/plots/5_accuracy_within_thresholds.png")
        plt.close()

        # Save json files
        with open(f"{self.args.model_path}/plots/pred_model.json", "w") as pred_file:
            json.dump(pred.tolist(), pred_file)
        with open(f"{self.args.model_path}/plots/target_model.json", "w") as target_file:
            json.dump(target.tolist(), target_file)
        with open(f"{self.args.model_path}/plots/pred_masked.json", "w") as pred_file:
            json.dump(pred_masked.tolist(), pred_file)
        with open(f"{self.args.model_path}/plots/target_masked.json", "w") as target_file:
            json.dump(target_masked.tolist(), target_file)
        with open(f"{self.args.model_path}/plots/pred_patched.json", "w") as pred_file:
            json.dump(pred_patched.tolist(), pred_file)
        with open(f"{self.args.model_path}/plots/target_patched.json", "w") as target_file:
            json.dump(target_patched.tolist(), target_file)
        with open(f"{self.args.model_path}/plots/pred_unpatched.json", "w") as pred_file:
            json.dump(pred_unpatched.tolist(), pred_file)
        with open(f"{self.args.model_path}/plots/target_unpatched.json", "w") as target_file:
            json.dump(target_unpatched.tolist(), target_file)
        with open(f"{self.args.model_path}/plots/pred_unnormalized.json", "w") as pred_file:
            json.dump(pred_unnormalized.tolist(), pred_file)
        with open(f"{self.args.model_path}/plots/target_unnormalized.json", "w") as target_file:
            json.dump(target_unnormalized.tolist(), target_file)

        return {
            'masked': {
                'predictions': pred_masked.detach().cpu().numpy(),
                'targets': target_masked.detach().cpu().numpy()
            },
            'full': {
                'normalized': {
                    'predictions': pred_numpy,
                    'targets': target_numpy
                },
                'original_scale': {
                    'predictions': pred_unnormalized,
                    'targets': target_unnormalized
                }
            },
            'spatial_shape': (N, T, H, W),
            'mask': mask.detach().cpu().numpy()
        }


    def Sample(self, test_data, step, mask_ratio, mask_strategy, seed=None, dataset='', index=0, Type='val'):
        
        print(f"Sample, Type {Type}, dataset {dataset}, mask_strategy {mask_strategy}, mask_ratio {mask_ratio}, index {index}")
        with torch.no_grad():
            all_predictions = []

            error_mae, error_norm, error, num, error2, num2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0]
            acc_within = {thresh: 0 for thresh in thresholds}
            acc_total = 0

            for idx, batch in enumerate(test_data[index]):
                
                loss, _, pred, target, mask, ids_restore, input_size = self.model_forward(batch, self.model, mask_ratio, mask_strategy, seed=seed, data = dataset, mode='forward')

                # Extract both masked and unmasked predictions
                results = self.extract_predictions(pred, target, mask, input_size, dataset)
                all_predictions.append(results)

                pred = torch.clamp(pred, min=-1, max=1)

                pred_mask = pred.squeeze(dim=2)
                target_mask = target.squeeze(dim=2)

                # Inverse transform to original scale
                pred_vals = self.args.scaler[dataset].inverse_transform(pred_mask[mask==1].reshape(-1,1).detach().cpu().numpy()).flatten()
                target_vals = self.args.scaler[dataset].inverse_transform(target_mask[mask==1].reshape(-1,1).detach().cpu().numpy()).flatten()

                epsilon = 1
                for thresh in thresholds:
                    within = np.abs(pred_vals - target_vals) <= (np.abs(target_vals) * thresh + epsilon)
                    acc_within[thresh] += np.sum(within)
                acc_total += len(target_vals)

                error += mean_squared_error(pred_vals, target_vals, squared=True) * mask.sum().item()
                error_mae += mean_absolute_error(pred_vals, target_vals) * mask.sum().item()
                error_norm += loss.item() * mask.sum().item()
                num += mask.sum().item()
                num2 += (1-mask).sum().item()

        rmse = np.sqrt(error / num)
        mae = error_mae / num
        loss_test = error_norm / num
        accuracy_metrics = {thresh: (acc_within[thresh] / acc_total if acc_total > 0 else 0.0) for thresh in thresholds}

        print(f"Dataset: {dataset}")
        print(f"Index: {index}")
        print(f"Step: {step}")
        print(f"Seed: {seed}")
        print(f"Type: {Type}")
        print(f"Batch: {len(batch)}")
        print(f"Predictions: {pred_vals.shape}")
        print(f"Targets: {target_vals.shape}")
        print(f"All Predictions: {len(all_predictions)}")
        print(f"Mask Strategy: {mask_strategy}")
        print(f"Mask Ratio: {mask_ratio}")

        print(f"Accuracy within thresholds:")
        for thresh in thresholds:
            print(f"  {int(thresh*100)}%: {accuracy_metrics[thresh]:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"Loss: {loss_test:.4f}")
        print(f"Num: {num:.4f}")
        print(f"Num2: {num2:.4f}")        

        return rmse, mae, loss_test, accuracy_metrics


    def Evaluation(self, test_data, epoch, seed=None, best=True, Type='val'):
        
        loss_list = []

        rmse_list = []
        rmse_key_result = {}

        for index, dataset_name in enumerate(self.args.dataset.split('*')):

            rmse_key_result[dataset_name] = {}

            if self.args.mask_strategy_random != 'none':
                for s in self.mask_list:
                    for m in self.mask_list[s]:
                        result, mae, loss_test, accuracy_metrics = self.Sample(test_data, epoch, mask_ratio=m, mask_strategy = s, seed=seed, dataset = dataset_name, index=index, Type=Type)
                        rmse_list.append(result)
                        loss_list.append(loss_test)
                        if s not in rmse_key_result[dataset_name]:
                            rmse_key_result[dataset_name][s] = {}
                        rmse_key_result[dataset_name][s][m] = result
                        
                        if Type == 'val':
                            self.writer.add_scalar('Evaluation/{}-{}-{}'.format(dataset_name.split('_C')[0], s, m), result, epoch)
                        elif Type == 'test':
                            self.writer.add_scalar('Test_RMSE/{}-{}-{}'.format(dataset_name.split('_C')[0], s, m), result, epoch)
                            self.writer.add_scalar('Test_MAE/MAE-{}-{}-{}'.format(dataset_name.split('_C')[0], s, m), mae, epoch)

            else:
                s = self.args.mask_strategy
                m = self.args.mask_ratio
                result, mae,  loss_test, accuracy_metrics = self.Sample(test_data, epoch, mask_ratio=m, mask_strategy = s, seed=seed, dataset = dataset_name, index=index, Type=Type)
                rmse_list.append(result)
                loss_list.append(loss_test)
                if s not in rmse_key_result[dataset_name]:
                    rmse_key_result[dataset_name][s] = {}
                rmse_key_result[dataset_name][s][m] = {'rmse':result, 'mae':mae}
                
                if Type == 'val':
                    self.writer.add_scalar('Evaluation/{}-{}-{}'.format(dataset_name.split('_C')[0], s, m), result, epoch)
                elif Type == 'test':
                    self.writer.add_scalar('Test_RMSE/Stage-{}-{}-{}-{}'.format(self.args.stage, dataset_name.split('_C')[0], s, m), result, epoch)
                    self.writer.add_scalar('Test_MAE/Stage-MAE-{}-{}-{}-{}'.format(self.args.stage, dataset_name.split('_C')[0], s, m), mae, epoch)
                    
        loss_test = np.mean(loss_list)

        if best:
            is_break = self.best_model_save(epoch, loss_test, rmse_key_result)
            return is_break

        else:
            return loss_test, rmse_key_result

    def best_model_save(self, step, rmse, rmse_key_result):
        if rmse < self.best_rmse:
            self.early_stop = 0
            torch.save(self.model.state_dict(), self.args.model_path+'model_save/model_best_stage_{}.pkl'.format(self.args.stage))
            torch.save(self.model.state_dict(), self.args.model_path+'model_save/model_best.pkl')
            self.best_rmse = rmse
            self.writer.add_scalar('Evaluation/RMSE_best', self.best_rmse, step)
            print('\nRMSE_best:{}\n'.format(self.best_rmse))
            print(str(rmse_key_result)+'\n')
            with open(self.args.model_path+'result.txt', 'w') as f:
                f.write('stage:{}, epoch:{}, best rmse: {}\n'.format(self.args.stage, step, self.best_rmse))
                f.write(str(rmse_key_result)+'\n')
            with open(self.args.model_path+'result_all.txt', 'a') as f:
                f.write('stage:{}, epoch:{}, best rmse: {}\n'.format(self.args.stage, step, self.best_rmse))
                f.write(str(rmse_key_result)+'\n')
            return 'save'

        else:
            self.early_stop += 1
            print('\nRMSE:{}, RMSE_best:{}, early_stop:{}\n'.format(rmse, self.best_rmse, self.early_stop))
            with open(self.args.model_path+'result_all.txt', 'a') as f:
                f.write('RMSE:{}, not optimized, early_stop:{}\n'.format(rmse, self.early_stop))
            if self.early_stop >= self.args.early_stop:
                print('Early stop!')
                with open(self.args.model_path+'result.txt', 'a') as f:
                    f.write('Early stop!\n')
                with open(self.args.model_path+'result_all.txt', 'a') as f:
                    f.write('Early stop!\n')
                    exit()

    def mask_select(self):
        if self.args.mask_strategy_random == 'none': #'none' or 'batch'
            mask_strategy = self.args.mask_strategy
            mask_ratio = self.args.mask_ratio
        else:
            mask_strategy=random.choice(['random','temporal','tube','block'])
            mask_ratio=random.choice(self.mask_list[mask_strategy])

        return mask_strategy, mask_ratio

    def run_loop(self):
        step = 0

        if self.args.mode == 'testing':
            print('Testing')
            self.Evaluation(self.val_data, 0, best=True, Type='test')
            exit()
        
        self.Evaluation(self.val_data, 0, best=True, Type='val')
        
        for epoch in range(self.args.total_epoches):
            print('Training')

            self.step = epoch
            
            loss_all, num_all, loss_real_all, num_all2 = 0.0, 0.0,0.0, 0.0
            start = time.time()
            for name, batch in self.data:
                mask_strategy, mask_ratio = self.mask_select()
                loss, num, loss_real, num2  = self.run_step(batch, step, mask_ratio=mask_ratio, mask_strategy = mask_strategy,index=0, name = name)
                step += 1
                loss_all += loss * num
                loss_real_all += loss_real * num
                num_all += num
                num_all2 += num2
            
            end = time.time()
            print('training time:{} min'.format(round((end-start)/60.0,2)))
            print('epoch:{}, training loss:{}, training rmse:{}'.format(epoch, loss_all / num_all,np.sqrt(loss_real_all / num_all)))

            if epoch % self.log_interval == 0 and epoch > 0 or epoch == 10 or epoch == self.args.total_epoches-1:
                print('Evaluation')
                eval_result = self.Evaluation(self.val_data, epoch, best=True, Type='val')

                if eval_result == 'save':
                    print('test evaluate!')
                    rmse_test, rmse_key_test = self.Evaluation(self.test_data, epoch, best=False, Type='test')
                    print('stage:{}, epoch:{}, test rmse: {}\n'.format(self.args.stage, epoch, rmse_test))
                    print(str(rmse_key_test)+'\n')
                    with open(self.args.model_path+'result.txt', 'a') as f:
                        f.write('stage:{}, epoch:{}, test rmse: {}\n'.format(self.args.stage, epoch, rmse_test))
                        f.write(str(rmse_key_test)+'\n')
                    with open(self.args.model_path+'result_all.txt', 'a') as f:
                        f.write('stage:{}, epoch:{}, test rmse: {}\n'.format(self.args.stage, epoch, rmse_test))
                        f.write(str(rmse_key_test)+'\n')

    def model_forward(self, batch, model, mask_ratio, mask_strategy, seed=None, data=None, mode='backward'):

        batch = [i.to(self.device) for i in batch]

        loss, loss2, pred, target, mask, ids_restore, input_size = self.model(
                batch,
                mask_ratio=mask_ratio,
                mask_strategy = mask_strategy, 
                seed = seed, 
                data = data,
                mode = mode, 
            )
        return loss, loss2, pred, target, mask, ids_restore, input_size 

    def forward_backward(self, batch, step, mask_ratio, mask_strategy, index, name=None):

        loss , _, pred, target, mask, _, _ = self.model_forward(batch, self.model, mask_ratio, mask_strategy, data=name, mode='backward')

        pred_mask = pred.squeeze(dim=2)[mask==1]
        target_mask = target.squeeze(dim=2)[mask==1]
        loss_real = mean_squared_error(self.args.scaler[name].inverse_transform(pred_mask.reshape(-1,1).detach().cpu().numpy()), self.args.scaler[name].inverse_transform(target_mask.reshape(-1,1).detach().cpu().numpy()), squared=True)
    
        loss.backward()

        self.writer.add_scalar('Training/Loss_step', np.sqrt(loss_real), step)
        return loss.item(), mask.sum().item(), loss_real, (1-mask).sum().item()

    def _anneal_lr(self):
        if self.step < self.warmup_steps:
            lr = self.lr * (self.step+1) / self.warmup_steps
        elif self.step < self.lr_anneal_steps:
            lr = self.min_lr + (self.lr - self.min_lr) * 0.5 * (
                1.0
                + math.cos(
                    math.pi
                    * (self.step - self.warmup_steps)
                    / (self.lr_anneal_steps - self.warmup_steps)
                )
            )
        else:
            lr = self.min_lr
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr
        self.writer.add_scalar('Training/LR', lr, self.step)
        return lr

