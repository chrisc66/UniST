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


    def extract_predictions(self, pred, target, mask, input_size, dataset_name, plot=False):
        """
        Extract both masked and unmasked predictions
        Args:
            pred: model predictions [N, L, patch_size**2 * t_patch_size * C_model]
            target: ground truth [N, L, patch_size**2 * t_patch_size * C_model]
            mask: masking tensor [N, L]
            input_size: tuple of (T_patch, H_patch, W_patch) for reshaping, where T_patch is T // t_patch_size etc.
            dataset_name: name of dataset for scaler
        """
        N = pred.shape[0]
        T_patch, H_patch, W_patch = input_size # These are patch counts, not original dimensions
        p = self.model.args.patch_size
        u = self.model.args.t_patch_size
        C_model = self.model.in_chans # Number of channels from the model

        # Original dimensions before patching
        T_orig = T_patch * u
        H_orig = H_patch * p
        W_orig = W_patch * p

        # Update patch_info with correct original and patch dimensions
        # N, T_orig, H_orig, W_orig, p_spatial, u_temporal, t_patch_count, h_patch_count, w_patch_count
        self.model.patch_info = (N, T_orig, H_orig, W_orig, p, u, T_patch, H_patch, W_patch)

        # Masked data (still in patched domain, normalized)
        # pred/target are [N, L, D_patch_full], mask is [N, L]
        mask_expanded = mask.unsqueeze(-1).expand_as(pred) # Expand mask to match pred/target dimensions
        pred_masked = pred[mask_expanded == 1] # Flattened masked predictions
        target_masked = target[mask_expanded == 1] # Flattened masked targets

        # Unpatch data to [N, C, T_orig, H_orig, W_orig]
        # The unpatchify method in the model should handle the C_model dimension correctly.
        pred_unpatched = self.model.unpatchify(pred)      # Shape: (N, C_model, T_orig, H_orig, W_orig)
        target_unpatched = self.model.unpatchify(target)  # Shape: (N, C_model, T_orig, H_orig, W_orig)

        # Un-normalize data to original scale using the per-channel scaler
        # The scaler expects torch.Tensor of shape [N, C, T, H, W]
        pred_unnormalized_tensor = self.args.scaler[dataset_name].inverse_transform(pred_unpatched.detach().cpu())
        target_unnormalized_tensor = self.args.scaler[dataset_name].inverse_transform(target_unpatched.detach().cpu())

        pred_unnormalized = pred_unnormalized_tensor.numpy()
        target_unnormalized = target_unnormalized_tensor.numpy()
        
        # Convert normalized unpatched data to numpy for consistency in return dict
        pred_numpy_normalized_unpatched = pred_unpatched.detach().cpu().numpy()
        target_numpy_normalized_unpatched = target_unpatched.detach().cpu().numpy()

        if plot:
            os.makedirs(f"{self.args.model_path}/plots", exist_ok=True)

            # 1. Masked Data (Plotting flattened masked values, might be very long)
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

            # 2 Patched Data (Plotting one sample, all patches, all channels combined in the last dim of 'pred')
            # pred is [N, L, D_patch_full]. We can take one sample.
            sample_idx_plot = 0
            plt.figure(figsize=(20, 10))

            plt.subplot(131)
            plt.imshow(target[sample_idx_plot].detach().cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
            plt.title(f'Target Patched (Sample {sample_idx_plot}, Shape: {target[sample_idx_plot].shape})')
            plt.colorbar()

            plt.subplot(132)
            plt.imshow(pred[sample_idx_plot].detach().cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
            plt.title(f'Prediction Patched (Sample {sample_idx_plot}, Shape: {pred[sample_idx_plot].shape})')
            plt.colorbar()

            plt.subplot(133)
            plt.imshow(np.abs(target[sample_idx_plot].detach().cpu().numpy() - pred[sample_idx_plot].detach().cpu().numpy()), aspect='auto', origin='lower', cmap='magma')
            plt.title('Absolute Error Patched')
            plt.colorbar()

            plt.suptitle('Patched Visualization (Normalized, All Channels in Patch Dim)')
            plt.tight_layout()
            plt.savefig(f"{self.args.model_path}/plots/2_patched.png")
            plt.close()

            batch_idx_plot_list = range(1) # Index for selecting batch element for plotting
            ch_vis_idx = 0 # Channel index for visualization
            for batch_idx_plot in batch_idx_plot_list:
                # 3 Unpatched Data (Normalized)
                # Plot for each channel
                plt.figure(figsize=(min(T_orig*2, 50), 10) ) # Adjust figure size
                timesteps_to_plot = range(T_orig)
                for i, t_idx in enumerate(timesteps_to_plot):
                    if i >= 12 and T_orig > 12 : # Limit plots if too many timesteps
                        break
                    plt.subplot(3, min(len(timesteps_to_plot),12), i + 1)
                    plt.imshow(target_numpy_normalized_unpatched[batch_idx_plot, ch_vis_idx, t_idx], vmin=-1, vmax=1, cmap='viridis')
                    plt.gca().invert_xaxis()
                    plt.title(f'Target t={t_idx}, ch={ch_vis_idx}')
                    plt.colorbar()

                    plt.subplot(3, min(len(timesteps_to_plot),12), i + 1 + min(len(timesteps_to_plot),12))
                    plt.imshow(pred_numpy_normalized_unpatched[batch_idx_plot, ch_vis_idx, t_idx], vmin=-1, vmax=1, cmap='viridis')
                    plt.gca().invert_xaxis()
                    plt.title(f'Pred t={t_idx}, ch={ch_vis_idx}')
                    plt.colorbar()

                    plt.subplot(3, min(len(timesteps_to_plot),12), i + 1 + 2 * min(len(timesteps_to_plot),12))
                    plt.imshow(np.abs(pred_numpy_normalized_unpatched[batch_idx_plot, ch_vis_idx, t_idx] - \
                                    target_numpy_normalized_unpatched[batch_idx_plot, ch_vis_idx, t_idx]), vmin=0, vmax=1, cmap='magma')
                    plt.gca().invert_xaxis()
                    plt.title(f'Error t={t_idx}, ch={ch_vis_idx}')
                    plt.colorbar()
                plt.suptitle(f'Unpatched Normalized Visualization - Channel {ch_vis_idx}')
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.savefig(f"{self.args.model_path}/plots/3_unpatched_normalized_sample_{batch_idx_plot}_channel_{ch_vis_idx}.png")
                plt.close()

                # 4 Un-normalized Data (Original Scale)
                plt.figure(figsize=(min(T_orig*2, 50), 10))
                timesteps_to_plot = range(T_orig)
                # Determine dynamic range for plotting for this channel
                ch_min_val = min(target_unnormalized[batch_idx_plot, ch_vis_idx].min(), pred_unnormalized[batch_idx_plot, ch_vis_idx].min())
                ch_max_val = max(target_unnormalized[batch_idx_plot, ch_vis_idx].max(), pred_unnormalized[batch_idx_plot, ch_vis_idx].max())
                if ch_min_val == ch_max_val: ch_max_val += 1e-6 # Avoid same min/max for colorbar

                for i, t_idx in enumerate(timesteps_to_plot):
                    if i >= 12 and T_orig > 12 :
                        break
                    plt.subplot(3, min(len(timesteps_to_plot),12), i + 1)
                    plt.imshow(target_unnormalized[batch_idx_plot, ch_vis_idx, t_idx], vmin=ch_min_val, vmax=ch_max_val, cmap='viridis')
                    plt.gca().invert_xaxis()
                    plt.title(f'Target t={t_idx}, ch={ch_vis_idx}')
                    plt.colorbar()

                    plt.subplot(3, min(len(timesteps_to_plot),12), i + 1 + min(len(timesteps_to_plot),12))
                    plt.imshow(pred_unnormalized[batch_idx_plot, ch_vis_idx, t_idx], vmin=ch_min_val, vmax=ch_max_val, cmap='viridis')
                    plt.gca().invert_xaxis()
                    plt.title(f'Pred t={t_idx}, ch={ch_vis_idx}')
                    plt.colorbar()

                    plt.subplot(3, min(len(timesteps_to_plot),12), i + 1 + 2 * min(len(timesteps_to_plot),12))
                    plt.imshow(np.abs(pred_unnormalized[batch_idx_plot, ch_vis_idx, t_idx] - target_unnormalized[batch_idx_plot, ch_vis_idx, t_idx]), vmin=0, vmax=max(1e-6, (ch_max_val-ch_min_val)/2), cmap='magma')
                    plt.gca().invert_xaxis()
                    plt.title(f'Error t={t_idx}, ch={ch_vis_idx}')
                    plt.colorbar()
                plt.suptitle(f'Un-normalized Visualization (Original Scale) - Channel {ch_vis_idx}')
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.savefig(f"{self.args.model_path}/plots/4_unnormalized_sample_{batch_idx_plot}_channel_{ch_vis_idx}.png")
                plt.close()

                # 5. Compute accuracy masks for each threshold
                epsilon = 1 # Use consistent epsilon
                thresholds = [0.1, 0.3, 0.5, 0.7, 1.0, 3.0, 5.0]
                threshold_labels = [f'{int(thresh * 100)}%' if thresh <= 1.0 else f'{int(thresh)}00%' for thresh in thresholds]

                acc_per_timestep_channel = {label: [] for label in threshold_labels}
                
                # Create grid plot for current channel
                # Figure size is very large, kept as is from original
                fig_grid, axes_grid = plt.subplots(T_orig - T_orig // 2, len(thresholds), figsize=(160, 60), squeeze=False)
                
                for t_idx_orig_offset, t_idx_orig in enumerate(range(T_orig // 2, T_orig)):
                    for i_thresh, (thresh, label) in enumerate(zip(thresholds, threshold_labels)):
                        ax = axes_grid[t_idx_orig_offset, i_thresh]
                        mask_within = np.abs(pred_unnormalized[batch_idx_plot, ch_vis_idx, t_idx_orig] - target_unnormalized[batch_idx_plot, ch_vis_idx, t_idx_orig]) <= \
                                        (np.abs(target_unnormalized[batch_idx_plot, ch_vis_idx, t_idx_orig]) * thresh + epsilon)
                        
                        ax.imshow(mask_within, cmap='Greens', vmin=0, vmax=1)
                        for y_ax in range(H_orig):
                            for x_ax in range(W_orig):
                                pred_val = pred_unnormalized[batch_idx_plot, ch_vis_idx, t_idx_orig, y_ax, x_ax]
                                target_val = target_unnormalized[batch_idx_plot, ch_vis_idx, t_idx_orig, y_ax, x_ax]
                                ax.text(x_ax, y_ax, f'{pred_val:.1f}\n{target_val:.1f}',
                                    ha='center', va='center', fontsize=6,
                                    color='black' if mask_within[y_ax, x_ax] else 'gray',
                                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1'))
                        
                        acc = np.sum(mask_within) / (H_orig * W_orig) if (H_orig * W_orig) > 0 else 0
                        acc_per_timestep_channel[label].append(acc)
                        ax.set_title(f'T={t_idx_orig} {label}, Acc={acc * 100:.2f}%', fontsize=8)
                        ax.axis('off')
                
                fig_grid.suptitle(f'Prediction Acceptance Rate Grid - Channel {ch_vis_idx} (Sample {batch_idx_plot})')
                plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to make space for suptitle
                plt.savefig(f"{self.args.model_path}/plots/5_acceptance_grid_sample_{batch_idx_plot}_channel_{ch_vis_idx}.png")
                plt.close(fig_grid)

                # Plot acceptance within X percent for each timestep for current channel
                plt.figure(figsize=(16, 6))
                for label in threshold_labels:
                    plt.plot(range(T_orig // 2, T_orig), acc_per_timestep_channel[label], marker='o', label=f'Within {label}')
                plt.xlabel('Timestep')
                plt.ylabel('Acceptance Rate')
                plt.title(f'Prediction Acceptance Rate vs. Timestep - Channel {ch_vis_idx} (Sample {batch_idx_plot})')
                plt.ylim(0, 1)
                plt.legend()
                plt.grid()
                plt.tight_layout()
                plt.savefig(f"{self.args.model_path}/plots/5_acceptance_vs_timestep_sample_{batch_idx_plot}_channel_{ch_vis_idx}.png")
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
                json.dump(pred.tolist(), pred_file) # Changed pred_patched to pred
            with open(f"{self.args.model_path}/plots/target_patched.json", "w") as target_file:
                json.dump(target.tolist(), target_file) # Changed target_patched to target
            with open(f"{self.args.model_path}/plots/pred_unpatched.json", "w") as pred_file:
                json.dump(pred_unpatched.tolist(), pred_file)
            with open(f"{self.args.model_path}/plots/target_unpatched.json", "w") as target_file:
                json.dump(target_unpatched.tolist(), target_file)
            with open(f"{self.args.model_path}/plots/pred_unnormalized.json", "w") as pred_file:
                json.dump(pred_unnormalized.tolist(), pred_file)
            with open(f"{self.args.model_path}/plots/target_unnormalized.json", "w") as target_file:
                json.dump(target_unnormalized.tolist(), target_file)

        return { # Return unpatched, unnormalized numpy arrays
            'masked': {
                'predictions': pred_masked.detach().cpu().numpy(), # still normalized, patched, masked
                'targets': target_masked.detach().cpu().numpy()   # still normalized, patched, masked
            },
            'full_unpatched': {
                'normalized': {
                    'predictions': pred_numpy_normalized_unpatched, # [N, C, T, H, W]
                    'targets': target_numpy_normalized_unpatched    # [N, C, T, H, W]
                },
                'original_scale': {
                    'predictions': pred_unnormalized, # [N, C, T, H, W]
                    'targets': target_unnormalized    # [N, C, T, H, W]
                }
            },
            'spatial_shape': (N, T_orig, H_orig, W_orig), # Original spatial-temporal shape
            'mask': mask.detach().cpu().numpy() # [N, L_patch]
        }


    def Sample(self, test_data, step, mask_ratio, mask_strategy, seed=None, dataset='', index=0, Type='val'):
        print(f"Sample, Type {Type}, dataset {dataset}, mask_strategy {mask_strategy}, mask_ratio {mask_ratio}, index {index}")
        
        print("\n=== Input Data Shape ===")
        print(f"Expected input shape: [batch, channels, time, height, width]")
        for batch in test_data[index]:
            print(f"Actual input shape: {[t.shape for t in batch]}")
            break

        self.model.eval()
        thresholds = [0.1, 0.3, 0.5, 0.7, 1.0, 3.0, 5.0]
        accuracy_metrics = {}
        epsilon = 1
        rmse, mae, loss_test, num = 0, 0, 0, 0

        with torch.no_grad():
            for idx, batch in enumerate(test_data[index]):
                # Forward pass
                loss_masked_norm, _, pred_patched, target_patched, mask, _, input_size = self.model_forward(
                    batch, self.model, mask_ratio, mask_strategy, seed=seed, data=dataset, mode='forward'
                )

                # Call extract_predictions
                extracted_data = self.extract_predictions(pred_patched, target_patched, mask, input_size, dataset)

                # Use only channel C=0 for accuracy metrics
                pred_c0 = extracted_data['full_unpatched']['original_scale']['predictions'][:, 0, :, :, :]
                target_c0 = extracted_data['full_unpatched']['original_scale']['targets'][:, 0, :, :, :]

                # Filter timesteps where T > his_len
                if pred_c0.shape[1] > self.args.his_len:
                    print(f"size pred_c0: {pred_c0.shape}, target_c0: {target_c0.shape}")
                    pred_c0 = pred_c0[:, self.args.his_len:, :, :]
                    target_c0 = target_c0[:, self.args.his_len:, :, :]
                else:
                    print(f"Calculating accuracy for all timesteps, shape: {pred_c0.shape}, T: {pred_c0.shape[1]}, his_len: {self.args.his_len}")

                # Calculate RMSE and MAE
                pred_np = pred_c0.flatten()
                target_np = target_c0.flatten()
                rmse += np.sqrt(mean_squared_error(pred_np, target_np))
                mae += mean_absolute_error(pred_np, target_np)

                # Calculate accuracy
                acc_within = {thresh: 0 for thresh in thresholds}
                acc_total = pred_np.size
                for thresh in thresholds:
                    mask_within = np.abs(pred_np - target_np) <= (np.abs(target_np) * thresh + epsilon)
                    acc_within[thresh] += np.sum(mask_within)
                accuracy_metrics = {thresh: acc_within[thresh] / acc_total for thresh in thresholds}

                # Accumulate loss and num
                loss_test += loss_masked_norm.item()
                num += mask.sum().item()

        # Average metrics over all batches
        rmse /= len(test_data[index])
        mae /= len(test_data[index])
        loss_test /= len(test_data[index])

        print(f"\n--- Event Prediction Acceptance Rate ---")
        for label, acc in accuracy_metrics.items():
            percent = float(label.strip('%')) if isinstance(label, str) and '%' in label else float(label) * 100
            print(f"Threshold {percent:.0f}%: Acceptance Rate = {acc * 100:.2f}%")
        print(f"--- RMSE and MAE ---")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"Loss: {loss_test:.4f}")
        print(f"Num: {num}")

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

        # loss is the mean squared error on normalized, masked patches from model's forward_loss
        loss , _, _pred, _target, mask, _, _ = self.model_forward(batch, self.model, mask_ratio, mask_strategy, data=name, mode='backward')
    
        # For logging 'loss_real' (which is used for RMSE logging during training),
        # we use the 'loss' itself, which is the mean squared error on normalized masked patches.
        # Detailed unnormalized metrics are computed during Sample/Evaluation phases.
        loss_real = loss.item() 
    
        loss.backward()

        # Log normalized RMSE for this training step
        self.writer.add_scalar('Training/Loss_step', np.sqrt(loss_real), step)
        
        # loss.item() is mean MSE for masked patches
        # mask.sum().item() is the number of masked patches
        # loss_real is also mean MSE for masked patches
        # (1-mask).sum().item() is the number of unmasked/visible patches
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

