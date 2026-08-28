## 1. Environment Setup

Create a Conda environment with Python 3.10 and install the cu126 dependencies:

```powershell
conda create -n diffusion python=3.10 -y
conda activate diffusion
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Train Diffusion

Default UNet, single GPU:

```powershell
python -m new_diffusion.train --dataset_root dataset/deepfashion --resolution 512 --batch_size 6 --num_workers 8 --latent_channels 4 --base_channels 256 --epochs 100 --sample_source_guidance_scale 3.0 --sample_pose_guidance_scale 3.0 --sd_vae_name_or_path stabilityai/sd-vae-ft-mse
```

Default UNet, multi GPU:

```powershell
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m new_diffusion.train --dataset_root dataset/deepfashion --resolution 512 --batch_size 3 --num_workers 8 --latent_channels 4 --base_channels 256 --epochs 100 --sample_source_guidance_scale 3.0 --sample_pose_guidance_scale 3.0 --sd_vae_name_or_path stabilityai/sd-vae-ft-mse
```


## 3. Single Image Inference

Use this when you have one source image and one target densepose png.

```powershell
python -m new_diffusion.predict --checkpoint checkpoints/new_diffusion/last.pt --source test.jpg --pose dataset/deepfashion/densepose/WOMEN-Tees_Tanks-id_00000142-01_1_front_densepose.png --output outputs/new_diffusion.png --resolution 512 --sampler ddim --steps 50 --source_guidance_scale 3.0 --pose_guidance_scale 3.0 --device auto
```

Use a converted BF16 checkpoint for lower inference memory and Tensor Core acceleration on a supported CUDA GPU:

```powershell
python -m new_diffusion.predict --checkpoint checkpoints/new_diffusion/last_bf16.pt --dtype bf16 --source test.jpg --pose dataset/deepfashion/densepose/WOMEN-Tees_Tanks-id_00000142-01_1_front_densepose.png --output outputs/new_diffusion_bf16.png --resolution 512 --sampler ddim --steps 50 --source_guidance_scale 3.0 --pose_guidance_scale 3.0 --device auto
```

`--dtype` accepts `fp32` (default), `fp16`, and `bf16`. FP16/BF16 inference requires CUDA; BF16 additionally requires a BF16-capable GPU. The UNet and VAE weights are kept in the selected dtype while diffusion scheduler coefficients remain FP32.

## 4. Batch Inference from a Pairs File

Use this when you want to automatically expand every `(target, source)` pair from a pairs file such as `test_pairs.txt` or `train_pairs.txt`. The target pose is loaded automatically from `dataset/deepfashion/densepose`.

```powershell
python -m new_diffusion.predict --checkpoint checkpoints/new_diffusion/last.pt --dataset_root dataset/deepfashion --pairs_file test_pairs.txt --output_dir outputs/new_diffusion_test_pairs --resolution 512 --sampler ddim --steps 50 --source_guidance_scale 3.0 --pose_guidance_scale 3.0 --device auto --batch_size 16

# Resume an interrupted run: completed PNG files are verified and skipped.
python -m new_diffusion.predict --checkpoint checkpoints/new_diffusion/last.pt --dataset_root dataset/deepfashion --pairs_file test_pairs.txt --output_dir outputs/new_diffusion_test_pairs --resolution 512 --sampler ddim --steps 50 --source_guidance_scale 3.0 --pose_guidance_scale 3.0 --device auto --batch_size 16 --resume
```

For faster batch inference on 2 GPUs:

```powershell
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m new_diffusion.predict --checkpoint checkpoints/new_diffusion/last.pt --dataset_root dataset/deepfashion --pairs_file test_pairs.txt --output_dir outputs/new_diffusion_test_pairs --resolution 512 --sampler ddim --steps 50 --source_guidance_scale 3.0 --pose_guidance_scale 3.0 --device auto --batch_size 16
```

Example for generating diffusion outputs on the training set:

```powershell
python -m new_diffusion.predict --checkpoint checkpoints/new_diffusion/last.pt --dataset_root dataset/deepfashion --pairs_file train_pairs.txt --output_dir outputs/new_diffusion_train_pairs --resolution 512 --sampler ddim --steps 50 --source_guidance_scale 3.0 --pose_guidance_scale 3.0 --device auto --batch_size 16
```

## 5. Evaluate Metrics from `test_pairs.txt`

Prepare training-set real images for FID. Use the FID-real directory that matches the evaluation resolution:

```powershell
python dataset/prepare_fid_real.py --dataset_root dataset/deepfashion --output_dir dataset/deepfashion/fid_real_256x176 --resolution 256 176
python dataset/prepare_fid_real.py --dataset_root dataset/deepfashion --output_dir dataset/deepfashion/fid_real_512x352 --resolution 512 352
```

Generate paired GT folders for evaluation. This creates both `outputs/gt/256` and `outputs/gt/512`:

```powershell
python dataset/prepare_gt.py --dataset_root dataset/deepfashion --pairs_file test_pairs.txt --output_dir outputs/gt
```

Evaluate `256x176` predictions:

```powershell
python evaluate.py --gt_path outputs/gt --img_path outputs/new_diffusion_test_pairs --training_path dataset/deepfashion --fid_real_path dataset/deepfashion/fid_real_256x176 --resolution 256
```

Evaluate `512x352` predictions:

```powershell
python evaluate.py --gt_path outputs/gt --img_path outputs/new_diffusion_test_pairs --training_path dataset/deepfashion --fid_real_path dataset/deepfashion/fid_real_512x352 --resolution 512
```

## 6. Train Refiner

Train the refiner with diffusion training-set outputs as input and GT as supervision. Validation is required: each epoch the script loads diffusion outputs from the test set and computes `val_l1` against test-set GT.

The training objective is `L1 + lpips_weight * LPIPS` when `--lpips_weight` is greater than zero. Gradient/Sobel loss is not used.

`--resolution` controls both training and validation forward passes. With `256`, inference uses `256x256`, while PSNR/SSIM/LPIPS/FID use `256x176` and `fid_real_256x176`. With `512`, inference uses `512x512`, while metrics use `512x352` and `fid_real_512x352`.


```powershell
python -m Refiner.train_refiner --dataset_root dataset/deepfashion --pred_dir outputs/new_diffusion_train_pairs --pairs_file train_pairs.txt --val_pred_dir outputs/new_diffusion_test_pairs --val_pairs_file test_pairs.txt --resolution 512 --batch_size 4 --epochs 10 --lr 2e-4 --model_dim 24 --checkpoint_dir checkpoints/refiner --lpips_weight 0.2
```

Train with two GPUs：

```powershell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 -m Refiner.train_refiner --dataset_root dataset/deepfashion --pred_dir outputs/new_diffusion_train_pairs --pairs_file train_pairs.txt --val_pred_dir outputs/new_diffusion_test_pairs --val_pairs_file test_pairs.txt --resolution 512 --batch_size 2 --epochs 10 --lr 2e-4 --model_dim 24 --checkpoint_dir checkpoints/refiner --lpips_weight 0.2
```

## 7. Refiner Inference

Run the refiner on a single diffusion result:

```powershell
python -m Refiner.predict_refiner --checkpoint checkpoints/refiner/last.pt --input outputs/new_diffusion_train_pairs/example.png --output outputs/refiner_example.png --resolution 512
```

Run the refiner on a whole directory of diffusion results:

```powershell
python -m Refiner.predict_refiner --checkpoint checkpoints/refiner/last.pt --input_dir outputs/new_diffusion_test_pairs --output_dir outputs/new_diffusion_test_pairs_refined --resolution 512 --batch_size 16 --num_workers 8
```

## 8. Demo: Diffusion + Refiner

Run diffusion and then Refiner in one command. Refiner post-processing is enabled by default, and the command saves both the diffusion result and refined result.

```powershell
python demo.py --source test.jpg --pose dataset/deepfashion/densepose/WOMEN-Tees_Tanks-id_00000142-01_1_front_densepose.png --diffusion_checkpoint checkpoints/new_diffusion/last.pt --refiner_checkpoint checkpoints/refiner/last.pt --resolution 512 --sampler ddim --steps 50 --source_guidance_scale 3.0 --pose_guidance_scale 3.0 --device auto
```

Diffusion always uses `512x512` source and DensePose inputs. `--resolution` only controls Refiner inference: `512` uses `512x512`, while `256` resizes the diffusion output to `256x256` before refinement.

Use `--no-refine` to disable Refiner post-processing and save only the diffusion output. `--refine` (or `-refine`) explicitly enables the default behavior.

## 9. Masked Appearance Editing

`demo_edit.py` transfers the clothing appearance from `--source` into the white region of `--mask`, while retaining the identity and background of `--reference`. The 3-channel DensePose and mask must be spatially aligned with the reference image. In the mask, white pixels are edited and black pixels are preserved.

Run diffusion-only appearance editing:

```powershell
python demo_edit.py --source donor.jpg --reference target.jpg --pose target_densepose.png --mask target_upper_mask.png --checkpoint checkpoints/new_diffusion/last.pt --output outputs/appearance_edit.png --comparison-output outputs/appearance_edit_comparison.png --resolution 512 --sampler ddim --steps 50 --source-guidance-scale 3.0 --pose-guidance-scale 3.0 --seed 42 --device auto
```

Add `--refine` to post-process the diffusion result with Refiner:

```powershell
python demo_edit.py --source donor.jpg --reference target.jpg --pose target_densepose.png --mask target_upper_mask.png --checkpoint checkpoints/new_diffusion/last.pt --refine --refiner-checkpoint checkpoints/refiner/last.pt --output outputs/appearance_edit_refined.png --comparison-output outputs/appearance_edit_refined_comparison.png --resolution 512 --sampler ddim --steps 50 --source-guidance-scale 3.0 --pose-guidance-scale 3.0 --seed 42 --device auto
```

The script also saves a comparison image by default. Use `--no-comparison` to disable it. Use `--validate-only` to validate the source, reference, DensePose, and mask without loading the model.

## 10. Appearance Style Interpolation

`demo_interpolate.py` interpolates the model's multi-scale reference features between two appearance images. All frames use the same initial noise so that the visual change mainly comes from the appearance interpolation. Refiner post-processing is disabled by default.

Generate 11 transition frames with one fixed 3-channel DensePose:

```powershell
python demo_interpolate.py --style1 style_1.jpg --style2 style_2.jpg --pose target_densepose.png --frames 11 --interpolation linear --checkpoint checkpoints/new_diffusion/last.pt --output-dir outputs/style_interpolation --strip-output outputs/style_interpolation.png --batch-size 1 --resolution 512 --sampler ddim --steps 50 --source-guidance-scale 3.0 --pose-guidance-scale 3.0 --seed 42 --device auto
```

Use an ordered directory of DensePose images to change pose during the appearance transition:

```powershell
python demo_interpolate.py --style1 style_1.jpg --style2 style_2.jpg --pose-dir densepose_sequence --frames 11 --interpolation linear --checkpoint checkpoints/new_diffusion/last.pt --output-dir outputs/style_interpolation_pose_sequence --strip-output outputs/style_interpolation_pose_sequence.png --batch-size 1 --resolution 512 --sampler ddim --steps 50 --source-guidance-scale 3.0 --pose-guidance-scale 3.0 --seed 42 --device auto
```

Add `--refine` to post-process every generated frame with Refiner:

```powershell
python demo_interpolate.py --style1 style_1.jpg --style2 style_2.jpg --pose target_densepose.png --frames 11 --interpolation linear --checkpoint checkpoints/new_diffusion/last.pt --refine --refiner-checkpoint checkpoints/refiner/last.pt --output-dir outputs/style_interpolation_refined --strip-output outputs/style_interpolation_refined.png --batch-size 1 --resolution 512 --sampler ddim --steps 50 --source-guidance-scale 3.0 --pose-guidance-scale 3.0 --seed 42 --device auto
```

Available interpolation modes are `linear` and `slerp`. Individual frames and `manifest.json` are always saved under `--output-dir`; a horizontal summary image is also saved unless `--no-strip` is specified. Use `--validate-only` to validate the appearance and DensePose inputs without loading the model.
