@echo off
REM Example script to run federated learning with SAM+LoRA on Fed-KITS
REM This script demonstrates the complete workflow

echo ==========================================
echo Federated Learning with SAM+LoRA
echo ==========================================
echo.

REM Configuration
set SAM_CHECKPOINT=sam_vit_b_01ec64.pth
set RAW_DATA_DIR=path\to\kits_raw
set PREPROCESSED_DIR=.\data\kits_preprocessed
set NUM_CLIENTS=5
set NUM_ROUNDS=10

REM Step 1: Check SAM checkpoint
if not exist "%SAM_CHECKPOINT%" (
    echo Step 1: SAM checkpoint not found
    echo ----------------------------------------
    echo Please download SAM checkpoint:
    echo   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    echo   or use curl:
    echo   curl -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    echo.
    exit /b 1
) else (
    echo SAM checkpoint found
    echo.
)

REM Step 2: Preprocess dataset (if not already done)
if not exist "%PREPROCESSED_DIR%" (
    echo Step 2: Preprocessing KiTS dataset...
    echo ----------------------------------------
    python preprocess_data.py ^
        --raw_root "%RAW_DATA_DIR%" ^
        --output_root "%PREPROCESSED_DIR%" ^
        --num_clients %NUM_CLIENTS% ^
        --train_frac 0.8 ^
        --seed 13
    echo.
) else (
    echo Preprocessed data already exists
    echo.
)

REM Step 3: Run federated learning
echo Step 3: Running federated learning...
echo ----------------------------------------
python flower_sim.py ^
    --sam_checkpoint "%SAM_CHECKPOINT%" ^
    --data_root "%PREPROCESSED_DIR%" ^
    --model_type vit_b ^
    --num_clients %NUM_CLIENTS% ^
    --num_rounds %NUM_ROUNDS% ^
    --local_epochs 1 ^
    --batch_size 2 ^
    --learning_rate 1e-4 ^
    --lora_rank 4 ^
    --lora_alpha 16.0 ^
    --output_dir .\fl_output

echo.
echo ==========================================
echo Federated learning completed!
echo ==========================================

pause
