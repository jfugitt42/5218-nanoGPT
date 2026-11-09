# Group 7 - Assignment 11: nanoGPT hyperparameter experiments

1. Sepideh Salimi
2. Mohammed Abutayyem
3. Jonathan Fugitt

CSCE 5218 Deep Learning, University of North Texas, Fall 2026.

The assignment splits 2^7 = 128 hyperparameter combinations of Karpathy's nanoGPT on the
character-level Tiny Shakespeare dataset into four partitions of 32 runs, one per group
member. Group 7 has three members, so three partitions were run. Every member keeps
block_size and n_layer fixed and runs all combinations of n_head (4, 8), n_embd
(128, 256), batch_size (8, 16), max_iters (1000, 2000) and dropout (0.1, 0.2).

| Partition | block_size | n_layer | Member | Runs | Folder |
| --- | --- | --- | --- | --- | --- |
| 1 | 64 | 4 | Member 1 | 32 / 32 | member1_block64_layer4/ (report and its results table; notebook and logs held by Member 1) |
| 2 | 64 | 6 | Member 2 (Jonathan Fugitt) | 32 / 32 | member2_block64_layer6/ |
| 3 | 128 | 4 | Member 3 | 32 / 32 | member3_block128_layer4/ |

The fourth partition on the assignment slide, (128, 6), had no member in this three-person
group and was not run. 96 configurations were trained, all from scratch; the
instructor-supplied checkpoint (ckpt-1.pt) was not used to initialize any run.

## Reports

- reports/nanoGPT_Group1_Report_Member1_block64_layer4.pdf - Member 1's 12-page report for
  the (64, 4) partition, including the code-analysis answers for model.py, train.py and
  sample.py.
- reports/Group7_nanoGPT_Combined_Report_PartI_PartII.pdf - Part I (Member 1's report) and
  Part II (Member 3's (128, 4) partition) with a joint interpretation.
- reports/Member2_Report_jfugitt_block64_layer6.pdf - Member 2's (64, 6) partition.

## What each member folder contains

- *_Colab.ipynb - the Google Colab notebook that ran the 32 experiments, with its outputs.
- config/train_shakespeare_char.py - the nanoGPT config file every run was launched with.
- train.py, model.py, sample.py - the nanoGPT source as it was run: commit
  3adf61e154c3fe3fca428ad6bc3818b27a3b8291 of https://github.com/karpathy/nanoGPT with
  small edits (weights_only=False in sample.py's torch.load; a peak-GPU-memory print
  at the end of train.py; Member 3's copy also carries the assignment's auto-checkpoint
  snippet and a fixed meta.pkl path in sample.py).
- results/memberN/results.csv - one row per run: hyperparameters, final sampled train and
  validation loss, training time in seconds, parameter count, peak GPU memory in MB.
- results/memberN/gN_XX/training.log - full train.py output for the run, starting with
  the Overriding: lines that record the run's exact configuration.
- results/memberN/gN_XX/loss_history.csv - train and validation loss every 250 iterations.
- results/memberN/gN_XX/sample.txt - 200 characters generated from the prompt ROMEO:
  (temperature 0.8, top_k 40, seed 1337).
- results/memberN/loss_curves.png, comparisons.png - the figures used in the reports.

Model checkpoints (ckpt.pt) are not included; the notebooks regenerate them.

## Training settings shared by partitions 2 and 3

Every run was launched with the same command, plus the partition's --block_size and
--n_layer and the run's five varied hyperparameters:

```bash
python train.py config/train_shakespeare_char.py \
    --device=cuda --compile=False --dtype=float32 --init_from=scratch \
    --gradient_accumulation_steps=1 \
    --learning_rate=0.001 --warmup_iters=100 --lr_decay_iters=2000 --min_lr=0.0001 --beta2=0.99 \
    --eval_interval=250 --eval_iters=20 --log_interval=50 --always_save_checkpoint=True \
    --block_size=64 --n_layer=6 \
    --n_head=4 --n_embd=128 --batch_size=8 --max_iters=1000 --dropout=0.1 \
    --out_dir=results/member2/g2_01
```

train.py seeds PyTorch with 1337. Validation loss is the original training loop's estimate
over 20 sampled batches with dropout off. Member 1's report uses a full-validation NLL
instead, so its validation numbers are not directly comparable with partitions 2 and 3.

## Reproduce

Open a member's notebook in Google Colab on a GPU runtime (T4) and run it top to bottom. It
clones nanoGPT, runs data/shakespeare_char/prepare.py, applies the edits above, trains the
32 configurations, generates the samples, writes results.csv after every run and zips the
results folder. Finished runs are skipped if a cell is re-run. Partition 2 took 14.2 minutes
of training on a T4; partition 3 took 12.3 minutes.
