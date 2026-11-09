# Member 1 partition: block_size 64, n_layer 4

The only artifact the group received from Member 1 is the 12-page report in this folder.
It documents 32 completed runs, the code-analysis answers, the measurement protocol
(custom exact-update runner, fixed evaluation batches, full-validation NLL on all 111,539
held-out characters, NVIDIA T1000, PyTorch 2.6.0, float32) and the results table.

`results_transcribed_from_report.csv` is that results table (report Section 05, page 5)
copied into CSV form so the partition's numbers sit beside the other two partitions'
`results.csv` files. `val_loss_full_validation_nll` is a different metric from the sampled
validation loss in `member2` and `member3` and should not be ranked against it.

The Colab notebook, per-run logs, `results/summary.csv`, `paired_effects.json` and the
sample JSON files that the report cites were not shared with the group and are held by
Member 1.
