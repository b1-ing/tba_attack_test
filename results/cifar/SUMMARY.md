# CIFAR-10 reference FilterFL runs (seed 1, 2000 epochs, 100 clients, 10 malicious)

| Config | Defence | Clean acc | ASR | Attackers caught / appearances | Benign flagged |
|---|---|---|---|---|---|
| shipped (alpha=1, poison_rate=0.5), job 23199459 | None | 86.33% | 77.00% | n/a | n/a |
| shipped | ours | 85.76% | 1.24% | 2028 / 2028 | 29 |
| alpha=0.5, poison_rate=0.3, job 23200242 | None | 85.68% | 97.17% | n/a | n/a |
| alpha=0.5, poison_rate=0.3 | ours | 85.51% | 94.23% | 831 / 2028 | 583 |

ASR = "Global_Test_Poison:True" accuracy: triggered non-class-0 test images (9000) classified as class 0.
Per run: `log.txt.gz` (reference log), `params.yaml` (as run), `slurm.out.gz` (stdout), params file used.
