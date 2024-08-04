import os
import time
import subprocess

datasets = """Dirty/Walmart-Amazon
Structured/Amazon-Google
Structured/Walmart-Amazon
Textual/Abt-Buy""".split('\n')


ops = """del
swap
drop_col
swap""".split('\n')


lms = ['roberta', 'roberta', 'roberta', 'roberta']

run_count = 0

try:
    for dataset, op, lm in zip(datasets, ops, lms):
        batch_size, epochs = 32, 15

        for da in [True, False]:
            for dk in [True, False]:
                for run_id in range(3):
                    cmd = ["python", "train_ditto.py",
                           "--task", dataset,
                           "--logdir", "results_ditto/",
                           "--finetuning",
                           "--batch_size", str(batch_size),
                           "--lr", "3e-5",
                           "--fp16",
                           "--lm", lm,
                           "--n_epochs", str(epochs),
                           "--run_id", str(run_id)]
                    
                    if da:
                        cmd.extend(['--da', op])
                    if dk:
                        cmd.extend(['--dk', 'product'])

                    run_count += 1
                    print("=" * 64)
                    print(f"Starting run {run_count}/156 at ", time.strftime("%Y-%m-%d %H:%M:%S"))
                    print(f"Dataset: {dataset}")
                    print(f"Operation: {op}")
                    print(f"Language Model: {lm}")
                    print(f"Batch Size: {batch_size}")
                    print(f"Epochs: {epochs}")
                    print(f"Command: {' '.join(cmd)}")
                    print("=" * 64)
                    
                    process = subprocess.Popen(cmd)
                    try:
                        process.wait()
                    except KeyboardInterrupt:
                        process.terminate()
                        process.wait()
                        print("Script stopped by user.")
                        print("=" * 64)
                        print("Finished at ", time.strftime("%Y-%m-%d %H:%M:%S"))
                        print("=" * 64)
                        raise

                    print("=" * 64)
                    print("Finished at ", time.strftime("%Y-%m-%d %H:%M:%S"))
                    print("=" * 64)


except KeyboardInterrupt:
    print("Script stopped by user.")
    print("=" * 64)
    print("Finished at ", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 64)
