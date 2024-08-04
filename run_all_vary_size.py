import os
import time
import subprocess

datasets = """Structured/Amazon-Google
Structured/Walmart-Amazon
Textual/Abt-Buy""".split('\n')

op = "swap"

lm = "roberta"
epochs = 20
run_count = 0

for dataset in datasets:
    for size in [500, 1000, 1500, 2000]:
        for run_id in range(3):
            for da in [True, False]:
                for dk in [True, False]:
                    ds = dataset
                    batch_size = 32
                    start = time.time()

                    cmd = [
                        "CUDA_VISIBLE_DEVICES=0", "python", "train_ditto.py",
                        "--task", ds,
                        "--logdir", "results_ditto/",
                        "--finetuning",
                        "--batch_size", str(batch_size),
                        "--lr", "3e-5",
                        "--fp16",
                        "--lm", lm,
                        "--n_epochs", str(epochs),
                        "--size", str(size),
                        "--run_id", str(run_id)
                    ]
                    
                    if da:
                        cmd.extend(['--da', op])
                    if dk:
                        cmd.extend(['--dk', 'product'])

                    run_count += 1
                    print("=" * 64)
                    print(f"Starting run {run_count} at ", time.strftime("%Y-%m-%d %H:%M:%S"))
                    print(f"Dataset: {dataset}")
                    print(f"Operation: {op}")
                    print(f"Language Model: {lm}")
                    print(f"Batch Size: {batch_size}")
                    print(f"Epochs: {epochs}")
                    print(f"Size: {size}")
                    print(f"Run ID: {run_id}")
                    print(f"DA: {da}")
                    print(f"DK: {dk}")
                    print(f"Command: {' '.join(cmd)}")
                    print("=" * 64)
                    
                    process = subprocess.Popen(' '.join(cmd), shell=True)
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
                    except Exception as e:
                        print(f"Error occurred: {e}")
                        process.terminate()
                        process.wait()
                    
                    print("=" * 64)
                    print("Finished at ", time.strftime("%Y-%m-%d %H:%M:%S"))
                    print("=" * 64)

        # Push notification command after completing each dataset size loop
        push_cmd = f"push {dataset} {size}"
        print("=" * 64)
        print(f"Executing push command for dataset {dataset} with size {size}")
        print(f"Command: {push_cmd}")
        print("=" * 64)
        
        push_process = subprocess.Popen(push_cmd, shell=True)
        try:
            push_process.wait()
        except KeyboardInterrupt:
            push_process.terminate()
            push_process.wait()
            print("Push command stopped by user.")
            print("=" * 64)
            print("Finished at ", time.strftime("%Y-%m-%d %H:%M:%S"))
            print("=" * 64)
            raise
        except Exception as e:
            print(f"Error occurred during push: {e}")
            push_process.terminate()
            push_process.wait()
        
        print("=" * 64)
        print("Push command finished at ", time.strftime("%Y-%m-%d %H:%M:%S"))
        print("=" * 64)
