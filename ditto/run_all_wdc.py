import os
import time
import subprocess

datasets = ["all", "computers", "cameras", "shoes", "watches"]
sizes = ["small", "medium", "large", "xlarge"]

gpu_id = 0
counter = 0

try:
    for d in datasets:
        for size in sizes:
            dataset = '_'.join(['wdc', d, size])
            for dk in [True, False]:
                for da in [True, False]:
                    for run_id in range(1):
                        cmd = [
                            "CUDA_VISIBLE_DEVICES=%d" % gpu_id, "python", "train_ditto.py",
                            "--task", dataset,
                            "--logdir", "results_wdc/",
                            "--fp16",
                            "--finetuning",
                            "--batch_size", "32",
                            "--lr", "3e-5",
                            "--n_epochs", "10",
                            "--run_id", str(run_id)
                        ]
                        if da:
                            cmd.extend(['--da', 'del'])
                        if dk:
                            cmd.extend(['--dk', 'product'])
                        counter += 1

                        print("=" * 64)
                        print(f"Starting run {counter} at ", time.strftime("%Y-%m-%d %H:%M:%S"))
                        print(f"Dataset: {dataset}")
                        print(f"Size: {size}")
                        print(f"Run ID: {run_id}")
                        print(f"DA: {da}")
                        print(f"DK: {dk}")
                        print(f"Batch Size: 32")
                        print(f"Learning Rate: 3e-5")
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

            # Push command after completing each dataset size loop
            push_cmd = f"push {dataset}"
            print("=" * 64)
            print(f"Executing push command for dataset {dataset}")
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

except KeyboardInterrupt:
    print("### Process interrupted by user. Exiting...")
    print("=" * 64)
    print("Finished at ", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 64)
