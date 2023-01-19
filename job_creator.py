import os.path
from pathlib import Path
import shutil

create_jobs = True
num_workers = 0

objs = ["gpsample", "hartmann", "plant", "airfoil"]
acquisitions = ["ts", "ucb", "etc_es0", "etc_es1", "etc_es2"]


missing_filenames = []
counter = 1
for obj_name in objs:
    if obj_name == "gpsample":
        budget = 50
    elif obj_name == "hartmann":
        budget = 50
    elif obj_name == "plant":
        budget = 200
    elif obj_name == "airfoil":
        budget = 50
    else:
        raise NotImplementedError

    base_dir = "results/" + obj_name + "/"
    save_dir = base_dir
    pickles_dir = base_dir + "pickles/"
    Path(pickles_dir).mkdir(parents=True, exist_ok=True)

    for costs_id in range(3):
        costs_dict = {0: "Cheap", 1: "Moderate", 2: "Expensive"}
        costs_alias = costs_dict[costs_id]
        for var_id in [2, 3, 4]:
            for acquisition in acquisitions:
                if acquisition == "ucb" or acquisition == "ts":
                    acq_alias = acquisition + "_es0"
                    if obj_name == "airfoil":
                        virtual_costs_id = costs_id
                        virtual_var_id = var_id
                    else:
                        # if full control query set exists,
                        # costs and variances do not matter
                        # for UCB-PSQ and TS-PSQ
                        virtual_costs_id = 0
                        virtual_var_id = 0
                else:
                    acq_alias = acquisition
                    virtual_costs_id = costs_id
                    virtual_var_id = var_id

                for seed in range(10):
                    filename = (
                        f"{obj_name}_{acq_alias}_c{virtual_costs_id}"
                        f"_var{virtual_var_id}_C{budget}_seed{seed}"
                    )
                    filename = filename.replace(".", ",") + ".p"

                    if not os.path.isfile(pickles_dir + filename):
                        if filename not in missing_filenames:
                            missing_filenames.append(filename)
                            print(f"{counter}. {filename} is missing")
                            counter += 1

if create_jobs:
    # Create job files
    job_dir = "jobs/"
    if os.path.exists(job_dir):  # empty the job_dir directory
        shutil.rmtree(job_dir)
    Path(job_dir).mkdir(parents=True, exist_ok=True)
    if num_workers != 0:
        for i, f in enumerate(missing_filenames):
            with open(job_dir + f"job{i % num_workers}.txt", "a") as file:
                file.write(f"{f}\n")
    else:
        for i, f in enumerate(missing_filenames):
            with open(job_dir + f"job.txt", "a") as file:
                file.write(f"{f}\n")
