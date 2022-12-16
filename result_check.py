import os.path
from pathlib import Path
import shutil

# objs = ["gpsample", "hartmann", "plant"]
num_workers = 8
objs = ["gpsample", "hartmann"]
acquisitions = ["ucb-cs_es0", "ucb-cs_es1", "ucb", "ts"]

missing_filenames = []


for obj_name in objs:
    if obj_name == "gpsample":
        budget = 100
    else:
        budget = 500

    base_dir = "results/" + obj_name + "/"
    save_dir = base_dir
    pickles_dir = base_dir + "pickles/"

    for costs_id in range(3):
        costs_dict = {0: "Cheap", 1: "Moderate", 2: "Expensive"}
        costs_alias = costs_dict[costs_id]
        for var_id in range(3):
            var_dict = {0: 0.01, 1: 0.04, 2: 0.08}
            variance = var_dict[var_id]
            for acquisition in acquisitions:
                if acquisition == "ucb" or acquisition == "ts":
                    acq_alias = acquisition + "_es0"
                    virtual_costs_id = 0
                    virtual_var_id = 0
                else:
                    acq_alias = acquisition
                    virtual_costs_id = costs_id
                    virtual_var_id = var_id

                for seed in range(5):
                    filename = (
                        f"{obj_name}_{acq_alias}_c{virtual_costs_id}"
                        f"_var{virtual_var_id}_C{budget}_seed{seed}"
                    )
                    filename = filename.replace(".", ",") + ".p"

                    if not os.path.isfile(pickles_dir + filename):
                        missing_filenames.append(filename)
                        print(f"{filename} is missing")

# Create job files
job_dir = "jobs/"
if os.path.exists(job_dir):  # empty the job_dir directory
    shutil.rmtree(job_dir)
Path(job_dir).mkdir(parents=True, exist_ok=True)


for i, f in enumerate(missing_filenames):
    with open(job_dir + f"job{i % num_workers}.txt", "a") as file:
        file.write(f"{f}\n")
