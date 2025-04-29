"""
runexp.py

Batch runner script to automate RL training experiments for different traffic scenarios.
- Sets random seeds
- Prepares SUMO and config paths
- Iterates through traffic file and model name combinations
- Modifies configuration files dynamically
- Launches training via traffic_light_dqn.main()
"""

# =================== Configuration (Only Edit These) ======================
SEED = 31200
setting_memo = "one_run"

# Traffic files to use for training and pretraining
list_traffic_files = [
    [["cross.2phases_rou1_switch_rou0.xml"], ["cross.2phases_rou1_switch_rou0.xml"]],
]

# Model names to train
list_model_name = ["Deeplight"]
# ============================================================================

# Random seed setup for reproducibility
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)

# System imports
import json
import os
import time
import traffic_light_dqn

# Paths
PATH_TO_CONF = os.path.join("conf", setting_memo)

# SUMO Commands
sumoBinary = r"/usr/bin/sumo-gui"
sumoBinary_nogui = r"/usr/bin/sumo"
sumoCmd = [sumoBinary, '-c', r'{0}/data/{1}/cross.sumocfg'.format(os.path.split(os.path.realpath(__file__))[0], setting_memo)]
sumoCmd_pretrain = [sumoBinary, '-c', r'{0}/data/{1}/cross_pretrain.sumocfg'.format(os.path.split(os.path.realpath(__file__))[0], setting_memo)]
sumoCmd_nogui = [sumoBinary_nogui, '-c', r'{0}/data/{1}/cross.sumocfg'.format(os.path.split(os.path.realpath(__file__))[0], setting_memo)]
sumoCmd_nogui_pretrain = [sumoBinary_nogui, '-c', r'{0}/data/{1}/cross_pretrain.sumocfg'.format(os.path.split(os.path.realpath(__file__))[0], setting_memo)]

# =================== Experiment Loop ======================
for model_name in list_model_name:
    for traffic_file, traffic_file_pretrain in list_traffic_files:

        # Load and update exp.conf
        dic_exp = json.load(open(os.path.join(PATH_TO_CONF, "exp.conf"), "r"))
        dic_exp["MODEL_NAME"] = model_name
        dic_exp["TRAFFIC_FILE"] = traffic_file
        dic_exp["TRAFFIC_FILE_PRETRAIN"] = traffic_file_pretrain

        if "real" in traffic_file[0]:
            dic_exp["RUN_COUNTS"] = 86400
        elif "2phase" in traffic_file[0]:
            dic_exp["RUN_COUNTS"] = 7200
        elif "synthetic" in traffic_file[0]:
            dic_exp["RUN_COUNTS"] = 216000

        json.dump(dic_exp, open(os.path.join(PATH_TO_CONF, "exp.conf"), "w"), indent=4)

        # Load and update sumo_agent.conf
        dic_sumo = json.load(open(os.path.join(PATH_TO_CONF, "sumo_agent.conf"), "r"))
        dic_sumo["MIN_ACTION_TIME"] = 5 if model_name == "Deeplight" else 1
        json.dump(dic_sumo, open(os.path.join(PATH_TO_CONF, "sumo_agent.conf"), "w"), indent=4)

        # Build prefix
        prefix = "{0}_{1}_{2}_{3}".format(
            dic_exp["MODEL_NAME"],
            dic_exp["TRAFFIC_FILE"],
            dic_exp["TRAFFIC_FILE_PRETRAIN"],
            time.strftime('%m_%d_%H_%M_%S_', time.localtime(time.time())) + "seed_%d" % SEED
        )

        # Run experiment
        traffic_light_dqn.main(memo=setting_memo, f_prefix=prefix, sumo_cmd_str=sumoCmd_nogui, sumo_cmd_pretrain_str=sumoCmd_nogui_pretrain)

        print(f"Finished training with traffic file {traffic_file}")

    print(f"Finished all experiments for model {model_name}")
