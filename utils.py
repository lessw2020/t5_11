import subprocess
import os
import time
import json
import logging as log


def run_command(cmd: str):
    log.debug("Running command: {}".format(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    raw_output, raw_err = p.communicate()
    return raw_output


def wait_for_slurm_log(job_id: str, iteration: int = 30, sleep: int = 10):
    log.debug("{} Log generation {}".format("*" * 50, "*" * 50))
    slurm_log_file: str = "slurm-{}.out".format(job_id)
    for _ in range(iteration):
        log.debug("Waiting for slurm log file - {} to be created".format(slurm_log_file))
        if os.path.exists(slurm_log_file):
            log.debug("Slurm log file - {}found\n".format(slurm_log_file))
            return slurm_log_file
        else:
            time.sleep(sleep)
    else:
        return False


def wait_for_training_start(log_file: str, iteration: int = 30, sleep: int = 10):
    log.debug("{} Training {}".format("*" * 50, "*" * 50))
    for _ in range(iteration):
        log.debug("Waiting for training to get started")
        with open(log_file) as fp:
            data = fp.read()
        if "Epoch 0" in data:
            log.debug("Training start found\n")
            return True
        else:
            time.sleep(sleep)
    else:
        return False


def start_training(slurm_script):
    log.debug("{} Invoke slurm job {}".format("*" * 50, "*" * 50))
    cmd = "sbatch {}".format(slurm_script)
    output = run_command(cmd)
    if not isinstance(output, bytes) or "Submitted batch job" not in output.decode("utf-8"):
        raise Exception("Unable to start training with command: {}".format(cmd))
    else:
        job_id = output.decode("utf-8").split(" ")[-1].rstrip("\n")
        log.debug("Retreived Job ID: {}".format(job_id))
        log.debug("Slurm job started successfully \n")
        slurm_log_file: str = wait_for_slurm_log(job_id=job_id)
        training_status = wait_for_training_start(log_file=slurm_log_file)
        if not training_status:
            raise Exception("Training start not detected")
    return slurm_log_file, job_id


def inject_error(host: str, gpu_id: int, f_param: int, v_param: int, name: str):
    log.debug("{} Error injection {}".format("*" * 50, "*" * 50))
    cmd = "dcgmi test --host {} --inject --gpuid {} -f {} -v {}".format(
        host, gpu_id, f_param, v_param
    )
    try:
        response = run_command(cmd)
        log.debug(response)
        log.debug("\n")
    except Exception as e:
        raise
    return True


def check_dcgm_error(log_file: str, error_msg: str, iteration: int = 30, sleep: int = 10):
    log.debug("{} Listening for violation {}".format("*" * 50, "*" * 50))
    for _ in range(iteration):
        log.debug("Waiting for violation msg - {}".format(error_msg))
        with open(log_file) as fp:
            data = fp.read()
            if error_msg in data:
                log.debug("Violation found\n")
                return True
            else:
                time.sleep(sleep)
    else:
        log.debug("Violation not found")
        return False


def read_dcgm_errors_file(file: str):
    if not os.path.exists(file):
        raise FileNotFoundError("DCGM error file - {} not found".format(file))
    else:
        with open(file) as fp:
            data = json.load(fp)

    return data


def cancel_dcgm_job(job_id):
    log.debug("{} Cancelling job {}".format("*" * 50, "*" * 50))
    cmd = "scancel {}".format(job_id)
    output = run_command(cmd)
    log.debug(output)

    cmd = "squeue"
    for _ in range(30):
        output = run_command(cmd)
        log.debug("squeue output: {}".format(output))
        if job_id in output.decode("utf-8").split("\n")[1]:
            log.debug("Job still running - {}".format(output))
            time.sleep(10)
        else:
            log.debug("Job Cancelled successfully\n")
            return True
    else:
        log.debug("Unable to cancel the existing job")
        return False
