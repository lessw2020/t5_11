import pytest
from utils import (
    wait_for_slurm_log,
    wait_for_training_start,
    start_training,
    inject_error,
    check_dcgm_error,
    read_dcgm_errors_file,
    cancel_dcgm_job,
)


DCGM_ERROR_FILE = "dcgm_errors.json"
SLURM_SCRIPT = "bert.slurm"
HOST = ""
GPU_ID = 0


@pytest.fixture
def setup():
    dcgm_errors_content = read_dcgm_errors_file(file=DCGM_ERROR_FILE)

    slurm_log_file, job_id = start_training(SLURM_SCRIPT)
    assert job_id is not None
    assert wait_for_slurm_log(job_id=job_id)
    assert wait_for_training_start(log_file=slurm_log_file)
    yield slurm_log_file, dcgm_errors_content
    cancel_dcgm_job(job_id)


@pytest.mark.parametrize("error_name", ["XID", "ECC", "NVLink", "PCIReplay"])
def test_gpu_error_injection(setup, error_name):
    slurm_log_file, dcgm_errors_content = setup
    assert error_name in dcgm_errors_content

    error_info = dcgm_errors_content[error_name]
    assert inject_error(
        host=HOST,
        gpu_id=GPU_ID,
        name=error_name,
        f_param=error_info["f_param"],
        v_param=error_info["v_param"],
    )

    assert check_dcgm_error(log_file=slurm_log_file, error_msg=error_info["msg"])
