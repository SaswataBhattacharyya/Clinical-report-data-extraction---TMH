import os
import stat
import time
import subprocess
import webbrowser
from pathlib import Path
from typing import Optional

import paramiko
import streamlit as st


# --------- CONSTANTS YOU CAN ADJUST ----------
DEFAULT_LOGIN_HOST = "ln1.hpc.com"
DEFAULT_LOCAL_TUNNEL_PORT = 8501

# Base project dir on HPC (user-specific part will be filled with username)
HPC_BASE_TEMPLATE = "/home/{user}/saswata/Clinical-report-data-extraction---TMH"
HPC_TEST_SUBDIR = "test"

# PBS scripts (assumed to live in HPC base dir)
LLM_PBS_NAME = "mtb.pbs"       # for llm_app
GRAPH_PBS_NAME = "mtb_app.pbs" # for graphapp

GPU_QUEUE_NAME = "a40"         # queue to inspect with qstat -q

# Where graphapp writes output on HPC (to be pulled back)
REMOTE_OUTPUT_SUBDIR = "output"  # under test/
# --------------------------------------------


# ---------- HELPER FUNCTIONS ----------

def get_hpc_base_dir(username: str) -> str:
    return HPC_BASE_TEMPLATE.format(user=username)


def ssh_connect(login_host: str, username: str, password: str) -> paramiko.SSHClient:
    """Create a new SSH connection to login node with password auth."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(login_host, username=username, password=password)
    return client


def run_remote_cmd(ssh_client: paramiko.SSHClient, cmd: str) -> str:
    """Run a command on the remote login node and return stdout + stderr."""
    stdin, stdout, stderr = ssh_client.exec_command(cmd)
    out = stdout.read().decode("utf-8", errors="ignore")
    err = stderr.read().decode("utf-8", errors="ignore")
    return out + err


def read_remote_file(ssh_client: paramiko.SSHClient, path: str) -> Optional[str]:
    """Read a text file from remote host; return content or None if missing."""
    sftp = ssh_client.open_sftp()
    try:
        with sftp.file(path, "r") as f:
            return f.read().strip()
    except IOError:
        return None
    finally:
        sftp.close()


def sftp_upload_files(
    ssh_client: paramiko.SSHClient,
    local_files,
    remote_dir: str,
    show_progress_label: str,
):
    """Upload a list of uploaded files to a remote directory via SFTP."""
    sftp = ssh_client.open_sftp()

    # Ensure remote dir exists
    _ensure_remote_dir(sftp, remote_dir)

    total = len(local_files)
    progress = st.progress(0, text=show_progress_label)

    for i, f in enumerate(local_files, start=1):
        remote_path = os.path.join(remote_dir, f.name)
        with sftp.file(remote_path, "wb") as remote_f:
            remote_f.write(f.read())
        progress.progress(i / total, text=f"{show_progress_label}: {i}/{total}")

    sftp.close()
    progress.empty()


def sftp_upload_single_file(
    ssh_client: paramiko.SSHClient,
    local_file,
    remote_dir: str,
    remote_name: str,
    show_progress_label: str,
):
    """Upload a single file to a remote directory via SFTP."""
    sftp = ssh_client.open_sftp()

    _ensure_remote_dir(sftp, remote_dir)

    progress = st.progress(0, text=show_progress_label)
    remote_path = os.path.join(remote_dir, remote_name)
    with sftp.file(remote_path, "wb") as remote_f:
        data = local_file.read()
        remote_f.write(data)
    progress.progress(1.0, text=f"{show_progress_label}: done")

    sftp.close()
    progress.empty()


def _ensure_remote_dir(sftp: paramiko.SFTPClient, remote_dir: str):
    """Ensure a remote directory exists, creating parents as needed."""
    try:
        sftp.stat(remote_dir)
    except IOError:
        parts = remote_dir.strip("/").split("/")
        cur = ""
        for p in parts:
            cur = f"{cur}/{p}" if cur else f"/{p}"
            try:
                sftp.stat(cur)
            except IOError:
                sftp.mkdir(cur)


def start_ssh_tunnel_with_sshpass(
    username: str,
    password: str,
    login_host: str,
    compute_node: str,
    local_port: int,
):
    """
    Start an SSH tunnel using sshpass + ssh:
      sshpass -p PASSWORD ssh -J user@login_host user@compute_node -L local_port:127.0.0.1:8501 -N
    NOTE: Requires sshpass to be installed on the laptop system.
    """
    cmd = [
        "sshpass",
        "-p",
        password,
        "ssh",
        "-J",
        f"{username}@{login_host}",
        f"{username}@{compute_node}",
        "-L",
        f"{local_port}:127.0.0.1:8501",
        "-N",  # no remote command; just tunnel
        "-o",
        "StrictHostKeyChecking=no",
    ]
    proc = subprocess.Popen(cmd)
    return proc


def parse_a40_free(qstat_q_output: str) -> bool:
    """
    Very simple heuristic: look for the line beginning with GPU_QUEUE_NAME,
    parse Run/Que columns, and consider 'free' if Que == 0.
    """
    for line in qstat_q_output.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(GPU_QUEUE_NAME):
            parts = line.split()
            if len(parts) < 7:
                return False
            # Example:
            # a40  -- -- 48:00:00 2 4 1 -- E R
            try:
                run_val = int(parts[-5])  # Run
                que_val = int(parts[-4])  # Que
            except Exception:
                return False
            # here we treat "no queued jobs" as "nodes free"
            return que_val == 0
    # If we didn't find the queue line, be conservative
    return False


def submit_pbs_job(ssh_client: paramiko.SSHClient, hpc_base: str, script_name: str) -> str:
    """Run qsub on a PBS script located in the HPC base dir; return jobid."""
    cmd = f"cd {hpc_base} && qsub {script_name}"
    out = run_remote_cmd(ssh_client, cmd).strip()
    # Typically qsub returns "<jobid>.hn1"
    jobid = out.split()[0] if out else ""
    if not jobid:
        raise RuntimeError(f"Could not parse job id from qsub output: {out}")
    return jobid


def wait_for_job_running(
    ssh_client: paramiko.SSHClient,
    jobid: str,
    timeout_seconds: int = 120,
    interval_seconds: int = 30,
) -> Optional[str]:
    """
    Poll qstat -f jobid until job_state=R or timeout.
    Returns compute node name (parsed from exec_host) or None if not running in time.
    """
    elapsed = 0
    while elapsed <= timeout_seconds:
        info = run_remote_cmd(ssh_client, f"qstat -f {jobid}")
        if "Unknown Job Id" in info:
            # might not be registered yet; wait
            time.sleep(interval_seconds)
            elapsed += interval_seconds
            continue

        state = None
        exec_host = None
        for line in info.splitlines():
            line = line.strip()
            if line.startswith("job_state ="):
                state = line.split("=", 1)[1].strip()
            if line.startswith("exec_host ="):
                exec_host = line.split("=", 1)[1].strip()

        if state == "R":
            if exec_host:
                # exec_host looks like "cn043/0*4"
                node = exec_host.split("/")[0].strip()
                return node
            else:
                # running but no exec_host? rare, but fallback
                return None

        # if state is C/E or something else, break
        if state in ("C", "E"):
            return None

        time.sleep(interval_seconds)
        elapsed += interval_seconds

    return None


def sftp_download_dir(
    ssh_client: paramiko.SSHClient,
    remote_dir: str,
    local_dir: str,
):
    """
    Recursively download a remote directory over SFTP to a local directory.
    Raises if remote dir missing or empty.
    """
    sftp = ssh_client.open_sftp()

    try:
        attrs = sftp.stat(remote_dir)
    except IOError:
        sftp.close()
        raise RuntimeError(f"Remote directory {remote_dir} not found.")

    if not stat.S_ISDIR(attrs.st_mode):
        sftp.close()
        raise RuntimeError(f"{remote_dir} is not a directory on the remote host.")

    items = sftp.listdir_attr(remote_dir)
    if not items:
        sftp.close()
        raise RuntimeError(f"Remote directory {remote_dir} is empty.")

    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    def _download_recursive(remote_path: str, local_path: Path):
        for item in sftp.listdir_attr(remote_path):
            rname = item.filename
            rpath = f"{remote_path}/{rname}"
            lpath = local_path / rname
            if stat.S_ISDIR(item.st_mode):
                lpath.mkdir(exist_ok=True)
                _download_recursive(rpath, lpath)
            else:
                with sftp.open(rpath, "rb") as rf, open(lpath, "wb") as lf:
                    lf.write(rf.read())

    _download_recursive(remote_dir, local_path)
    sftp.close()


# ---------- STREAMLIT APP ----------

st.set_page_config(page_title="HPC Streamlit Connector", layout="centered")

st.title("HPC Helper: Connect, Upload & Launch Apps")
st.caption(
    "Runs on your laptop. Connects to ln1.hpc.com, uploads data, and launches Streamlit apps on GPU nodes."
)

# Session state
if "ssh_ok" not in st.session_state:
    st.session_state.ssh_ok = False
if "ssh_password" not in st.session_state:
    st.session_state.ssh_password = ""
if "ssh_username" not in st.session_state:
    st.session_state.ssh_username = ""
if "ssh_login_host" not in st.session_state:
    st.session_state.ssh_login_host = DEFAULT_LOGIN_HOST
if "tunnel_proc" not in st.session_state:
    st.session_state.tunnel_proc = None
if "current_jobid" not in st.session_state:
    st.session_state.current_jobid = None
if "current_app" not in st.session_state:
    st.session_state.current_app = None
if "graphapp_selected" not in st.session_state:
    st.session_state.graphapp_selected = False


st.header("Step 1 – Connect to HPC login node")

with st.form("hpc_login_form"):
    login_host = st.text_input("IP address", value=st.session_state.ssh_login_host)
    username = st.text_input("Username", value=st.session_state.ssh_username)
    password = st.text_input("Password", type="password", value=st.session_state.ssh_password)
    submitted = st.form_submit_button("Test Connection / Login")

    if submitted:
        if not login_host or not username or not password:
            st.error("Please fill in host, username, and password.")
        else:
            try:
                client = ssh_connect(login_host, username, password)
                client.close()
                st.session_state.ssh_ok = True
                st.session_state.ssh_password = password
                st.session_state.ssh_username = username
                st.session_state.ssh_login_host = login_host
                st.success("✅ Connected successfully to login node.")
            except Exception as e:
                st.session_state.ssh_ok = False
                st.error(f"❌ Connection failed: {e}")

if not st.session_state.ssh_ok:
    st.stop()

st.info(
    "You are connected (credentials verified). The app will use these to upload data "
    "and start PBS jobs + SSH tunnel. Password is kept only in memory for this session."
)


def get_ssh_client() -> paramiko.SSHClient:
    return ssh_connect(
        st.session_state.ssh_login_host,
        st.session_state.ssh_username,
        st.session_state.ssh_password,
    )


hpc_base = get_hpc_base_dir(st.session_state.ssh_username)
hpc_test_dir = os.path.join(hpc_base, HPC_TEST_SUBDIR)
st.markdown(f"**HPC base dir:** `{hpc_base}`")
st.markdown(f"**HPC test dir:** `{hpc_test_dir}`")

# ----------------- STEP 2: UPLOADS -----------------

st.header("Step 2 – Optional data uploads to HPC")

st.subheader("2.1 Molecular biology folder → `mol_bio`")
mol_bio_files = st.file_uploader(
    "Upload files for mol_bio (you can select multiple files)",
    type=None,
    accept_multiple_files=True,
    key="mol_bio_files",
)
upload_mol_bio = st.checkbox("Upload mol_bio files to HPC", value=False, key="upload_mol_bio")

st.subheader("2.2 Clinician folder → `clinician`")
clinician_files = st.file_uploader(
    "Upload files for clinician (you can select multiple files)",
    type=None,
    accept_multiple_files=True,
    key="clinician_files",
)
upload_clinician = st.checkbox("Upload clinician files to HPC", value=False, key="upload_clinician")

st.subheader("2.3 Excel2 file → `excel2/excel2.xlsx`")
excel2_file = st.file_uploader(
    "Upload Excel2 file (excel2.xlsx)",
    type=["xlsx"],
    accept_multiple_files=False,
    key="excel2_file",
)
upload_excel2 = st.checkbox(
    "Upload Excel2 file to HPC", value=False, key="upload_excel2"
)

if st.button("Run uploads (if selected)"):
    client = None
    try:
        client = get_ssh_client()
        st.write("Connected to login node for uploads.")

        # 2.1 mol_bio
        if upload_mol_bio and mol_bio_files:
            remote_dir = os.path.join(hpc_test_dir, "mol_bio")
            st.write(f"Uploading mol_bio files to `{remote_dir}` ...")
            sftp_upload_files(client, mol_bio_files, remote_dir, "Uploading mol_bio files")

        # 2.2 clinician
        if upload_clinician and clinician_files:
            remote_dir = os.path.join(hpc_test_dir, "clinician")
            st.write(f"Uploading clinician files to `{remote_dir}` ...")
            sftp_upload_files(client, clinician_files, remote_dir, "Uploading clinician files")

        # 2.3 excel2
        if upload_excel2 and excel2_file:
            remote_dir = os.path.join(hpc_test_dir, "excel2")
            st.write(f"Uploading Excel2 to `{remote_dir}/excel2.xlsx` ...")
            sftp_upload_single_file(
                client,
                excel2_file,
                remote_dir,
                "excel2.xlsx",
                "Uploading Excel2 file",
            )

        st.success("✅ Upload step finished.")
    except Exception as e:
        st.error(f"❌ Upload failed: {e}")
    finally:
        if client:
            client.close()


# ----------------- STEP 3: LAUNCH APPS & TUNNEL -----------------

st.header("Step 3 – Launch Streamlit app on GPU node & connect")

st.markdown(
    "We will:\n"
    "1. Check GPU queue `a40` with `qstat -q`.\n"
    "2. If nodes are free (no queued jobs), submit PBS job.\n"
    "   - For **llm_app**: `qsub mtb.pbs`\n"
    "   - For **graphapp**: `qsub mtb_app.pbs`\n"
    "3. Poll `qstat -f JOBID` every 30 seconds for up to 2 minutes until job is running.\n"
    "4. When running, read `exec_host` to find the compute node (e.g., `cn043`).\n"
    "5. Start SSH tunnel to that node and open `http://localhost:8501` in your browser."
)

st.markdown(
    "⚠️ For automated password-based tunneling, this app assumes `sshpass` is installed on your laptop.\n"
    "Install it with e.g. `sudo apt install sshpass` on Ubuntu if needed."
)


def launch_app_and_tunnel(app_name: str, pbs_script: str):
    """Common logic for both llm_app and graphapp buttons."""
    client = None
    try:
        client = get_ssh_client()

        # Check queue status
        st.write(f"Checking GPU queue `{GPU_QUEUE_NAME}` on HPC...")
        q_out = run_remote_cmd(client, f"qstat -q {GPU_QUEUE_NAME}")
        st.code(q_out, language="bash")
        if not parse_a40_free(q_out):
            st.error("Nodes occupied or queue has waiting jobs. Try again later.")
            return

        st.write(f"Queue looks free. Submitting PBS job `{pbs_script}` for {app_name}...")
        jobid = submit_pbs_job(client, hpc_base, pbs_script)
        st.session_state.current_jobid = jobid
        st.session_state.current_app = app_name
        st.info(f"Submitted job: `{jobid}`. Waiting for it to start running...")

        # Poll qstat -f JOBID
        node = wait_for_job_running(client, jobid, timeout_seconds=120, interval_seconds=30)
        if not node:
            st.error(
                "Job did not reach running state within 2 minutes or ended early. "
                "Please check HPC job logs."
            )
            return

        st.success(f"Job `{jobid}` is running on compute node `{node}`.")

        # Start SSH tunnel
        st.write("Starting SSH tunnel from laptop to compute node...")
        proc = start_ssh_tunnel_with_sshpass(
            username=st.session_state.ssh_username,
            password=st.session_state.ssh_password,
            login_host=st.session_state.ssh_login_host,
            compute_node=node,
            local_port=DEFAULT_LOCAL_TUNNEL_PORT,
        )
        st.session_state.tunnel_proc = proc
        st.success(
            f"✅ Tunnel started to `{node}`. Opening http://localhost:{DEFAULT_LOCAL_TUNNEL_PORT} ..."
        )
        webbrowser.open(f"http://localhost:{DEFAULT_LOCAL_TUNNEL_PORT}", new=2)

    except FileNotFoundError:
        st.error(
            "sshpass not found on your laptop. Install it with e.g. `sudo apt install sshpass`, "
            "or switch to SSH keys."
        )
    except Exception as e:
        st.error(f"❌ Failed to launch {app_name} and start tunnel: {e}")
    finally:
        if client:
            client.close()


col1, col2 = st.columns(2)

with col1:
    if st.button("Launch & connect llm_app"):
        st.session_state.graphapp_selected = False
        launch_app_and_tunnel("llm_app", LLM_PBS_NAME)

with col2:
    if st.button("Launch & connect graphapp"):
        st.session_state.graphapp_selected = True
        launch_app_and_tunnel("graphapp", GRAPH_PBS_NAME)

if st.session_state.tunnel_proc is not None:
    st.warning(
        f"Tunnel appears to be running for `{st.session_state.current_app}`. "
        "Use 'Disconnect tunnel' to stop it."
    )

if st.button("Disconnect tunnel"):
    proc = st.session_state.tunnel_proc
    if proc is not None:
        try:
            proc.terminate()
        except Exception:
            pass
        st.session_state.tunnel_proc = None
        st.success("Tunnel process terminated (best effort).")
    else:
        st.info("No tunnel process stored.")


# ----------------- EXTRA: GRAPHAPP OUTPUT RETRIEVAL -----------------

if st.session_state.graphapp_selected:
    st.header("Step 4 – Retrieve graphapp output from HPC")

    st.markdown(
        "Graphapp writes its output to the remote folder:\n"
        f"`{hpc_test_dir}/{REMOTE_OUTPUT_SUBDIR}`\n\n"
        "Use the controls below to copy that folder to your laptop."
    )

    local_output_dir = st.text_input(
        "Local folder path to store output (on your laptop)",
        value=str(Path.home() / "graphapp_output"),
    )

    if st.button("Retrieve output"):
        if not local_output_dir:
            st.error("Please specify a local folder path.")
        else:
            client = None
            try:
                client = get_ssh_client()
                remote_output_dir = os.path.join(hpc_test_dir, REMOTE_OUTPUT_SUBDIR)
                st.write(f"Copying remote `{remote_output_dir}` → `{local_output_dir}` ...")
                sftp_download_dir(client, remote_output_dir, local_output_dir)
                st.success("Success bro 😄  Output folder copied to your local machine.")
            except Exception as e:
                st.error(f"❌ Failed to retrieve output: {e}")
            finally:
                if client:
                    client.close()
