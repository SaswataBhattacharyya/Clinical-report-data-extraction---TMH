import os
import subprocess
from pathlib import Path
from typing import Optional

import streamlit as st
import paramiko


# --------- CONSTANTS YOU CAN ADJUST ----------
DEFAULT_LOGIN_HOST = "ln1.hpc.com"
DEFAULT_LOCAL_TUNNEL_PORT = 8501
# Base project dir on HPC (user-specific part will be filled with username)
HPC_BASE_TEMPLATE = "/home/{user}/saswata/Clinical-report-data-extraction---TMH"
HPC_TEST_SUBDIR = "test"
HPC_NODE_INFO_REL = "node_info.txt"  # optional file created by your HPC job
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


def sftp_upload_files(
    ssh_client: paramiko.SSHClient,
    local_files,
    remote_dir: str,
    show_progress_label: str,
):
    """Upload a list of uploaded files to a remote directory via SFTP."""
    sftp = ssh_client.open_sftp()

    # Ensure remote dir exists
    try:
        sftp.stat(remote_dir)
    except IOError:
        # dir does not exist, create
        parts = remote_dir.strip("/").split("/")
        cur = ""
        for p in parts:
            cur = f"{cur}/{p}" if cur else f"/{p}"
            try:
                sftp.stat(cur)
            except IOError:
                sftp.mkdir(cur)

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

    # Ensure remote dir exists
    try:
        sftp.stat(remote_dir)
    except IOError:
        # dir does not exist, create
        parts = remote_dir.strip("/").split("/")
        cur = ""
        for p in parts:
            cur = f"{cur}/{p}" if cur else f"/{p}"
            try:
                sftp.stat(cur)
            except IOError:
                sftp.mkdir(cur)

    progress = st.progress(0, text=show_progress_label)
    remote_path = os.path.join(remote_dir, remote_name)
    with sftp.file(remote_path, "wb") as remote_f:
        data = local_file.read()
        remote_f.write(data)
    progress.progress(1.0, text=f"{show_progress_label}: done")

    sftp.close()
    progress.empty()


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


def start_ssh_tunnel_with_sshpass(
    username: str,
    password: str,
    login_host: str,
    compute_node: str,
    local_port: int,
):
    """
    Start an SSH tunnel using sshpass + ssh:
      sshpass -p PASSWORD ssh -J user@login_host user@compute_node -L local_port:127.0.0.1:8501
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
        "-N",           # no remote command; just tunnel
        "-o",
        "StrictHostKeyChecking=no",
    ]
    # Run in background
    proc = subprocess.Popen(cmd)
    return proc


# ---------- STREAMLIT APP ----------

st.set_page_config(page_title="HPC Streamlit Connector", layout="centered")

st.title("HPC Helper: Connect & Upload Data")
st.caption(
    "Runs on your laptop. Connects to ln1.hpc.com, uploads data, and opens a tunnel "
    "to your Streamlit app running on the HPC."
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
if "compute_node" not in st.session_state:
    st.session_state.compute_node = ""


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
    "and start the SSH tunnel. Password is kept only in memory for this session."
)

# Reconnect a fresh client when needed
def get_ssh_client() -> paramiko.SSHClient:
    return ssh_connect(
        st.session_state.ssh_login_host,
        st.session_state.ssh_username,
        st.session_state.ssh_password,
    )


st.header("Step 2 – Optional data uploads to HPC")

hpc_base = get_hpc_base_dir(st.session_state.ssh_username)
hpc_test_dir = os.path.join(hpc_base, HPC_TEST_SUBDIR)

st.markdown(f"**HPC base dir:** `{hpc_base}`")
st.markdown(f"Files will go under: `{hpc_test_dir}`")

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
upload_clinician = st.checkbox(
    "Upload clinician files to HPC", value=False, key="upload_clinician"
)

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


st.header("Step 3 – Connect tunnel to Streamlit app on GPU node")

st.markdown(
    "Your Streamlit app (`llm_app.py`) is running on a compute node (e.g., `cn043`) "
    "and listening on `127.0.0.1:8501` **on that node**.\n\n"
    "We will establish a tunnel:\n\n"
    "`localhost:8501 (laptop) → login node → compute-node:8501`"
)

compute_node_source = st.radio(
    "How to get the compute node hostname?",
    ["Enter manually", "Read from node_info file on HPC"],
    index=1,
)

compute_node = ""
if compute_node_source == "Enter manually":
    compute_node = st.text_input(
        "Compute node hostname (e.g. cn043)",
        value=st.session_state.compute_node or "",
        key="compute_node_manual",
    )
else:
    # read from node_info.txt
    if st.button("Read compute node from node_info file on HPC"):
        try:
            client = get_ssh_client()
            node_info_path = os.path.join(hpc_base, HPC_NODE_INFO_REL)
            content = read_remote_file(client, node_info_path)
            client.close()
            if content:
                st.session_state.compute_node = content.strip()
                st.success(f"Found compute node in `{node_info_path}`: `{content}`")
            else:
                st.error(f"Could not read `{node_info_path}` or file is empty.")
        except Exception as e:
            st.error(f"Error reading node_info file: {e}")

    compute_node = st.text_input(
        "Compute node hostname (from file or manually)",
        value=st.session_state.compute_node,
        key="compute_node_final",
    )

st.markdown(
    "⚠️ For automated password-based tunneling, this app assumes `sshpass` is installed on your laptop.\n"
    "If it is not, install it via your system package manager (e.g., `sudo apt install sshpass` on Ubuntu) "
    "or switch to SSH keys later."
)

if st.session_state.tunnel_proc is not None:
    st.warning("Tunnel appears to be running (process stored).")

if st.button("Connect to Streamlit app (open tunnel)"):
    if not compute_node:
        st.error("Please specify a compute node hostname.")
    else:
        try:
            proc = start_ssh_tunnel_with_sshpass(
                username=st.session_state.ssh_username,
                password=st.session_state.ssh_password,
                login_host=st.session_state.ssh_login_host,
                compute_node=compute_node,
                local_port=DEFAULT_LOCAL_TUNNEL_PORT,
            )
            st.session_state.tunnel_proc = proc
            st.success(
                f"✅ Tunnel started to `{compute_node}`. "
                f"Open http://localhost:{DEFAULT_LOCAL_TUNNEL_PORT} in your browser."
            )
        except FileNotFoundError:
            st.error(
                "sshpass not found. Please install it on your laptop "
                "(e.g., `sudo apt install sshpass`) or configure SSH keys and "
                "use a different tunneling method."
            )
        except Exception as e:
            st.error(f"❌ Failed to start tunnel: {e}")

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
