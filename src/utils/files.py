import os
import tarfile
import shutil
import paramiko
from scp import SCPClient
from dotenv import load_dotenv

def send_to_remote(local_path):
    load_dotenv()
    # 1. Create SSH client
    ssh = paramiko.SSHClient()

    # 2. Load system and policy keys for unknown hosts
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # 3. Connect to remote server
        ssh.connect(hostname=os.getenv("REMOTE_HOST"), username=os.getenv(
            "REMOTE_USERNAME"), password=os.getenv("REMOTE_PWD"))

        # 4. Start SCP session
        with SCPClient(ssh.get_transport()) as scp:
            scp.put(local_path, remote_path=os.getenv("SCP_DEST"))
            print(
                f"Transfer completed to {os.getenv("REMOTE_HOST")}:{os.getenv("SCP_DEST")}/{local_path}")

    except Exception as e:
        print(f"Error while sending to remote: {e}")
    finally:
        ssh.close()


def backup(POLICIES_FOLDER: str, FOLDER_PREFIX: str, XML_FILE: str):
    # Save also the XML model file used for the environment saved in the folder models/ in the root of the project
    model_path = os.path.join(os.path.dirname(__file__), "../../models/SBRobot_Snello.xml")
    with open(f"{POLICIES_FOLDER}/{FOLDER_PREFIX}/SBRobot_Snello.xml", 'w') as f:
        with open(model_path, 'r') as original_xml:
            f.write(original_xml.read())

    # Save the scene.xml file
    scene_path = XML_FILE
    with open(f"{POLICIES_FOLDER}/{FOLDER_PREFIX}/scene.xml", 'w') as f:
        with open(scene_path, 'r') as original_scene:
            f.write(original_scene.read())

    # Save the mesh files
    mesh_folder = os.path.join(os.path.dirname(__file__), "../../models/mesh")
    dest_mesh_folder = f"{POLICIES_FOLDER}/{FOLDER_PREFIX}/mesh"
    if not os.path.exists(dest_mesh_folder):
        os.makedirs(dest_mesh_folder)
    for mesh_file in os.listdir(mesh_folder):
        with open(os.path.join(dest_mesh_folder, mesh_file), 'w') as f:
            with open(os.path.join(mesh_folder, mesh_file), 'r') as original_mesh:
                f.write(original_mesh.read())

    # Copy the reward.py file in folder f"{POLICIES_FOLDER}/{FOLDER_PREFIX}/"
    reward_path = os.path.join(os.path.dirname(__file__), "../env/wrappers/reward.py")
    with open(os.path.join(f"{POLICIES_FOLDER}/{FOLDER_PREFIX}/reward.py"), 'w') as f:
        with open(reward_path, 'r') as original_reward:
            f.write(original_reward.read())

    folder_to_compress = f"{POLICIES_FOLDER}/{FOLDER_PREFIX}"
    return folder_to_compress

def compress_and_remove(folder_to_compress, POLICY_SCP: bool = False):
    # Compress the folder
    with tarfile.open(f"{folder_to_compress}.tar.gz", mode='w:gz') as tar:
        tar.add(folder_to_compress, arcname=os.path.basename(folder_to_compress))
    # Remove the uncompressed folder
    shutil.rmtree(folder_to_compress)
    print(f"Training saved to: {folder_to_compress}.tar.gz")
    if POLICY_SCP:
        send_to_remote(f"{folder_to_compress}.tar.gz")