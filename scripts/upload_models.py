import os
import json
import time
import datetime
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
MODELS_DIR = "saved_models"
MODEL_LOG_PATH = "trained_models.json"
DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")

# ✅ Authenticate using service account

   # ✅ Authenticate using OAuth 2.0
def auth_drive():
    gauth = GoogleAuth()
    gauth.LoadClientConfigFile("client_secrets.json")  # Load your client_secrets.json
    gauth.LocalWebserverAuth()  # Creates a local webserver and automatically handles authentication.
    return GoogleDrive(gauth)

def restore_trashed_files(drive):
    trashed = drive.ListFile({'q': "trashed=true"}).GetList()
    print(f"🗑️ Found {len(trashed)} trashed files/folders.")
    for file in trashed:
        try:
            print(f"♻️ Restoring: {file['title']}")
            file['trashed'] = False
            file.Upload()
        except Exception as e:
            print(f"❌ Failed to restore {file['title']}: {e}")
# ✅ Generate a timestamped versioned filename
def get_timestamped_name(original_name):
    name, ext = os.path.splitext(original_name)
    timestamp = datetime.datetime.now().strftime("v%Y-%m-%d_%H%M")
    return f"{name}_{timestamp}{ext}"

# ✅ Upload model folder and manage versioning
def upload_model_folder(drive, model_folder_path, _):
    model_name = os.path.basename(model_folder_path)
    folder_metadata = {
        'title': model_name,
        'parents': [{'id': DRIVE_FOLDER_ID}],
        'mimeType': 'application/vnd.google-apps.folder'
    }

    # Reuse existing folder if it exists
    existing_folders = drive.ListFile({
        'q': f"'{DRIVE_FOLDER_ID}' in parents and trashed=false and title='{model_name}' and mimeType='application/vnd.google-apps.folder'"
    }).GetList()

    if existing_folders:
        folder = existing_folders[0]
        print(f"📂 Using existing folder: {model_name}")
    else:
        folder = drive.CreateFile(folder_metadata)
        folder.Upload()
        print(f"✅ Created folder: {model_name}")

    metadata = {
        "name": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": []
    }

    # Get files already present in the folder
    existing_files = drive.ListFile({
        'q': f"'{folder['id']}' in parents and trashed=false"
    }).GetList()
    existing_file_map = {f['title']: f for f in existing_files}

    for fname in os.listdir(model_folder_path):
        fpath = os.path.join(model_folder_path, fname)
        if not os.path.isfile(fpath):
            continue

        if fname in existing_file_map:
            print(f"🟡 File already exists in Drive: {fname}")
            user_input = input("⚠️ Replace (y), Skip (n), or Versioned Upload (v)? [y/n/v]: ").strip().lower()

            if user_input == 'y':
                file_drive = existing_file_map[fname]
                file_drive.SetContentFile(fpath)
                file_drive.Upload()
                print(f"♻️ Replaced: {fname}")
                metadata["files"].append({
                    "name": fname,
                    "id": file_drive["id"],
                    "action": "replaced"
                })
            elif user_input == 'v':
                versioned_name = get_timestamped_name(fname)
                file_drive = drive.CreateFile({
                    'title': versioned_name,
                    'parents': [{'id': folder['id']}]
                })
                file_drive.SetContentFile(fpath)
                file_drive.Upload()
                print(f"🆕 Uploaded versioned file: {versioned_name}")
                metadata["files"].append({
                    "name": versioned_name,
                    "id": file_drive["id"],
                    "action": "versioned"
                })
            else:
                print(f"⏭️ Skipped: {fname}")
                continue
        else:
            try:
                file_drive = drive.CreateFile({
                    'title': fname,
                    'parents': [{'id': folder['id']}]
                })
                file_drive.SetContentFile(fpath)
                file_drive.Upload()
                print(f"✅ Uploaded: {fname}")
                metadata["files"].append({
                    "name": fname,
                    "id": file_drive["id"],
                    "action": "uploaded"
                })
            except Exception as e:
                print(f"❌ Upload failed for {fname}: {e}")

    return metadata

# ✅ Load local log
def load_existing_log():
    if os.path.exists(MODEL_LOG_PATH):
        try:
            with open(MODEL_LOG_PATH, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception as e:
            print(f"⚠️ Error loading log: {e}")
    return []

# ✅ Save local log
def save_log(log):
    with open(MODEL_LOG_PATH, 'w') as f:
        json.dump(log, f, indent=2)

# ✅ Main function
def main():
    print("🔐 Authenticating to Google Drive...")
    drive = auth_drive()

    print(f"📁 Upload target folder ID: {DRIVE_FOLDER_ID}")
    print("📦 Preparing to upload models...")

    uploaded_models = load_existing_log()

    for folder in os.listdir(MODELS_DIR):
        model_path = os.path.join(MODELS_DIR, folder)
        if not os.path.isdir(model_path):
            continue

        if any(m['name'] == folder for m in uploaded_models):
            print(f"🟡 Model already uploaded: {folder}")
            continue

        print(f"\n⬆️ Uploading model folder: {folder}")
        model_meta = upload_model_folder(drive, model_path, set())

        if model_meta:
            uploaded_models.append(model_meta)
        else:
            print(f"⚠️ No files uploaded for model: {folder}")

    save_log(uploaded_models)
    print("\n✅ All uploads complete. Log updated.")

# ✅ Run main
if __name__ == "__main__":
    main()
