import os
from minio import Minio
from tqdm import tqdm

client = Minio(
    "minio.iiit.ac.in",
    access_key="21IZTJBKGHHEBHME4E1Y",
    secret_key="BELJFjX7wQ+QvkICXxyqYInPyvLOCLVXyxCVZbAP",
    secure=True # Set to False if not using HTTPS
)

bucket_name = "vlg"
bucket_prefix = "curated_data/"      # upload dir
local_directory = "/scratch/akash"

all_files = []
for root, dirs, files in os.walk(local_directory):
    for file in files:
        local_file_path = os.path.join(root, file)
        relative_path = os.path.relpath(local_file_path, local_directory)
        object_name = relative_path.replace("\\", "/")
        all_files.append((local_file_path, object_name))

for local_file_path, object_name in tqdm(all_files, desc="Uploading", unit="file"):
    full_object_name = bucket_prefix + object_name
    client.fput_object(bucket_name, full_object_name, local_file_path)
