#from google.cloud import storage
import os
import shutil

"""
class GoogleFileStore:
    def __init__(self, project_id, bucket_path, file_prefix, credential_path) -> None:
        self.credential_path = credential_path
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.credential_path
        if not project_id:
            raise ValueError("Project ID is required.")
        if not file_prefix:
            raise ValueError("File prefix is required.")
        if not bucket_path:
            raise ValueError("Bucket path is required.")
        if not credential_path:
            raise ValueError("Credential path is required.")
        self.storage_client = storage.Client(project=project_id)
        self.file_prefix = file_prefix
        self.bucket = self.storage_client.bucket(bucket_path)



    def set_bucket(self, bucket_path):
        self.bucket = self.storage_client.bucket(bucket_path)
    
    def set_file_prefix(self, file_prefix):
        self.file_prefix = file_prefix

    def add_file(self, local_file_path, remote_file_name):
        if self.bucket is None:
            raise ValueError("Bucket not initialized. Please set the bucket using set_bucket() method.")
        blob = self.bucket.blob(self.file_prefix + remote_file_name)
        blob.upload_from_filename(local_file_path)
        print(f"File {local_file_path} uploaded to {self.file_prefix + remote_file_name}.")

    def get_file(self, remote_file_name, local_file_path):
        if self.bucket is None:
            raise ValueError("Bucket not initialized. Please set the bucket using set_bucket() method.")
        blob = self.bucket.blob(self.file_prefix + remote_file_name)
        blob.download_to_filename(local_file_path)
        print(f"File {self.file_prefix + remote_file_name} downloaded to {local_file_path}.")

    def delete_file(self, remote_file_name):
        if self.bucket is None:
            raise ValueError("Bucket not initialized. Please set the bucket using set_bucket() method.")
        blob = self.bucket.blob(self.file_prefix + remote_file_name)
        blob.delete()
        print(f"File {self.file_prefix + remote_file_name} deleted.")
    
    def update_file(self, local_file_path, remote_file_name):
        if self.bucket is None:
            raise ValueError("Bucket not initialized. Please set the bucket using set_bucket() method.")
        blob = self.bucket.blob(self.file_prefix + remote_file_name)
        blob.upload_from_filename(local_file_path)
        print(f"File {local_file_path} updated in {self.file_prefix + remote_file_name}.")

    def delete_prefix(self):
        if self.bucket is None:
            raise ValueError("Bucket not initialized. Please set the bucket using set_bucket() method.")
        blobs = self.bucket.list_blobs(prefix=self.file_prefix)
        for blob in blobs:
            blob.delete()
"""

class LocalFileStore:
    def __init__(self, base_path, file_prefix):
        self.base_path = base_path
        self.file_prefix = file_prefix
        
        # Create the base directory and prefix directory if they don't exist
        self.prefix_path = os.path.join(self.base_path, self.file_prefix)
        os.makedirs(self.prefix_path, exist_ok=True)

    def set_file_prefix(self, file_prefix):
        self.file_prefix = file_prefix
        self.prefix_path = os.path.join(self.base_path, self.file_prefix)
        os.makedirs(self.prefix_path, exist_ok=True)

    def _get_full_path(self, file_name):
        return os.path.join(self.prefix_path, file_name)

    def add_file(self, local_file_path, remote_file_name):
        full_path = self._get_full_path(remote_file_name)
        shutil.copy2(local_file_path, full_path)

    def get_file(self, remote_file_name, local_file_path):
        full_path = self._get_full_path(remote_file_name)
        shutil.copy2(full_path, local_file_path)

    def delete_file(self, remote_file_name):
        full_path = self._get_full_path(remote_file_name)
        os.remove(full_path)

    def update_file(self, local_file_path, remote_file_name):
        self.add_file(local_file_path, remote_file_name)

    def delete_prefix(self):
        if os.path.exists(self.prefix_path):
            shutil.rmtree(self.prefix_path)
        os.makedirs(self.prefix_path, exist_ok=True)  # Recreate the empty prefix directory

