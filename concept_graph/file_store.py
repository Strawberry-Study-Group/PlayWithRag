from google.cloud import storage
import os

class file_store:
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