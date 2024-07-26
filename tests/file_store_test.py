import unittest
import tempfile
import os
import sys
import json
from PIL import Image

sys.path.append("..")
from concept_graph.file_store import file_store

PROJECT_ID = "powerful-surf-415220"
LOCATION = "us-west1"
CREDENTIALS_PATH = "/home/eddy/.config/gcloud/application_default_credentials.json"
BUCKET_PATH = "concept_store"
FILE_PREFIX = "test_world/"

class TestFileStore(unittest.TestCase):
    def setUp(self):
        self.file_store = file_store(PROJECT_ID, BUCKET_PATH, FILE_PREFIX, CREDENTIALS_PATH)

    def test_set_bucket(self):
        new_bucket_path = "new-bucket-name"
        self.file_store.set_bucket(new_bucket_path)
        self.assertEqual(self.file_store.bucket.name, new_bucket_path)

    def test_set_file_prefix(self):
        new_file_prefix = "new_prefix/"
        self.file_store.set_file_prefix(new_file_prefix)
        self.assertEqual(self.file_store.file_prefix, new_file_prefix)

    def test_add_file_txt(self):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"Test content")
            temp_file_path = temp_file.name
        remote_file_name = "test_file.txt"
        self.file_store.add_file(temp_file_path, remote_file_name)
        blob = self.file_store.bucket.blob(FILE_PREFIX + remote_file_name)
        self.assertTrue(blob.exists())
        os.remove(temp_file_path)

    def test_add_file_jpg(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            image = Image.new("RGB", (100, 100), color="red")
            image.save(temp_file.name)
            temp_file_path = temp_file.name
        remote_file_name = "test_image.jpg"
        self.file_store.add_file(temp_file_path, remote_file_name)
        blob = self.file_store.bucket.blob(FILE_PREFIX + remote_file_name)
        self.assertTrue(blob.exists())
        os.remove(temp_file_path)

    def test_add_file_json(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as temp_file:
            data = {"key": "value"}
            json.dump(data, temp_file)
            temp_file_path = temp_file.name
        remote_file_name = "test_data.json"
        self.file_store.add_file(temp_file_path, remote_file_name)
        blob = self.file_store.bucket.blob(FILE_PREFIX + remote_file_name)
        self.assertTrue(blob.exists())
        os.remove(temp_file_path) 

    def test_get_file(self):
        remote_file_name = "test_file.txt"
        expected_content = "Test content"
        blob = self.file_store.bucket.blob(FILE_PREFIX + remote_file_name)
        blob.upload_from_string(expected_content)
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_path = temp_file.name
        self.file_store.get_file(remote_file_name, temp_file_path)
        with open(temp_file_path, "r") as downloaded_file:
            content = downloaded_file.read()
        self.assertEqual(content, expected_content)
        os.remove(temp_file_path)

    def test_delete_file(self):
        remote_file_name = "test_file.txt"
        blob = self.file_store.bucket.blob(FILE_PREFIX + remote_file_name)
        blob.upload_from_string("Test content")
        self.file_store.delete_file(remote_file_name)
        self.assertFalse(blob.exists())

    def test_update_file(self):
        remote_file_name = "test_file.txt"
        initial_content = "Initial content"
        updated_content = "Updated content"
        blob = self.file_store.bucket.blob(FILE_PREFIX + remote_file_name)
        blob.upload_from_string(initial_content)
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(updated_content.encode())
            temp_file_path = temp_file.name
        self.file_store.update_file(temp_file_path, remote_file_name)
        updated_blob = self.file_store.bucket.blob(FILE_PREFIX + remote_file_name)
        updated_content_downloaded = updated_blob.download_as_string().decode()
        self.assertEqual(updated_content_downloaded, updated_content)
        os.remove(temp_file_path)

    def test_add_file_without_bucket(self):
        self.file_store.bucket = None
        with self.assertRaises(ValueError):
            self.file_store.add_file("dummy_path", "dummy_file")

    def test_get_file_without_bucket(self):
        self.file_store.bucket = None
        with self.assertRaises(ValueError):
            self.file_store.get_file("dummy_file", "dummy_path")

    def test_delete_file_without_bucket(self):
        self.file_store.bucket = None
        with self.assertRaises(ValueError):
            self.file_store.delete_file("dummy_file")

if __name__ == "__main__":
    unittest.main()