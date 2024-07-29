import unittest
import tempfile
import os
import shutil
import sys
sys.path.append("..")
from concept_graph.file_store import LocalFileStore

class TestLocalFileStore(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.file_store = LocalFileStore(self.temp_dir, "test/")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_set_file_prefix(self):
        new_prefix = "new_prefix/"
        self.file_store.set_file_prefix(new_prefix)
        self.assertEqual(self.file_store.file_prefix, new_prefix)

    def test_add_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"Test content")
            temp_file_path = temp_file.name
        remote_file_name = "test_file.txt"
        os.makedirs(os.path.join(self.temp_dir, self.file_store.file_prefix), exist_ok=True)  # Create the prefix directory
        self.file_store.add_file(temp_file_path, remote_file_name)
        full_path = os.path.join(self.temp_dir, self.file_store.file_prefix, remote_file_name)
        self.assertTrue(os.path.exists(full_path))
        os.remove(temp_file_path)

    def test_get_file(self):
        remote_file_name = "test_file.txt"
        expected_content = "Test content"
        os.makedirs(os.path.join(self.temp_dir, self.file_store.file_prefix), exist_ok=True)
        full_path = os.path.join(self.temp_dir, self.file_store.file_prefix, remote_file_name)
        with open(full_path, "w") as file:
            file.write(expected_content)
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_path = temp_file.name
        self.file_store.get_file(remote_file_name, temp_file_path)
        with open(temp_file_path, "r") as downloaded_file:
            content = downloaded_file.read()
        self.assertEqual(content, expected_content)
        os.remove(temp_file_path)

    def test_delete_prefix(self):
        file_names = ["test_file1.txt", "test_file2.txt"]
        os.makedirs(os.path.join(self.temp_dir, self.file_store.file_prefix), exist_ok=True)
        for file_name in file_names:
            full_path = os.path.join(self.temp_dir, self.file_store.file_prefix, file_name)
            with open(full_path, "w") as file:
                file.write("Test content")
        other_file = "other_file.txt"
        with open(os.path.join(self.temp_dir, other_file), "w") as file:
            file.write("Other content")
        self.file_store.delete_prefix()
        prefix_path = os.path.join(self.temp_dir, self.file_store.file_prefix.rstrip("/"))
        self.assertFalse(os.path.exists(prefix_path))  # Check if the prefix folder exists
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, other_file)))  # Check if the other file still exists

    def test_update_file(self):
        remote_file_name = "test_file.txt"
        initial_content = "Initial content"
        updated_content = "Updated content"
        os.makedirs(os.path.join(self.temp_dir, self.file_store.file_prefix), exist_ok=True)
        full_path = os.path.join(self.temp_dir, self.file_store.file_prefix, remote_file_name)
        with open(full_path, "w") as file:
            file.write(initial_content)
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(updated_content.encode())
            temp_file_path = temp_file.name
        self.file_store.update_file(temp_file_path, remote_file_name)
        with open(full_path, "r") as updated_file:
            content = updated_file.read()
        self.assertEqual(content, updated_content)
        os.remove(temp_file_path)

    def test_delete_prefix(self):
        file_names = ["test_file1.txt", "test_file2.txt"]
        for file_name in file_names:
            full_path = os.path.join(self.file_store.prefix_path, file_name)
            with open(full_path, "w") as file:
                file.write("Test content")
        
        other_file = "other_file.txt"
        with open(os.path.join(self.temp_dir, other_file), "w") as file:
            file.write("Other content")
        
        self.file_store.delete_prefix()
        
        # Check that the prefix directory is empty
        self.assertEqual(len(os.listdir(self.file_store.prefix_path)), 0)
        
        # Check that the other file still exists
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, other_file)))

if __name__ == '__main__':
    unittest.main()