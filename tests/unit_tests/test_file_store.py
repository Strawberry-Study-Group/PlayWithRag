import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging

from memory.file_store import (
    LocalFileStore,
    FileStoreFactory,
    FileStoreError,
    FileNotFoundError,
    FileOperationError,
    IFileStore
)


class TestLocalFileStore:
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger for testing."""
        return Mock(spec=logging.Logger)
    
    @pytest.fixture
    def file_store(self, temp_dir, mock_logger):
        """Create a LocalFileStore instance for testing."""
        return LocalFileStore(temp_dir, "test_prefix", mock_logger)
    
    @pytest.fixture
    def test_file(self, temp_dir):
        """Create a test file for testing operations."""
        test_file_path = Path(temp_dir) / "test_input.txt"
        test_file_path.write_text("test content")
        return str(test_file_path)
    
    def test_init_creates_prefix_directory(self, temp_dir, mock_logger):
        """Test that initialization creates the prefix directory."""
        prefix = "test_prefix"
        store = LocalFileStore(temp_dir, prefix, mock_logger)
        
        expected_path = Path(temp_dir) / prefix
        assert expected_path.exists()
        assert expected_path.is_dir()
        assert store.prefix_path == expected_path
    
    def test_set_file_prefix_updates_path_and_creates_directory(self, file_store, temp_dir, mock_logger):
        """Test that set_file_prefix updates the path and creates new directory."""
        new_prefix = "new_prefix"
        file_store.set_file_prefix(new_prefix)
        
        expected_path = Path(temp_dir) / new_prefix
        assert expected_path.exists()
        assert expected_path.is_dir()
        assert file_store.file_prefix == new_prefix
        assert file_store.prefix_path == expected_path
        mock_logger.info.assert_called_with(f"File prefix updated to: {new_prefix}")
    
    def test_add_file_success(self, file_store, test_file, mock_logger):
        """Test successful file addition."""
        remote_name = "test_remote.txt"
        file_store.add_file(test_file, remote_name)
        
        expected_path = file_store.prefix_path / remote_name
        assert expected_path.exists()
        assert expected_path.read_text() == "test content"
        mock_logger.info.assert_called_with(f"File added: {remote_name}")
    
    def test_add_file_local_not_found(self, file_store):
        """Test add_file raises FileNotFoundError when local file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Local file not found"):
            file_store.add_file("nonexistent.txt", "remote.txt")
    
    def test_add_file_creates_subdirectory(self, file_store, test_file, mock_logger):
        """Test that add_file creates subdirectories as needed."""
        remote_name = "subdir/test_remote.txt"
        file_store.add_file(test_file, remote_name)
        
        expected_path = file_store.prefix_path / remote_name
        assert expected_path.exists()
        assert expected_path.read_text() == "test content"
    
    def test_get_file_success(self, file_store, test_file, temp_dir, mock_logger):
        """Test successful file retrieval."""
        remote_name = "test_remote.txt"
        local_output = str(Path(temp_dir) / "output.txt")
        
        # First add the file
        file_store.add_file(test_file, remote_name)
        
        # Then retrieve it
        file_store.get_file(remote_name, local_output)
        
        assert Path(local_output).exists()
        assert Path(local_output).read_text() == "test content"
        mock_logger.info.assert_called_with(f"File retrieved: {remote_name}")
    
    def test_get_file_remote_not_found(self, file_store, temp_dir):
        """Test get_file raises FileNotFoundError when remote file doesn't exist."""
        local_output = str(Path(temp_dir) / "output.txt")
        
        with pytest.raises(FileNotFoundError, match="Remote file not found"):
            file_store.get_file("nonexistent.txt", local_output)
    
    def test_delete_file_success(self, file_store, test_file, mock_logger):
        """Test successful file deletion."""
        remote_name = "test_remote.txt"
        
        # First add the file
        file_store.add_file(test_file, remote_name)
        expected_path = file_store.prefix_path / remote_name
        assert expected_path.exists()
        
        # Then delete it
        file_store.delete_file(remote_name)
        
        assert not expected_path.exists()
        mock_logger.info.assert_called_with(f"File deleted: {remote_name}")
    
    def test_delete_file_not_found(self, file_store):
        """Test delete_file raises FileNotFoundError when file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Remote file not found"):
            file_store.delete_file("nonexistent.txt")
    
    def test_update_file_success(self, file_store, test_file, mock_logger):
        """Test successful file update."""
        remote_name = "test_remote.txt"
        file_store.update_file(test_file, remote_name)
        
        expected_path = file_store.prefix_path / remote_name
        assert expected_path.exists()
        assert expected_path.read_text() == "test content"
        mock_logger.info.assert_called_with(f"File updated: {remote_name}")
    
    def test_file_exists_true(self, file_store, test_file):
        """Test file_exists returns True for existing file."""
        remote_name = "test_remote.txt"
        file_store.add_file(test_file, remote_name)
        
        assert file_store.file_exists(remote_name) is True
    
    def test_file_exists_false(self, file_store):
        """Test file_exists returns False for non-existing file."""
        assert file_store.file_exists("nonexistent.txt") is False
    
    def test_delete_prefix_success(self, file_store, test_file, mock_logger):
        """Test successful prefix deletion and recreation."""
        remote_name = "test_remote.txt"
        file_store.add_file(test_file, remote_name)
        
        # Verify file exists
        expected_path = file_store.prefix_path / remote_name
        assert expected_path.exists()
        
        # Delete prefix
        file_store.delete_prefix()
        
        # Verify directory is recreated but empty
        assert file_store.prefix_path.exists()
        assert file_store.prefix_path.is_dir()
        assert not expected_path.exists()
        mock_logger.info.assert_called_with(f"Prefix directory cleared: {file_store.file_prefix}")
    
    @patch('requests.get')
    @patch('PIL.Image.open')
    def test_save_img_from_url_success(self, mock_image_open, mock_get, file_store, mock_logger):
        """Test successful image download and save."""
        # Mock the HTTP response
        mock_response = Mock()
        mock_response.content = b"fake image data"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        # Mock the PIL Image
        mock_img = Mock()
        mock_img.mode = 'RGB'
        mock_img.save = Mock()
        mock_image_open.return_value = mock_img
        
        url = "https://example.com/image.jpg"
        file_name = "test_image"
        
        result = file_store.save_img_from_url(url, file_name)
        
        # Verify the result path
        expected_path = file_store.prefix_path / "imgs" / f"{file_name}.jpg"
        assert result == str(expected_path)
        
        # Verify HTTP request was made
        mock_get.assert_called_once_with(url, timeout=30)
        mock_response.raise_for_status.assert_called_once()
        
        # Verify image processing
        mock_img.save.assert_called_once_with(expected_path, 'JPEG')
        mock_logger.info.assert_called_with(f"Image saved from URL: {file_name}.jpg")
    
    @patch('requests.get')
    @patch('PIL.Image.open')
    def test_save_img_from_url_rgba_conversion(self, mock_image_open, mock_get, file_store):
        """Test RGBA to RGB conversion during image save."""
        # Mock the HTTP response
        mock_response = Mock()
        mock_response.content = b"fake image data"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        # Mock the PIL Image with RGBA mode
        mock_img = Mock()
        mock_img.mode = 'RGBA'
        mock_converted_img = Mock()
        mock_img.convert.return_value = mock_converted_img
        mock_converted_img.save = Mock()
        mock_image_open.return_value = mock_img
        
        url = "https://example.com/image.png"
        file_name = "test_image"
        
        file_store.save_img_from_url(url, file_name)
        
        # Verify RGBA conversion
        mock_img.convert.assert_called_once_with('RGB')
        mock_converted_img.save.assert_called()
    
    @patch('requests.get')
    def test_save_img_from_url_http_error(self, mock_get, file_store):
        """Test save_img_from_url raises FileOperationError on HTTP error."""
        mock_get.side_effect = Exception("HTTP Error")
        
        with pytest.raises(FileOperationError, match="Failed to save image from URL"):
            file_store.save_img_from_url("https://example.com/image.jpg", "test")
    
    @patch('pathlib.Path.mkdir')
    def test_ensure_directory_exists_failure(self, mock_mkdir, file_store):
        """Test _ensure_directory_exists raises FileOperationError on failure."""
        # Mock mkdir to raise an OSError
        mock_mkdir.side_effect = OSError("Permission denied")
        
        test_path = Path("/some/path")
        
        with pytest.raises(FileOperationError, match="Failed to create directory"):
            file_store._ensure_directory_exists(test_path)


class TestFileStoreFactory:
    
    def test_create_local_store(self):
        """Test factory creates LocalFileStore instance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = FileStoreFactory.create_local_store(temp_dir, "test_prefix")
            
            assert isinstance(store, LocalFileStore)
            assert isinstance(store, IFileStore)
            assert store.base_path == Path(temp_dir)
            assert store.file_prefix == "test_prefix"
    
    def test_create_local_store_with_logger(self):
        """Test factory creates LocalFileStore with custom logger."""
        logger = Mock(spec=logging.Logger)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            store = FileStoreFactory.create_local_store(temp_dir, "test_prefix", logger)
            
            assert store.logger == logger
    
    def test_create_cloud_store_not_implemented(self):
        """Test factory raises NotImplementedError for cloud storage."""
        with pytest.raises(NotImplementedError, match="Cloud storage provider 'gcs' not yet implemented"):
            FileStoreFactory.create_cloud_store("gcs")


class TestFileStoreInterface:
    
    def test_interface_methods_exist(self):
        """Test that IFileStore interface defines all required methods."""
        required_methods = [
            'add_file', 'get_file', 'delete_file', 'update_file', 
            'file_exists', 'save_img_from_url'
        ]
        
        for method_name in required_methods:
            assert hasattr(IFileStore, method_name)
            assert callable(getattr(IFileStore, method_name))
    
    def test_local_file_store_implements_interface(self):
        """Test that LocalFileStore properly implements IFileStore interface."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalFileStore(temp_dir, "test")
            
            assert isinstance(store, IFileStore)
            
            # Verify all interface methods are implemented
            required_methods = [
                'add_file', 'get_file', 'delete_file', 'update_file',
                'file_exists', 'save_img_from_url'
            ]
            
            for method_name in required_methods:
                method = getattr(store, method_name)
                assert callable(method)
                assert not getattr(method, '__isabstractmethod__', False)