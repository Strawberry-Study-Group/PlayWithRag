from abc import ABC, abstractmethod
import shutil
from pathlib import Path
from typing import Optional
import logging


class FileStoreError(Exception):
    """Base exception for file store operations."""
    pass


class FileNotFoundError(FileStoreError):
    """Raised when a requested file is not found."""
    pass


class FileOperationError(FileStoreError):
    """Raised when a file operation fails."""
    pass


class IFileStore(ABC):
    """Interface for file storage operations."""

    @abstractmethod
    def add_file(self, local_file_path: str, remote_file_name: str) -> None:
        """Add a file to the store."""
        pass

    @abstractmethod
    def get_file(self, remote_file_name: str, local_file_path: str) -> None:
        """Retrieve a file from the store."""
        pass

    @abstractmethod
    def delete_file(self, remote_file_name: str) -> None:
        """Delete a file from the store."""
        pass

    @abstractmethod
    def update_file(self, local_file_path: str, remote_file_name: str) -> None:
        """Update an existing file in the store."""
        pass

    @abstractmethod
    def file_exists(self, remote_file_name: str) -> bool:
        """Check if a file exists in the store."""
        pass

    @abstractmethod
    def save_img_from_url(self, url: str, file_name: str) -> str:
        """Download and save an image from URL."""
        pass


class LocalFileStore(IFileStore):
    def __init__(self, base_path: str, file_prefix: str,
                 logger: Optional[logging.Logger] = None):
        self.base_path = Path(base_path)
        self.file_prefix = file_prefix
        self.logger = logger or logging.getLogger(__name__)

        # Create the base directory and prefix directory if they don't exist
        self.prefix_path = self.base_path / self.file_prefix
        self._ensure_directory_exists(self.prefix_path)

    def set_file_prefix(self, file_prefix: str) -> None:
        """Update the file prefix and ensure directory exists."""
        self.file_prefix = file_prefix
        self.prefix_path = self.base_path / self.file_prefix
        self._ensure_directory_exists(self.prefix_path)
        self.logger.info(f"File prefix updated to: {file_prefix}")

    def _ensure_directory_exists(self, path: Path) -> None:
        """Ensure a directory exists, creating it if necessary."""
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise FileOperationError(f"Failed to create directory {path}: {e}")

    def _get_full_path(self, file_name: str) -> Path:
        """Get the full path for a file in the store."""
        return self.prefix_path / file_name

    def add_file(self, local_file_path: str, remote_file_name: str) -> None:
        """Add a file to the store."""
        try:
            local_path = Path(local_file_path)
            if not local_path.exists():
                raise FileNotFoundError(
                    f"Local file not found: {local_file_path}")

            full_path = self._get_full_path(remote_file_name)
            self._ensure_directory_exists(full_path.parent)
            shutil.copy2(local_path, full_path)
            self.logger.info(f"File added: {remote_file_name}")
        except (OSError, shutil.Error) as e:
            raise FileOperationError(
                f"Failed to add file {remote_file_name}: {e}")

    def get_file(self, remote_file_name: str, local_file_path: str) -> None:
        """Retrieve a file from the store."""
        try:
            full_path = self._get_full_path(remote_file_name)
            if not full_path.exists():
                raise FileNotFoundError(
                    f"Remote file not found: {remote_file_name}")

            local_path = Path(local_file_path)
            self._ensure_directory_exists(local_path.parent)
            shutil.copy2(full_path, local_path)
            self.logger.info(f"File retrieved: {remote_file_name}")
        except (OSError, shutil.Error) as e:
            raise FileOperationError(
                f"Failed to get file {remote_file_name}: {e}")

    def delete_file(self, remote_file_name: str) -> None:
        """Delete a file from the store."""
        try:
            full_path = self._get_full_path(remote_file_name)
            if not full_path.exists():
                raise FileNotFoundError(
                    f"Remote file not found: {remote_file_name}")

            full_path.unlink()
            self.logger.info(f"File deleted: {remote_file_name}")
        except OSError as e:
            raise FileOperationError(
                f"Failed to delete file {remote_file_name}: {e}")

    def update_file(self, local_file_path: str, remote_file_name: str) -> None:
        """Update an existing file in the store."""
        self.add_file(local_file_path, remote_file_name)
        self.logger.info(f"File updated: {remote_file_name}")

    def file_exists(self, remote_file_name: str) -> bool:
        """Check if a file exists in the store."""
        full_path = self._get_full_path(remote_file_name)
        return full_path.exists()

    def delete_prefix(self) -> None:
        """Delete all files in the current prefix directory."""
        try:
            if self.prefix_path.exists():
                shutil.rmtree(self.prefix_path)
            self._ensure_directory_exists(self.prefix_path)
            self.logger.info(f"Prefix directory cleared: {self.file_prefix}")
        except (OSError, shutil.Error) as e:
            raise FileOperationError(
                f"Failed to delete prefix {self.file_prefix}: {e}")

    def save_img_from_url(self, url: str, file_name: str) -> str:
        """Download and save an image from URL."""
        try:
            import requests
            from PIL import Image
            import io

            # Download the image
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Open the image using PIL
            img = Image.open(io.BytesIO(response.content))

            # Convert to RGB if the image is in RGBA mode
            if img.mode == 'RGBA':
                img = img.convert('RGB')

            # Create the imgs directory if it doesn't exist
            imgs_dir = self.prefix_path / "imgs"
            self._ensure_directory_exists(imgs_dir)

            # Append .jpg to the filename
            file_name_with_ext = f"{file_name}.jpg"
            full_path = imgs_dir / file_name_with_ext

            # Save the image as JPEG
            img.save(full_path, 'JPEG')
            self.logger.info(f"Image saved from URL: {file_name_with_ext}")

            return str(full_path)
        except Exception as e:
            raise FileOperationError(
                f"Failed to save image from URL {url}: {e}")


class FileStoreFactory:
    """Factory for creating file store instances."""

    @staticmethod
    def create_local_store(base_path: str, file_prefix: str,
                           logger: Optional[logging.Logger] = None) -> IFileStore:
        """Create a local file store instance."""
        return LocalFileStore(base_path, file_prefix, logger)

    @staticmethod
    def create_cloud_store(provider: str, **config) -> IFileStore:
        """Create a cloud file store instance (placeholder for future implementation)."""
        raise NotImplementedError(
            f"Cloud storage provider '{provider}' not yet implemented")
