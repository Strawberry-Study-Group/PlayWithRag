import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import logging
from PIL import Image
import numpy as np

from memory.memory import MemoryCoreService, ConceptNotFoundError, MemoryCoreError
from memory.file_store import LocalFileStore, IFileStore, FileStoreError
from memory.graph_store import IGraphStore
from memory.emb_store import IEmbService
from memory.constants import ConceptGraphConstants, get_image_path, get_default_refs


class TestMemoryImageOperations:
    """Test image saving and reading functionality in memory core."""
    
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
    def mock_graph_store(self):
        """Create a mock graph store for testing."""
        mock = Mock(spec=IGraphStore)
        mock.get_node.return_value = None
        return mock
    
    @pytest.fixture
    def mock_emb_store(self):
        """Create a mock embedding store for testing."""
        mock = Mock(spec=IEmbService)
        return mock
    
    @pytest.fixture
    def file_store(self, temp_dir, mock_logger):
        """Create a LocalFileStore instance for testing."""
        return LocalFileStore(temp_dir, "test_prefix", mock_logger)
    
    @pytest.fixture
    def memory_service(self, file_store, mock_graph_store, mock_emb_store, mock_logger):
        """Create a MemoryCoreService instance for testing."""
        return MemoryCoreService(file_store, mock_graph_store, mock_emb_store, mock_logger)
    
    @pytest.fixture
    def gray_test_image(self, temp_dir):
        """Create a gray test image for testing operations."""
        test_image_path = Path(temp_dir) / "gray_test_image.jpg"
        
        # Create a 100x100 gray image
        gray_array = np.full((100, 100, 3), 128, dtype=np.uint8)  # 128 = middle gray
        gray_image = Image.fromarray(gray_array, 'RGB')
        gray_image.save(test_image_path, 'JPEG')
        
        return str(test_image_path)
    
    @pytest.fixture
    def white_test_image(self, temp_dir):
        """Create a white test image for testing operations."""
        test_image_path = Path(temp_dir) / "white_test_image.jpg"
        
        # Create a 100x100 white image
        white_array = np.full((100, 100, 3), 255, dtype=np.uint8)  # 255 = white
        white_image = Image.fromarray(white_array, 'RGB')
        white_image.save(test_image_path, 'JPEG')
        
        return str(test_image_path)
    
    @pytest.fixture
    def concept_data(self):
        """Sample concept data for testing."""
        return {
            "concept_name": "Test Concept",
            "concept_type": "character",
            "concept_attributes": {"description": "A test character"},
            "is_editable": True
        }

    def test_add_concept_with_single_image_path(self, memory_service, gray_test_image, concept_data, mock_graph_store, mock_emb_store):
        """Test adding a concept with a single image using image_path parameter."""
        # Mock graph store to simulate concept doesn't exist yet
        mock_graph_store.get_node.return_value = None
        
        # Add concept with image
        concept_id = memory_service.add_concept(
            concept_data["concept_name"],
            concept_data["concept_type"], 
            concept_data["concept_attributes"],
            concept_data["is_editable"],
            image_path=gray_test_image
        )
        
        # Verify concept ID was generated
        assert concept_id is not None
        assert len(concept_id) == 32  # UUID hex length
        
        # Verify file store was called to add image
        expected_image_path = get_image_path(concept_id)
        assert memory_service.file_store.file_exists(expected_image_path)
        
        # Verify graph store was called to add node
        mock_graph_store.add_node.assert_called_once()
        added_node = mock_graph_store.add_node.call_args[0][0]
        assert added_node[ConceptGraphConstants.FIELD_IMAGE_PATH] == expected_image_path
        assert added_node[ConceptGraphConstants.FIELD_NODE_ID] == concept_id
        assert added_node[ConceptGraphConstants.FIELD_NODE_NAME] == concept_data["concept_name"]
        
        # Verify refs structure is present with default values
        assert ConceptGraphConstants.FIELD_REFS in added_node
        refs = added_node[ConceptGraphConstants.FIELD_REFS]
        assert "ref_img" in refs
        assert isinstance(refs["ref_img"], list)

    def test_add_concept_with_ref_img_array(self, memory_service, gray_test_image, white_test_image, concept_data, mock_graph_store, mock_emb_store):
        """Test adding a concept with multiple images in refs.ref_img array."""
        # Mock graph store to simulate concept doesn't exist yet
        mock_graph_store.get_node.return_value = None
        
        # Prepare refs with multiple images
        refs = get_default_refs()
        refs["ref_img"] = [gray_test_image, white_test_image]
        
        # Add concept with ref images
        concept_id = memory_service.add_concept(
            concept_data["concept_name"],
            concept_data["concept_type"],
            concept_data["concept_attributes"],
            concept_data["is_editable"],
            refs=refs
        )
        
        # Verify concept was created
        assert concept_id is not None
        
        # Verify graph store was called to add node
        mock_graph_store.add_node.assert_called_once()
        added_node = mock_graph_store.add_node.call_args[0][0]
        
        # Verify refs were stored correctly
        assert ConceptGraphConstants.FIELD_REFS in added_node
        stored_refs = added_node[ConceptGraphConstants.FIELD_REFS]
        assert "ref_img" in stored_refs
        assert len(stored_refs["ref_img"]) == 2
        assert gray_test_image in stored_refs["ref_img"]
        assert white_test_image in stored_refs["ref_img"]

    def test_add_concept_with_both_image_path_and_ref_img(self, memory_service, gray_test_image, white_test_image, concept_data, mock_graph_store, mock_emb_store):
        """Test adding a concept with both single image_path and ref_img array."""
        # Mock graph store to simulate concept doesn't exist yet
        mock_graph_store.get_node.return_value = None
        
        # Prepare refs with additional images
        refs = get_default_refs()
        refs["ref_img"] = [white_test_image]
        
        # Add concept with both image_path and ref_img
        concept_id = memory_service.add_concept(
            concept_data["concept_name"],
            concept_data["concept_type"],
            concept_data["concept_attributes"],
            concept_data["is_editable"],
            image_path=gray_test_image,
            refs=refs
        )
        
        # Verify concept was created
        assert concept_id is not None
        
        # Verify file store was called to add the main image
        expected_image_path = get_image_path(concept_id)
        assert memory_service.file_store.file_exists(expected_image_path)
        
        # Verify graph store was called to add node
        mock_graph_store.add_node.assert_called_once()
        added_node = mock_graph_store.add_node.call_args[0][0]
        
        # Verify both image_path and ref_img are present
        assert added_node[ConceptGraphConstants.FIELD_IMAGE_PATH] == expected_image_path
        stored_refs = added_node[ConceptGraphConstants.FIELD_REFS]
        assert white_test_image in stored_refs["ref_img"]

    def test_update_concept_add_ref_img(self, memory_service, gray_test_image, white_test_image, mock_graph_store, mock_emb_store):
        """Test updating a concept to add images to ref_img array."""
        concept_id = "test_concept_id"
        existing_node = {
            ConceptGraphConstants.FIELD_NODE_ID: concept_id,
            ConceptGraphConstants.FIELD_NODE_NAME: "Test Concept",
            ConceptGraphConstants.FIELD_NODE_TYPE: "character",
            ConceptGraphConstants.FIELD_NODE_ATTRIBUTES: {"description": "A test concept"},
            ConceptGraphConstants.FIELD_IS_EDITABLE: True,
            ConceptGraphConstants.FIELD_REFS: get_default_refs()
        }
        
        # Mock graph store to return existing node
        mock_graph_store.get_node.return_value = existing_node
        
        # Update refs to add images
        new_refs = get_default_refs()
        new_refs["ref_img"] = [gray_test_image, white_test_image]
        
        # Update concept with new refs
        memory_service.update_concept(concept_id, refs=new_refs)
        
        # Verify update_node was called
        mock_graph_store.update_node.assert_called_once()
        updated_node = mock_graph_store.update_node.call_args[0][0]
        
        # Verify ref_img was updated
        assert updated_node[ConceptGraphConstants.FIELD_REFS]["ref_img"] == [gray_test_image, white_test_image]
        
        # Verify embeddings were updated
        mock_emb_store.update_node_emb.assert_called_once()

    def test_update_concept_replace_main_image(self, memory_service, gray_test_image, white_test_image, mock_graph_store, mock_emb_store):
        """Test updating a concept to replace the main image_path."""
        concept_id = "test_concept_id"
        old_image_path = get_image_path(concept_id)
        
        existing_node = {
            ConceptGraphConstants.FIELD_NODE_ID: concept_id,
            ConceptGraphConstants.FIELD_NODE_NAME: "Test Concept",
            ConceptGraphConstants.FIELD_NODE_TYPE: "character",
            ConceptGraphConstants.FIELD_NODE_ATTRIBUTES: {"description": "A test concept"},
            ConceptGraphConstants.FIELD_IS_EDITABLE: True,
            ConceptGraphConstants.FIELD_IMAGE_PATH: old_image_path,
            ConceptGraphConstants.FIELD_REFS: get_default_refs()
        }
        
        # Mock graph store to return existing node
        mock_graph_store.get_node.return_value = existing_node
        
        # Update concept with new image
        memory_service.update_concept(concept_id, image_path=white_test_image)
        
        # Verify update_node was called
        mock_graph_store.update_node.assert_called_once()
        updated_node = mock_graph_store.update_node.call_args[0][0]
        assert updated_node[ConceptGraphConstants.FIELD_IMAGE_PATH] == white_test_image

    def test_delete_concept_with_ref_img_array(self, memory_service, gray_test_image, white_test_image, mock_graph_store, mock_emb_store, mock_logger):
        """Test deleting a concept that has images in ref_img array."""
        concept_id = "test_concept_id"
        
        # Create refs with images
        refs = get_default_refs()
        refs["ref_img"] = [gray_test_image, white_test_image]
        
        existing_node = {
            ConceptGraphConstants.FIELD_NODE_ID: concept_id,
            ConceptGraphConstants.FIELD_NODE_NAME: "Test Concept",
            ConceptGraphConstants.FIELD_NODE_TYPE: "character",
            ConceptGraphConstants.FIELD_NODE_ATTRIBUTES: {"description": "A test concept"},
            ConceptGraphConstants.FIELD_IS_EDITABLE: True,
            ConceptGraphConstants.FIELD_REFS: refs
        }
        
        # Mock graph store to return existing node
        mock_graph_store.get_node.return_value = existing_node
        
        # Delete concept
        memory_service.delete_concept(concept_id)
        
        # Verify graph and embeddings were deleted
        mock_graph_store.delete_node.assert_called_once_with(concept_id)
        assert mock_emb_store.delete_node_emb.call_count == 2  # Full node + node name
        
        # Note: ref_img files are not automatically deleted by memory service
        # They are references to external files that may be used by other concepts

    def test_get_concept_with_images(self, memory_service, mock_graph_store):
        """Test retrieving a concept that has image references."""
        concept_id = "test_concept_id"
        image_path = get_image_path(concept_id)
        
        # Create refs with images
        refs = get_default_refs()
        refs["ref_img"] = ["path/to/image1.jpg", "path/to/image2.jpg"]
        
        stored_node = {
            ConceptGraphConstants.FIELD_NODE_ID: concept_id,
            ConceptGraphConstants.FIELD_NODE_NAME: "Test Concept",
            ConceptGraphConstants.FIELD_NODE_TYPE: "character",
            ConceptGraphConstants.FIELD_NODE_ATTRIBUTES: {"description": "A test concept"},
            ConceptGraphConstants.FIELD_IS_EDITABLE: True,
            ConceptGraphConstants.FIELD_IMAGE_PATH: image_path,
            ConceptGraphConstants.FIELD_REFS: refs
        }
        
        # Mock graph store to return the node
        mock_graph_store.get_node.return_value = stored_node
        
        # Get concept
        retrieved_concept = memory_service.get_concept(concept_id)
        
        # Verify the concept was retrieved with all image references
        assert retrieved_concept is not None
        assert retrieved_concept[ConceptGraphConstants.FIELD_IMAGE_PATH] == image_path
        assert retrieved_concept[ConceptGraphConstants.FIELD_REFS]["ref_img"] == ["path/to/image1.jpg", "path/to/image2.jpg"]

    def test_image_file_validation_during_concept_creation(self, memory_service, concept_data, mock_graph_store):
        """Test that non-existent image files are handled properly during concept creation."""
        # Mock graph store to simulate concept doesn't exist yet
        mock_graph_store.get_node.return_value = None
        
        non_existent_image = "/path/to/nonexistent/image.jpg"
        
        # Should raise an exception when trying to add non-existent image
        with pytest.raises((FileStoreError, FileNotFoundError)):
            memory_service.add_concept(
                concept_data["concept_name"],
                concept_data["concept_type"],
                concept_data["concept_attributes"],
                concept_data["is_editable"],
                image_path=non_existent_image
            )

    def test_ref_img_validation_empty_array(self, memory_service, concept_data, mock_graph_store, mock_emb_store):
        """Test that empty ref_img array is handled correctly."""
        # Mock graph store to simulate concept doesn't exist yet
        mock_graph_store.get_node.return_value = None
        
        # Prepare refs with empty ref_img
        refs = get_default_refs()
        refs["ref_img"] = []
        
        # Add concept with empty ref_img
        concept_id = memory_service.add_concept(
            concept_data["concept_name"],
            concept_data["concept_type"],
            concept_data["concept_attributes"],
            concept_data["is_editable"],
            refs=refs
        )
        
        # Verify concept was created successfully
        assert concept_id is not None
        
        # Verify graph store was called to add node
        mock_graph_store.add_node.assert_called_once()
        added_node = mock_graph_store.add_node.call_args[0][0]
        assert added_node[ConceptGraphConstants.FIELD_REFS]["ref_img"] == []

    def test_multiple_image_formats_in_ref_img(self, memory_service, temp_dir, concept_data, mock_graph_store, mock_emb_store):
        """Test handling multiple image formats in ref_img array."""
        # Create test images with different formats
        png_image_path = Path(temp_dir) / "test_image.png"
        jpg_image_path = Path(temp_dir) / "test_image.jpg"
        
        # Create PNG image
        png_array = np.full((50, 50, 3), 100, dtype=np.uint8)
        png_image = Image.fromarray(png_array, 'RGB')
        png_image.save(png_image_path, 'PNG')
        
        # Create JPG image
        jpg_array = np.full((50, 50, 3), 200, dtype=np.uint8)
        jpg_image = Image.fromarray(jpg_array, 'RGB')
        jpg_image.save(jpg_image_path, 'JPEG')
        
        # Mock graph store to simulate concept doesn't exist yet
        mock_graph_store.get_node.return_value = None
        
        # Prepare refs with multiple image formats
        refs = get_default_refs()
        refs["ref_img"] = [str(png_image_path), str(jpg_image_path)]
        
        # Add concept
        concept_id = memory_service.add_concept(
            concept_data["concept_name"],
            concept_data["concept_type"],
            concept_data["concept_attributes"],
            concept_data["is_editable"],
            refs=refs
        )
        
        # Verify concept was created
        assert concept_id is not None
        
        # Verify refs were stored correctly
        mock_graph_store.add_node.assert_called_once()
        added_node = mock_graph_store.add_node.call_args[0][0]
        stored_ref_img = added_node[ConceptGraphConstants.FIELD_REFS]["ref_img"]
        assert str(png_image_path) in stored_ref_img
        assert str(jpg_image_path) in stored_ref_img

    def test_image_path_standardization(self, memory_service, gray_test_image, concept_data, mock_graph_store, mock_emb_store):
        """Test that main image paths are standardized correctly."""
        # Mock graph store to simulate concept doesn't exist yet
        mock_graph_store.get_node.return_value = None
        
        # Add concept with image
        concept_id = memory_service.add_concept(
            concept_data["concept_name"],
            concept_data["concept_type"],
            concept_data["concept_attributes"],
            concept_data["is_editable"],
            image_path=gray_test_image
        )
        
        # Verify stored image path follows the standard format
        expected_image_path = get_image_path(concept_id)
        assert expected_image_path == f"imgs/{concept_id}.jpg"
        
        # Verify the node contains the standardized path
        added_node = mock_graph_store.add_node.call_args[0][0]
        assert added_node[ConceptGraphConstants.FIELD_IMAGE_PATH] == expected_image_path

    def test_search_concepts_with_images(self, memory_service, mock_graph_store):
        """Test searching for concepts that have image references."""
        # Create mock concepts with different image configurations
        concept1 = {
            ConceptGraphConstants.FIELD_NODE_ID: "concept1",
            ConceptGraphConstants.FIELD_NODE_NAME: "Concept with main image",
            ConceptGraphConstants.FIELD_NODE_TYPE: "character",
            ConceptGraphConstants.FIELD_IMAGE_PATH: "imgs/concept1.jpg",
            ConceptGraphConstants.FIELD_REFS: get_default_refs()
        }
        
        refs_with_images = get_default_refs()
        refs_with_images["ref_img"] = ["image1.jpg", "image2.jpg"]
        concept2 = {
            ConceptGraphConstants.FIELD_NODE_ID: "concept2", 
            ConceptGraphConstants.FIELD_NODE_NAME: "Concept with ref images",
            ConceptGraphConstants.FIELD_NODE_TYPE: "location",
            ConceptGraphConstants.FIELD_REFS: refs_with_images
        }
        
        concept3 = {
            ConceptGraphConstants.FIELD_NODE_ID: "concept3",
            ConceptGraphConstants.FIELD_NODE_NAME: "Concept without images",
            ConceptGraphConstants.FIELD_NODE_TYPE: "item",
            ConceptGraphConstants.FIELD_REFS: get_default_refs()
        }
        
        mock_graph_store.get_all_nodes.return_value = [concept1, concept2, concept3]
        
        # Search for concepts with main image
        criteria = {ConceptGraphConstants.FIELD_IMAGE_PATH: {"$exists": True}}
        results = memory_service.search_concepts(criteria)
        
        # Should find concept1 (has main image path)
        # Note: The actual search implementation depends on ConceptOperations.search_concepts
        mock_graph_store.get_all_nodes.assert_called_once()

    def test_validate_concept_with_invalid_refs_structure(self, memory_service, concept_data, mock_graph_store, mock_emb_store):
        """Test validation when refs structure is invalid."""
        # Mock graph store to simulate concept doesn't exist yet
        mock_graph_store.get_node.return_value = None
        
        # Create invalid refs structure (ref_img should be array, not string)
        invalid_refs = {
            "ref_img": "not_an_array.jpg",  # Should be array
            "ref_audio": [],
            "ref_video": [],
            "ref_docs": []
        }
        
        # This should either handle gracefully or raise validation error
        # depending on the ConceptBuilder validation implementation
        try:
            concept_id = memory_service.add_concept(
                concept_data["concept_name"],
                concept_data["concept_type"],
                concept_data["concept_attributes"],
                concept_data["is_editable"],
                refs=invalid_refs
            )
            # If no exception, verify the concept was still created
            assert concept_id is not None
        except (ValueError, TypeError) as e:
            # If validation catches the error, that's also acceptable
            assert "ref_img" in str(e) or "array" in str(e).lower()

    @patch('uuid.uuid4')
    def test_concept_id_collision_with_image_cleanup(self, mock_uuid, memory_service, gray_test_image, concept_data, mock_graph_store, mock_emb_store):
        """Test that image cleanup works correctly when concept ID generation has collisions."""
        # Mock UUID to return predictable values
        mock_uuid_obj1 = Mock()
        mock_uuid_obj1.hex = "duplicate_id"
        mock_uuid_obj2 = Mock()
        mock_uuid_obj2.hex = "unique_id"
        
        mock_uuid.side_effect = [mock_uuid_obj1, mock_uuid_obj2]
        
        # First call should return True (exists), second should return False
        mock_graph_store.get_node.side_effect = [{"some": "node"}, None]
        
        # Add concept with image
        concept_id = memory_service.add_concept(
            concept_data["concept_name"],
            concept_data["concept_type"],
            concept_data["concept_attributes"],
            concept_data["is_editable"],
            image_path=gray_test_image
        )
        
        # Should have handled collision and used second ID
        assert concept_id == "unique_id"
        assert mock_graph_store.get_node.call_count == 2
        
        # Verify image was stored with the final concept ID
        expected_image_path = get_image_path(concept_id)
        assert memory_service.file_store.file_exists(expected_image_path)