"""Constants for concept graph module."""

from typing import Dict, List


class ConceptGraphConstants:
    """Constants used throughout the concept graph module."""

    # Embedding namespaces
    NAMESPACE_FULL_NODE = "full_node"
    NAMESPACE_NODE_NAME = "node_name"

    # Separators and delimiters
    NODE_NAME_SEPARATOR = "||"

    # Default values
    DEFAULT_REFS = {
        "ref_img": [],
        "ref_audio": [],
        "ref_video": [],
        "ref_docs": []
    }

    # File paths
    DEFAULT_IMAGE_PREFIX = "imgs"
    DEFAULT_IMAGE_EXTENSION = ".jpg"

    # Node field names
    FIELD_NODE_ID = "node_id"
    FIELD_NODE_NAME = "node_name"
    FIELD_NODE_TYPE = "node_type"
    FIELD_NODE_ATTRIBUTES = "node_attributes"
    FIELD_IS_EDITABLE = "is_editable"
    FIELD_REFS = "refs"
    FIELD_IMAGE_PATH = "image_path"

    # Edge field names
    FIELD_SOURCE_NODE_ID = "source_node_id"
    FIELD_TARGET_NODE_ID = "target_node_id"
    FIELD_EDGE_TYPE = "edge_type"

    # Reference types
    REF_IMG = "ref_img"
    REF_AUDIO = "ref_audio"
    REF_VIDEO = "ref_video"
    REF_DOCS = "ref_docs"


def get_default_refs() -> Dict[str, List[str]]:
    """Get a fresh copy of default refs dictionary."""
    return ConceptGraphConstants.DEFAULT_REFS.copy()


def get_image_path(concept_id: str) -> str:
    """Get standardized image path for a concept."""
    return f"{ConceptGraphConstants.DEFAULT_IMAGE_PREFIX}/{concept_id}{ConceptGraphConstants.DEFAULT_IMAGE_EXTENSION}"


def get_node_name_embedding_id(concept_id: str) -> str:
    """Get standardized embedding ID for node name."""
    return f"{concept_id}{ConceptGraphConstants.NODE_NAME_SEPARATOR}{ConceptGraphConstants.FIELD_NODE_NAME}"
