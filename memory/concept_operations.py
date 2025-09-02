"""Operations and visitors for concept graph processing."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import json
import logging

from .constants import ConceptGraphConstants


class IConceptVisitor(ABC):
    """Visitor interface for concept operations."""

    @abstractmethod
    def visit_concept(self, concept: Dict[str, Any]) -> Any:
        """Visit a concept node."""
        pass


class ConceptTextExtractor(IConceptVisitor):
    """Visitor to extract text representation from concepts for embeddings."""

    def __init__(self, include_refs: bool = False):
        self.include_refs = include_refs

    def visit_concept(self, concept: Dict[str, Any]) -> str:
        """Extract text representation of a concept (excludes refs by default)."""
        node_name = concept.get(ConceptGraphConstants.FIELD_NODE_NAME, "")
        node_type = concept.get(ConceptGraphConstants.FIELD_NODE_TYPE, "")
        node_attributes = concept.get(
            ConceptGraphConstants.FIELD_NODE_ATTRIBUTES, {})

        text = f"Node: {node_name}\nType: {node_type}\nAttributes: {json.dumps(node_attributes)}"

        # Include refs if requested (useful for certain operations)
        if self.include_refs:
            refs = concept.get(ConceptGraphConstants.FIELD_REFS, {})
            if refs:
                text += f"\nReferences: {json.dumps(refs)}"

        return text


class ConceptValidator(IConceptVisitor):
    """Visitor to validate concept data."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.validation_errors: List[str] = []

    def visit_concept(self, concept: Dict[str, Any]) -> bool:
        """Validate a concept and return True if valid."""
        self.validation_errors.clear()

        # Check required fields
        required_fields = [
            ConceptGraphConstants.FIELD_NODE_ID,
            ConceptGraphConstants.FIELD_NODE_NAME,
            ConceptGraphConstants.FIELD_NODE_TYPE,
            ConceptGraphConstants.FIELD_NODE_ATTRIBUTES,
            ConceptGraphConstants.FIELD_IS_EDITABLE
        ]

        for field in required_fields:
            if field not in concept:
                self.validation_errors.append(
                    f"Missing required field: {field}")

        # Validate field types
        if ConceptGraphConstants.FIELD_NODE_NAME in concept:
            if not isinstance(
                    concept[ConceptGraphConstants.FIELD_NODE_NAME], str):
                self.validation_errors.append("node_name must be a string")
            elif not concept[ConceptGraphConstants.FIELD_NODE_NAME].strip():
                self.validation_errors.append("node_name cannot be empty")

        if ConceptGraphConstants.FIELD_NODE_TYPE in concept:
            if not isinstance(
                    concept[ConceptGraphConstants.FIELD_NODE_TYPE], str):
                self.validation_errors.append("node_type must be a string")
            elif not concept[ConceptGraphConstants.FIELD_NODE_TYPE].strip():
                self.validation_errors.append("node_type cannot be empty")

        if ConceptGraphConstants.FIELD_NODE_ATTRIBUTES in concept:
            if not isinstance(
                    concept[ConceptGraphConstants.FIELD_NODE_ATTRIBUTES], dict):
                self.validation_errors.append(
                    "node_attributes must be a dictionary")

        if ConceptGraphConstants.FIELD_IS_EDITABLE in concept:
            if not isinstance(
                    concept[ConceptGraphConstants.FIELD_IS_EDITABLE], bool):
                self.validation_errors.append("is_editable must be a boolean")

        # Validate refs structure if present
        if ConceptGraphConstants.FIELD_REFS in concept:
            refs = concept[ConceptGraphConstants.FIELD_REFS]
            if not isinstance(refs, dict):
                self.validation_errors.append("refs must be a dictionary")
            else:
                expected_ref_types = [
                    ConceptGraphConstants.REF_IMG,
                    ConceptGraphConstants.REF_AUDIO,
                    ConceptGraphConstants.REF_VIDEO,
                    ConceptGraphConstants.REF_DOCS
                ]

                for ref_type in expected_ref_types:
                    if ref_type in refs and not isinstance(
                            refs[ref_type], list):
                        self.validation_errors.append(
                            f"refs.{ref_type} must be a list")

        if self.validation_errors:
            self.logger.warning(
                f"Concept validation failed: {', '.join(self.validation_errors)}")
            return False

        return True

    def get_validation_errors(self) -> List[str]:
        """Get list of validation errors from last validation."""
        return self.validation_errors.copy()


class ConceptStatisticsCollector(IConceptVisitor):
    """Visitor to collect statistics about concepts."""

    def __init__(self):
        self.stats = {
            "total_concepts": 0,
            "concept_types": {},
            "editable_count": 0,
            "with_images": 0,
            "with_refs": 0,
            "attribute_keys": set()
        }

    def visit_concept(self, concept: Dict[str, Any]) -> Dict[str, Any]:
        """Collect statistics from a concept."""
        self.stats["total_concepts"] += 1

        # Count by type
        concept_type = concept.get(
            ConceptGraphConstants.FIELD_NODE_TYPE, "unknown")
        self.stats["concept_types"][concept_type] = self.stats["concept_types"].get(
            concept_type, 0) + 1

        # Count editable
        if concept.get(ConceptGraphConstants.FIELD_IS_EDITABLE, False):
            self.stats["editable_count"] += 1

        # Count with images
        if concept.get(ConceptGraphConstants.FIELD_IMAGE_PATH):
            self.stats["with_images"] += 1

        # Count with refs
        refs = concept.get(ConceptGraphConstants.FIELD_REFS, {})
        if any(refs.get(ref_type, []) for ref_type in [
            ConceptGraphConstants.REF_IMG,
            ConceptGraphConstants.REF_AUDIO,
            ConceptGraphConstants.REF_VIDEO,
            ConceptGraphConstants.REF_DOCS
        ]):
            self.stats["with_refs"] += 1

        # Collect attribute keys
        attributes = concept.get(
            ConceptGraphConstants.FIELD_NODE_ATTRIBUTES, {})
        if isinstance(attributes, dict):
            self.stats["attribute_keys"].update(attributes.keys())

        return self.get_statistics()

    def get_statistics(self) -> Dict[str, Any]:
        """Get collected statistics."""
        stats_copy = self.stats.copy()
        stats_copy["attribute_keys"] = list(self.stats["attribute_keys"])
        return stats_copy


class ConceptSearcher(IConceptVisitor):
    """Visitor to search concepts by criteria."""

    def __init__(self, search_criteria: Dict[str, Any]):
        self.search_criteria = search_criteria
        self.results: List[Dict[str, Any]] = []

    def visit_concept(self, concept: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search concept against criteria."""
        if self._matches_criteria(concept):
            self.results.append(concept)
        return self.results

    def _matches_criteria(self, concept: Dict[str, Any]) -> bool:
        """Check if concept matches search criteria."""
        for key, value in self.search_criteria.items():
            if key == ConceptGraphConstants.FIELD_NODE_TYPE:
                if concept.get(key) != value:
                    return False
            elif key == ConceptGraphConstants.FIELD_NODE_NAME:
                concept_name = concept.get(key, "")
                if isinstance(value, str) and value.lower(
                ) not in concept_name.lower():
                    return False
            elif key == ConceptGraphConstants.FIELD_IS_EDITABLE:
                if concept.get(key) != value:
                    return False
            elif key in concept.get(ConceptGraphConstants.FIELD_NODE_ATTRIBUTES, {}):
                if concept[ConceptGraphConstants.FIELD_NODE_ATTRIBUTES][key] != value:
                    return False
        return True

    def get_results(self) -> List[Dict[str, Any]]:
        """Get search results."""
        return self.results.copy()


class ConceptOperations:
    """Utility class for common concept operations using visitors."""

    @staticmethod
    def extract_text(concept: Dict[str, Any],
                     include_refs: bool = False) -> str:
        """Extract text representation from a concept."""
        extractor = ConceptTextExtractor(include_refs=include_refs)
        return extractor.visit_concept(concept)

    @staticmethod
    def validate_concept(
            concept: Dict[str, Any], logger: Optional[logging.Logger] = None) -> bool:
        """Validate a concept."""
        validator = ConceptValidator(logger)
        return validator.visit_concept(concept)

    @staticmethod
    def get_validation_errors(
            concept: Dict[str, Any], logger: Optional[logging.Logger] = None) -> List[str]:
        """Get validation errors for a concept."""
        validator = ConceptValidator(logger)
        validator.visit_concept(concept)
        return validator.get_validation_errors()

    @staticmethod
    def collect_statistics(concepts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collect statistics from a list of concepts."""
        collector = ConceptStatisticsCollector()
        for concept in concepts:
            collector.visit_concept(concept)
        return collector.get_statistics()

    @staticmethod
    def search_concepts(
            concepts: List[Dict[str, Any]], criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search concepts by criteria."""
        searcher = ConceptSearcher(criteria)
        for concept in concepts:
            searcher.visit_concept(concept)
        return searcher.get_results()


class ConceptBuilder:
    """Builder pattern for creating concept dictionaries with proper validation."""

    def __init__(self, concept_id: str, concept_name: str, concept_type: str):
        self.concept = {
            ConceptGraphConstants.FIELD_NODE_ID: concept_id,
            ConceptGraphConstants.FIELD_NODE_NAME: concept_name,
            ConceptGraphConstants.FIELD_NODE_TYPE: concept_type,
            ConceptGraphConstants.FIELD_NODE_ATTRIBUTES: {},
            ConceptGraphConstants.FIELD_IS_EDITABLE: True,
            ConceptGraphConstants.FIELD_REFS: ConceptGraphConstants.DEFAULT_REFS.copy(),
            ConceptGraphConstants.FIELD_IMAGE_PATH: None}

    def with_attributes(self, attributes: Dict[str, Any]) -> 'ConceptBuilder':
        """Add attributes to the concept."""
        if not isinstance(attributes, dict):
            raise ValueError("Attributes must be a dictionary")
        self.concept[ConceptGraphConstants.FIELD_NODE_ATTRIBUTES] = attributes
        return self

    def with_editability(self, is_editable: bool) -> 'ConceptBuilder':
        """Set concept editability."""
        self.concept[ConceptGraphConstants.FIELD_IS_EDITABLE] = is_editable
        return self

    def with_image_path(self, image_path: str) -> 'ConceptBuilder':
        """Add image path to the concept."""
        self.concept[ConceptGraphConstants.FIELD_IMAGE_PATH] = image_path
        return self

    def with_refs(self, refs: Dict[str, List[str]]) -> 'ConceptBuilder':
        """Add references to the concept."""
        if not isinstance(refs, dict):
            raise ValueError("Refs must be a dictionary")

        # Merge with defaults to ensure all ref types exist
        merged_refs = ConceptGraphConstants.DEFAULT_REFS.copy()
        merged_refs.update(refs)
        self.concept[ConceptGraphConstants.FIELD_REFS] = merged_refs
        return self

    def build(self, validate: bool = True) -> Dict[str, Any]:
        """Build and optionally validate the concept."""
        if validate:
            if not ConceptOperations.validate_concept(self.concept):
                errors = ConceptOperations.get_validation_errors(self.concept)
                raise ValueError(f"Invalid concept: {', '.join(errors)}")

        return self.concept.copy()
