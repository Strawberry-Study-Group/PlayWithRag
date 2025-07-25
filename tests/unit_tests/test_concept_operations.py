import pytest
from unittest.mock import Mock
import logging

from concept_graph.concept_operations import (
    ConceptTextExtractor,
    ConceptValidator,
    ConceptStatisticsCollector,
    ConceptSearcher,
    ConceptOperations,
    ConceptBuilder
)
from concept_graph.constants import ConceptGraphConstants


class TestConceptTextExtractor:
    
    @pytest.fixture
    def sample_concept(self):
        return {
            ConceptGraphConstants.FIELD_NODE_ID: "test1",
            ConceptGraphConstants.FIELD_NODE_NAME: "Hero",
            ConceptGraphConstants.FIELD_NODE_TYPE: "character",
            ConceptGraphConstants.FIELD_NODE_ATTRIBUTES: {"level": 1, "health": 100},
            ConceptGraphConstants.FIELD_REFS: {
                "ref_img": ["hero.jpg"],
                "ref_audio": ["hero_voice.mp3"],
                "ref_video": [],
                "ref_docs": []
            }
        }
    
    def test_extract_text_excludes_refs_by_default(self, sample_concept):
        extractor = ConceptTextExtractor()
        text = extractor.visit_concept(sample_concept)
        
        assert "Hero" in text
        assert "character" in text
        assert "level" in text
        assert "refs" not in text
        assert "hero.jpg" not in text
        assert "hero_voice.mp3" not in text
    
    def test_extract_text_includes_refs_when_requested(self, sample_concept):
        extractor = ConceptTextExtractor(include_refs=True)
        text = extractor.visit_concept(sample_concept)
        
        assert "Hero" in text
        assert "character" in text
        assert "level" in text
        assert "References:" in text
        assert "hero.jpg" in text
        assert "hero_voice.mp3" in text


class TestConceptValidator:
    
    @pytest.fixture
    def mock_logger(self):
        return Mock(spec=logging.Logger)
    
    def test_validate_valid_concept(self, mock_logger):
        validator = ConceptValidator(mock_logger)
        
        valid_concept = {
            ConceptGraphConstants.FIELD_NODE_ID: "test1",
            ConceptGraphConstants.FIELD_NODE_NAME: "Hero",
            ConceptGraphConstants.FIELD_NODE_TYPE: "character",
            ConceptGraphConstants.FIELD_NODE_ATTRIBUTES: {"level": 1},
            ConceptGraphConstants.FIELD_IS_EDITABLE: True
        }
        
        assert validator.visit_concept(valid_concept) is True
        assert len(validator.get_validation_errors()) == 0
    
    def test_validate_missing_required_field(self, mock_logger):
        validator = ConceptValidator(mock_logger)
        
        invalid_concept = {
            ConceptGraphConstants.FIELD_NODE_ID: "test1",
            ConceptGraphConstants.FIELD_NODE_NAME: "Hero"
            # Missing required fields
        }
        
        assert validator.visit_concept(invalid_concept) is False
        errors = validator.get_validation_errors()
        assert len(errors) > 0
        assert any("Missing required field" in error for error in errors)
    
    def test_validate_invalid_field_types(self, mock_logger):
        validator = ConceptValidator(mock_logger)
        
        invalid_concept = {
            ConceptGraphConstants.FIELD_NODE_ID: "test1",
            ConceptGraphConstants.FIELD_NODE_NAME: 123,  # Should be string
            ConceptGraphConstants.FIELD_NODE_TYPE: "character",
            ConceptGraphConstants.FIELD_NODE_ATTRIBUTES: "not_a_dict",  # Should be dict
            ConceptGraphConstants.FIELD_IS_EDITABLE: "not_a_bool"  # Should be bool
        }
        
        assert validator.visit_concept(invalid_concept) is False
        errors = validator.get_validation_errors()
        assert len(errors) >= 3  # At least 3 type errors


class TestConceptStatisticsCollector:
    
    def test_collect_statistics_single_concept(self):
        collector = ConceptStatisticsCollector()
        
        concept = {
            ConceptGraphConstants.FIELD_NODE_ID: "test1",
            ConceptGraphConstants.FIELD_NODE_NAME: "Hero",
            ConceptGraphConstants.FIELD_NODE_TYPE: "character",
            ConceptGraphConstants.FIELD_NODE_ATTRIBUTES: {"level": 1, "health": 100},
            ConceptGraphConstants.FIELD_IS_EDITABLE: True,
            ConceptGraphConstants.FIELD_IMAGE_PATH: "hero.jpg",
            ConceptGraphConstants.FIELD_REFS: {
                "ref_img": ["img1.jpg"],
                "ref_audio": [],
                "ref_video": [],
                "ref_docs": []
            }
        }
        
        stats = collector.visit_concept(concept)
        
        assert stats["total_concepts"] == 1
        assert stats["concept_types"]["character"] == 1
        assert stats["editable_count"] == 1
        assert stats["with_images"] == 1
        assert stats["with_refs"] == 1
        assert "level" in stats["attribute_keys"]
        assert "health" in stats["attribute_keys"]
    
    def test_collect_statistics_multiple_concepts(self):
        collector = ConceptStatisticsCollector()
        
        concepts = [
            {
                ConceptGraphConstants.FIELD_NODE_ID: "test1",
                ConceptGraphConstants.FIELD_NODE_NAME: "Hero",
                ConceptGraphConstants.FIELD_NODE_TYPE: "character",
                ConceptGraphConstants.FIELD_NODE_ATTRIBUTES: {"level": 1},
                ConceptGraphConstants.FIELD_IS_EDITABLE: True
            },
            {
                ConceptGraphConstants.FIELD_NODE_ID: "test2",
                ConceptGraphConstants.FIELD_NODE_NAME: "Sword",
                ConceptGraphConstants.FIELD_NODE_TYPE: "item",
                ConceptGraphConstants.FIELD_NODE_ATTRIBUTES: {"damage": 10},
                ConceptGraphConstants.FIELD_IS_EDITABLE: False
            }
        ]
        
        for concept in concepts:
            collector.visit_concept(concept)
        
        stats = collector.get_statistics()
        assert stats["total_concepts"] == 2
        assert stats["concept_types"]["character"] == 1
        assert stats["concept_types"]["item"] == 1
        assert stats["editable_count"] == 1


class TestConceptSearcher:
    
    def test_search_by_concept_type(self):
        searcher = ConceptSearcher({ConceptGraphConstants.FIELD_NODE_TYPE: "character"})
        
        concepts = [
            {
                ConceptGraphConstants.FIELD_NODE_ID: "test1",
                ConceptGraphConstants.FIELD_NODE_NAME: "Hero",
                ConceptGraphConstants.FIELD_NODE_TYPE: "character",
                ConceptGraphConstants.FIELD_NODE_ATTRIBUTES: {"level": 1},
                ConceptGraphConstants.FIELD_IS_EDITABLE: True
            },
            {
                ConceptGraphConstants.FIELD_NODE_ID: "test2",
                ConceptGraphConstants.FIELD_NODE_NAME: "Sword",
                ConceptGraphConstants.FIELD_NODE_TYPE: "item",
                ConceptGraphConstants.FIELD_NODE_ATTRIBUTES: {"damage": 10},
                ConceptGraphConstants.FIELD_IS_EDITABLE: True
            }
        ]
        
        for concept in concepts:
            searcher.visit_concept(concept)
        
        results = searcher.get_results()
        assert len(results) == 1
        assert results[0][ConceptGraphConstants.FIELD_NODE_NAME] == "Hero"
    
    def test_search_by_name_partial_match(self):
        searcher = ConceptSearcher({ConceptGraphConstants.FIELD_NODE_NAME: "her"})
        
        concept = {
            ConceptGraphConstants.FIELD_NODE_ID: "test1",
            ConceptGraphConstants.FIELD_NODE_NAME: "Hero",
            ConceptGraphConstants.FIELD_NODE_TYPE: "character",
            ConceptGraphConstants.FIELD_NODE_ATTRIBUTES: {"level": 1},
            ConceptGraphConstants.FIELD_IS_EDITABLE: True
        }
        
        searcher.visit_concept(concept)
        results = searcher.get_results()
        assert len(results) == 1
        assert results[0][ConceptGraphConstants.FIELD_NODE_NAME] == "Hero"


class TestConceptOperations:
    
    def test_extract_text_static_method(self):
        concept = {
            ConceptGraphConstants.FIELD_NODE_NAME: "Hero",
            ConceptGraphConstants.FIELD_NODE_TYPE: "character",
            ConceptGraphConstants.FIELD_NODE_ATTRIBUTES: {"level": 1}
        }
        
        text = ConceptOperations.extract_text(concept)
        assert "Hero" in text
        assert "character" in text
        assert "level" in text
    
    def test_validate_concept_static_method(self):
        valid_concept = {
            ConceptGraphConstants.FIELD_NODE_ID: "test1",
            ConceptGraphConstants.FIELD_NODE_NAME: "Hero",
            ConceptGraphConstants.FIELD_NODE_TYPE: "character",
            ConceptGraphConstants.FIELD_NODE_ATTRIBUTES: {"level": 1},
            ConceptGraphConstants.FIELD_IS_EDITABLE: True
        }
        
        assert ConceptOperations.validate_concept(valid_concept) is True
    
    def test_collect_statistics_static_method(self):
        concepts = [
            {
                ConceptGraphConstants.FIELD_NODE_ID: "test1",
                ConceptGraphConstants.FIELD_NODE_NAME: "Hero",
                ConceptGraphConstants.FIELD_NODE_TYPE: "character",
                ConceptGraphConstants.FIELD_NODE_ATTRIBUTES: {"level": 1},
                ConceptGraphConstants.FIELD_IS_EDITABLE: True
            }
        ]
        
        stats = ConceptOperations.collect_statistics(concepts)
        assert stats["total_concepts"] == 1
        assert stats["concept_types"]["character"] == 1
    
    def test_search_concepts_static_method(self):
        concepts = [
            {
                ConceptGraphConstants.FIELD_NODE_ID: "test1",
                ConceptGraphConstants.FIELD_NODE_NAME: "Hero",
                ConceptGraphConstants.FIELD_NODE_TYPE: "character",
                ConceptGraphConstants.FIELD_NODE_ATTRIBUTES: {"level": 1},
                ConceptGraphConstants.FIELD_IS_EDITABLE: True
            },
            {
                ConceptGraphConstants.FIELD_NODE_ID: "test2",
                ConceptGraphConstants.FIELD_NODE_NAME: "Sword",
                ConceptGraphConstants.FIELD_NODE_TYPE: "item",
                ConceptGraphConstants.FIELD_NODE_ATTRIBUTES: {"damage": 10},
                ConceptGraphConstants.FIELD_IS_EDITABLE: True
            }
        ]
        
        results = ConceptOperations.search_concepts(concepts, {"node_type": "character"})
        assert len(results) == 1
        assert results[0][ConceptGraphConstants.FIELD_NODE_NAME] == "Hero"


class TestConceptBuilder:
    
    def test_build_basic_concept(self):
        builder = ConceptBuilder("test1", "Hero", "character")
        concept = builder.build()
        
        assert concept[ConceptGraphConstants.FIELD_NODE_ID] == "test1"
        assert concept[ConceptGraphConstants.FIELD_NODE_NAME] == "Hero"
        assert concept[ConceptGraphConstants.FIELD_NODE_TYPE] == "character"
        assert concept[ConceptGraphConstants.FIELD_NODE_ATTRIBUTES] == {}
        assert concept[ConceptGraphConstants.FIELD_IS_EDITABLE] is True
        assert concept[ConceptGraphConstants.FIELD_REFS] == ConceptGraphConstants.DEFAULT_REFS
    
    def test_build_concept_with_attributes(self):
        builder = ConceptBuilder("test1", "Hero", "character")
        concept = builder.with_attributes({"level": 1, "health": 100}).build()
        
        assert concept[ConceptGraphConstants.FIELD_NODE_ATTRIBUTES]["level"] == 1
        assert concept[ConceptGraphConstants.FIELD_NODE_ATTRIBUTES]["health"] == 100
    
    def test_build_concept_with_invalid_attributes(self):
        builder = ConceptBuilder("test1", "Hero", "character")
        
        with pytest.raises(ValueError, match="Attributes must be a dictionary"):
            builder.with_attributes("invalid_attributes")
    
    def test_build_concept_with_refs(self):
        builder = ConceptBuilder("test1", "Hero", "character")
        refs = {"ref_img": ["hero.jpg"], "ref_audio": ["hero_voice.mp3"]}
        concept = builder.with_refs(refs).build()
        
        assert concept[ConceptGraphConstants.FIELD_REFS]["ref_img"] == ["hero.jpg"]
        assert concept[ConceptGraphConstants.FIELD_REFS]["ref_audio"] == ["hero_voice.mp3"]
        # Should still have defaults for other ref types
        assert concept[ConceptGraphConstants.FIELD_REFS]["ref_video"] == []
        assert concept[ConceptGraphConstants.FIELD_REFS]["ref_docs"] == []
    
    def test_build_concept_with_image_path(self):
        builder = ConceptBuilder("test1", "Hero", "character")
        concept = builder.with_image_path("hero.jpg").build()
        
        assert concept[ConceptGraphConstants.FIELD_IMAGE_PATH] == "hero.jpg"
    
    def test_build_concept_validation_failure(self):
        builder = ConceptBuilder("", "", "")  # Invalid empty values
        
        with pytest.raises(ValueError, match="Invalid concept"):
            builder.build(validate=True)
    
    def test_build_concept_skip_validation(self):
        builder = ConceptBuilder("", "", "")  # Invalid empty values
        
        # Should not raise error when validation is skipped
        concept = builder.build(validate=False)
        assert concept[ConceptGraphConstants.FIELD_NODE_NAME] == ""