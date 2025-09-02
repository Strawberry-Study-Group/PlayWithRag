"""Comprehensive large-scale system test for memory core functionality."""

import pytest
import sys
import os
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from memory.memory import MemoryCoreFactory, MemoryCoreService
from memory.memory_core_schema import MemoryCoreInitializer
from tests.test_memory_core_utils import create_memory_core_config


class TestComprehensiveLargeScale:
    """Comprehensive test with 1000+ nodes and edges for large-scale validation."""
    
    def test_comprehensive_large_memory_core(self):
        """Create and test a comprehensive memory core with 1000+ nodes and edges."""
        # Import OpenAI API key from test config
        sys.path.append(str(Path(__file__).parent.parent))
        from test_config import OPENAI_API_KEY, has_valid_openai_key
        
        if not has_valid_openai_key():
            pytest.skip("Valid OpenAI API key not configured in test_config.py")
        
        openai_api_key = OPENAI_API_KEY
        
        print("\n🚀 Starting comprehensive large-scale memory core test...")
        start_time = time.time()
        
        # Create persistent memory core in system test directory (won't be deleted)
        system_test_dir = Path(__file__).parent
        memory_core_dir = system_test_dir / "test_memory_cores"
        memory_core_dir.mkdir(exist_ok=True)
        
        # Create unique memory core name with timestamp
        timestamp = int(time.time())
        memory_core_path = str(memory_core_dir / f"comprehensive_memory_core_{timestamp}")
        
        print(f"📁 Memory core will be created at: {memory_core_path}")
        
        # Initialize memory core with proper structure
        initializer = MemoryCoreInitializer()
        initializer.initialize_memory_core(
            memory_core_path=memory_core_path,
            api_key=openai_api_key,
            provider="local",
            model="text-embedding-3-small",
            dim=1536,
            validate=True
        )
        
        # Create memory core service
        memory_core = MemoryCoreFactory.create_from_memory_core(memory_core_path)
        memory_core.empty_graph()  # Start clean
        
        print("✅ Memory core initialized successfully")
        
        # Phase 1: Create Fantasy World Concepts (Characters, Locations, Items)
        print("\n📚 Phase 1: Creating fantasy world concepts...")
        character_ids = self._create_fantasy_characters(memory_core)
        location_ids = self._create_fantasy_locations(memory_core) 
        item_ids = self._create_fantasy_items(memory_core)
        skill_ids = self._create_skills_and_abilities(memory_core)
        
        print(f"   Created {len(character_ids)} characters")
        print(f"   Created {len(location_ids)} locations") 
        print(f"   Created {len(item_ids)} items")
        print(f"   Created {len(skill_ids)} skills")
        
        # Phase 2: Create Modern World Concepts (People, Organizations, Technologies)
        print("\n🏢 Phase 2: Creating modern world concepts...")
        person_ids = self._create_modern_people(memory_core)
        org_ids = self._create_organizations(memory_core)
        tech_ids = self._create_technologies(memory_core)
        
        print(f"   Created {len(person_ids)} modern people")
        print(f"   Created {len(org_ids)} organizations")
        print(f"   Created {len(tech_ids)} technologies")
        
        # Phase 3: Create Scientific Concepts (Research, Discoveries, Theories)  
        print("\n🔬 Phase 3: Creating scientific concepts...")
        research_ids = self._create_research_topics(memory_core)
        discovery_ids = self._create_discoveries(memory_core)
        theory_ids = self._create_theories(memory_core)
        
        print(f"   Created {len(research_ids)} research topics")
        print(f"   Created {len(discovery_ids)} discoveries")
        print(f"   Created {len(theory_ids)} theories")
        
        # Phase 4: Create Historical Concepts (Events, Periods, Figures)
        print("\n📜 Phase 4: Creating historical concepts...")
        event_ids = self._create_historical_events(memory_core)
        period_ids = self._create_historical_periods(memory_core)
        figure_ids = self._create_historical_figures(memory_core)
        
        print(f"   Created {len(event_ids)} historical events")
        print(f"   Created {len(period_ids)} historical periods") 
        print(f"   Created {len(figure_ids)} historical figures")
        
        # Phase 5: Create Complex Relationships
        print("\n🔗 Phase 5: Creating complex relationships...")
        total_relations = 0
        
        # Fantasy world relationships
        total_relations += self._create_fantasy_relationships(memory_core, character_ids, location_ids, item_ids, skill_ids)
        
        # Modern world relationships  
        total_relations += self._create_modern_relationships(memory_core, person_ids, org_ids, tech_ids)
        
        # Scientific relationships
        total_relations += self._create_scientific_relationships(memory_core, research_ids, discovery_ids, theory_ids, person_ids)
        
        # Historical relationships
        total_relations += self._create_historical_relationships(memory_core, event_ids, period_ids, figure_ids)
        
        # Cross-domain relationships (most interesting!)
        total_relations += self._create_cross_domain_relationships(memory_core, {
            'characters': character_ids, 'people': person_ids, 'figures': figure_ids,
            'locations': location_ids, 'organizations': org_ids,
            'items': item_ids, 'technologies': tech_ids,
            'skills': skill_ids, 'research': research_ids,
            'events': event_ids, 'discoveries': discovery_ids
        })
        
        print(f"   Created {total_relations} total relationships")
        
        # Phase 6: Comprehensive Testing
        print("\n🧪 Phase 6: Comprehensive functionality testing...")
        self._test_similarity_search(memory_core)
        self._test_relationship_traversal(memory_core, character_ids, location_ids)
        self._test_complex_queries(memory_core)
        self._test_graph_statistics(memory_core)
        
        # Phase 7: Performance and Persistence Testing
        print("\n⚡ Phase 7: Performance and persistence testing...")
        self._test_save_and_load_performance(memory_core)
        self._test_large_scale_operations(memory_core)
        
        # Final statistics and validation
        stats = memory_core.get_graph_statistics()
        total_nodes = stats["total_nodes"]
        total_edges = stats["total_edges"]
        
        print(f"\n📊 Final Statistics:")
        print(f"   Total nodes: {total_nodes}")
        print(f"   Total edges: {total_edges}")
        print(f"   Node types: {len(stats['node_types'])}")
        print(f"   Edge types: {len(stats['edge_types'])}")
        print(f"   Memory core size: {self._get_directory_size(memory_core_path):.2f} MB")
        
        elapsed_time = time.time() - start_time
        print(f"   Total execution time: {elapsed_time:.2f} seconds")
        print(f"   Nodes per second: {total_nodes/elapsed_time:.2f}")
        print(f"   Edges per second: {total_edges/elapsed_time:.2f}")
        
        # Validation assertions
        assert total_nodes >= 1000, f"Expected at least 1000 nodes, got {total_nodes}"
        assert total_edges >= 1000, f"Expected at least 1000 edges, got {total_edges}"
        assert len(stats["node_types"]) >= 10, f"Expected diverse node types, got {len(stats['node_types'])}"
        
        print(f"\n✅ Comprehensive test completed successfully!")
        print(f"🔍 Memory core preserved for inspection at: {memory_core_path}")
        print(f"📝 You can explore the following directories:")
        print(f"   - {memory_core_path}/config.json (configuration)")
        print(f"   - {memory_core_path}/data/graph.json (graph data)")
        print(f"   - {memory_core_path}/data/emb_index.json (embeddings)")
        print(f"   - {memory_core_path}/assets/img/ (images directory)")
        
        return memory_core_path  # Return path for external inspection
    
    def _create_fantasy_characters(self, memory_core: MemoryCoreService) -> List[str]:
        """Create diverse fantasy characters with rich attributes."""
        character_data = [
            # Heroes and Protagonists
            ("Aeliana Starweaver", "elf_mage", {"class": "archmage", "alignment": "neutral_good", "level": 20, "specialization": "celestial_magic", "homeland": "Silverleaf_Forest", "age": 342}),
            ("Thorgan Ironbeard", "dwarf_warrior", {"class": "paladin", "alignment": "lawful_good", "level": 18, "weapon": "blessed_warhammer", "clan": "Ironbeard", "age": 156}),
            ("Zara Shadowstep", "human_rogue", {"class": "assassin", "alignment": "chaotic_neutral", "level": 15, "guild": "Silent_Daggers", "homeland": "Port_Blackwater", "age": 28}),
            ("Finn Lightbringer", "halfling_cleric", {"class": "high_priest", "alignment": "lawful_good", "level": 16, "deity": "Pelor", "homeland": "Green_Hills", "age": 87}),
            
            # Villains and Antagonists  
            ("Malachar the Corrupt", "human_necromancer", {"class": "lich", "alignment": "chaotic_evil", "level": 22, "undead_army": "Legion_of_Bones", "fortress": "Tower_of_Screams", "age": 1200}),
            ("Grimjaw Bloodfang", "orc_warlord", {"class": "barbarian", "alignment": "chaotic_evil", "level": 17, "tribe": "Bloodfang_Clan", "weapon": "Soulsplitter_Axe", "age": 45}),
            ("Lady Vespera Nightshade", "vampire_noble", {"class": "aristocrat", "alignment": "lawful_evil", "level": 19, "estate": "Ravenshollow_Manor", "servants": "thrall_network", "age": 800}),
            ("Thaxon the Destroyer", "dragon_ancient", {"class": "sorcerer", "alignment": "chaotic_evil", "level": 25, "type": "red_dragon", "hoard": "volcanic_treasures", "age": 2100}),
            
            # Supporting Characters
            ("Elder Oakenheart", "treant_guardian", {"class": "druid", "alignment": "neutral", "level": 20, "forest": "Ancient_Grove", "duty": "forest_protector", "age": 1500}),
            ("Captain Marcus Steelwind", "human_fighter", {"class": "knight", "alignment": "lawful_neutral", "level": 14, "unit": "Silver_Hawks", "homeland": "Valorian_Empire", "age": 42}),
        ]
        
        # Expand with generated characters to reach target count
        generated_chars = []
        races = ["elf", "dwarf", "human", "halfling", "gnome", "tiefling", "dragonborn", "orc", "goblin", "centaur"]
        classes = ["warrior", "mage", "rogue", "cleric", "ranger", "bard", "monk", "sorcerer", "warlock", "artificer"]
        alignments = ["lawful_good", "neutral_good", "chaotic_good", "lawful_neutral", "neutral", "chaotic_neutral", "lawful_evil", "neutral_evil", "chaotic_evil"]
        
        for i in range(140):  # Generate 140 more characters
            race = races[i % len(races)]
            char_class = classes[i % len(classes)]
            alignment = alignments[i % len(alignments)]
            
            name = f"{self._generate_fantasy_name()}_{i} {self._generate_fantasy_surname()}"
            concept_type = f"{race}_{char_class}"
            attributes = {
                "class": char_class,
                "race": race,
                "alignment": alignment,
                "level": (i % 20) + 1,
                "homeland": self._generate_place_name(),
                "age": (i * 7 + 25) % 500 + 20,
                "background": self._generate_background(),
                "notable_feature": self._generate_feature()
            }
            generated_chars.append((name, concept_type, attributes))
        
        # Create all characters in memory core
        character_ids = []
        all_characters = character_data + generated_chars
        
        for name, concept_type, attributes in all_characters:
            char_id = memory_core.add_concept(name, concept_type, attributes)
            character_ids.append(char_id)
        
        return character_ids
    
    def _create_fantasy_locations(self, memory_core: MemoryCoreService) -> List[str]:
        """Create diverse fantasy locations."""
        location_data = [
            # Major Cities
            ("Eldermoor", "major_city", {"population": 150000, "government": "council_of_mages", "specialization": "magical_research", "climate": "temperate", "defenses": "ward_barriers"}),
            ("Port Blackwater", "port_city", {"population": 80000, "government": "merchant_guild", "specialization": "maritime_trade", "climate": "coastal", "defenses": "naval_fleet"}),
            ("Ironhold Citadel", "fortress_city", {"population": 45000, "government": "military_command", "specialization": "weapon_forging", "climate": "mountainous", "defenses": "stone_walls"}),
            
            # Natural Locations
            ("Whispering Woods", "enchanted_forest", {"area": "vast", "magical_level": "high", "inhabitants": "fey_creatures", "climate": "mystical", "dangers": "illusion_magic"}),
            ("Dragon's Spine Mountains", "mountain_range", {"height": "towering", "resources": "precious_metals", "inhabitants": "dragons_dwarves", "climate": "harsh", "dangers": "avalanches_dragons"}),
            ("Sunless Sea", "cursed_waters", {"depth": "unfathomable", "magical_level": "dark", "inhabitants": "undead_pirates", "climate": "perpetual_storm", "dangers": "ghost_ships"}),
            
            # Dungeons and Ruins
            ("Tomb of the Forgotten King", "ancient_tomb", {"age": "millennia", "magical_level": "necromantic", "treasures": "royal_regalia", "climate": "cold_damp", "dangers": "undead_guardians"}),
            ("Crystal Caverns", "magical_dungeon", {"depth": "deep", "magical_level": "elemental", "treasures": "magic_crystals", "climate": "varies", "dangers": "crystal_golems"}),
        ]
        
        # Generate additional locations
        location_types = ["village", "town", "castle", "tower", "cave", "temple", "ruin", "swamp", "desert_oasis", "floating_island"]
        for i in range(92):  # Generate 92 more locations
            name = f"{self._generate_place_name()}_{i} {self._generate_location_suffix()}"
            loc_type = location_types[i % len(location_types)]
            attributes = {
                "size": ["tiny", "small", "medium", "large", "huge"][i % 5],
                "population": (i * 100 + 50) if loc_type in ["village", "town", "city"] else 0,
                "magical_level": ["none", "low", "medium", "high", "extreme"][i % 5],
                "climate": ["temperate", "cold", "hot", "tropical", "arctic", "desert"][i % 6],
                "notable_feature": self._generate_location_feature(),
                "accessibility": ["easy", "moderate", "difficult", "treacherous"][i % 4]
            }
            location_data.append((name, loc_type, attributes))
        
        location_ids = []
        for name, concept_type, attributes in location_data:
            loc_id = memory_core.add_concept(name, concept_type, attributes)
            location_ids.append(loc_id)
        
        return location_ids
    
    def _create_fantasy_items(self, memory_core: MemoryCoreService) -> List[str]:
        """Create magical items and equipment."""
        item_data = [
            # Legendary Weapons
            ("Dawnbreaker", "legendary_sword", {"damage": "3d6+5", "material": "celestial_steel", "enchantment": "radiant_burst", "rarity": "legendary", "creator": "Solarian_Smiths"}),
            ("Staff of Storms", "legendary_staff", {"damage": "2d8+3", "material": "storm_crystal", "enchantment": "lightning_mastery", "rarity": "legendary", "creator": "Archmage_Tempestas"}),
            ("Shadowfang Dagger", "artifact_dagger", {"damage": "1d6+4", "material": "void_steel", "enchantment": "shadow_step", "rarity": "artifact", "creator": "unknown"}),
            
            # Magical Armor
            ("Aegis of the Phoenix", "legendary_armor", {"ac_bonus": 8, "material": "phoenix_feather_steel", "enchantment": "fire_immunity", "rarity": "legendary", "creator": "Phoenix_Order"}),
            ("Cloak of Starlight", "rare_cloak", {"ac_bonus": 2, "material": "starweave_silk", "enchantment": "invisibility", "rarity": "rare", "creator": "Night_Elves"}),
            
            # Magical Accessories  
            ("Ring of Elemental Mastery", "artifact_ring", {"bonus": "spell_power", "material": "elemental_gold", "enchantment": "element_control", "rarity": "artifact", "creator": "Elemental_Lords"}),
            ("Amulet of True Sight", "rare_amulet", {"bonus": "perception", "material": "crystal_silver", "enchantment": "detect_illusion", "rarity": "rare", "creator": "Order_of_Truth"}),
        ]
        
        # Generate additional items
        item_types = ["sword", "bow", "staff", "armor", "shield", "ring", "amulet", "potion", "scroll", "wand"]
        materials = ["steel", "mithril", "adamantine", "silver", "gold", "crystal", "bone", "wood", "leather", "cloth"]
        enchantments = ["sharpness", "protection", "speed", "strength", "wisdom", "fire", "ice", "lightning", "healing", "luck"]
        
        for i in range(143):  # Generate 143 more items  
            item_type = item_types[i % len(item_types)]
            material = materials[i % len(materials)]
            enchantment = enchantments[i % len(enchantments)]
            
            name = f"{enchantment.title()} {item_type.title()}_{i} of {material.title()}"
            rarity = ["common", "uncommon", "rare", "very_rare", "legendary"][min(i // 20, 4)]
            
            attributes = {
                "type": item_type,
                "material": material,
                "enchantment": enchantment,
                "rarity": rarity,
                "value": (i + 1) * 100,
                "weight": (i % 10) + 1,
                "creator": self._generate_creator_name(),
                "history": self._generate_item_history()
            }
            item_data.append((name, f"{rarity}_{item_type}", attributes))
        
        item_ids = []
        for name, concept_type, attributes in item_data:
            item_id = memory_core.add_concept(name, concept_type, attributes)
            item_ids.append(item_id)
        
        return item_ids
    
    def _create_skills_and_abilities(self, memory_core: MemoryCoreService) -> List[str]:
        """Create skills, spells, and abilities."""
        skill_data = [
            # Combat Abilities
            ("Whirlwind Strike", "combat_ability", {"type": "melee", "damage": "multi_target", "cooldown": "3_rounds", "requirements": "warrior_level_10"}),
            ("Fireball", "spell", {"type": "evocation", "damage": "6d6", "range": "150_feet", "requirements": "mage_level_5"}),
            ("Shadow Clone", "ninja_technique", {"type": "illusion", "effect": "duplicate_creation", "duration": "10_minutes", "requirements": "rogue_level_8"}),
            
            # Utility Skills
            ("Master Lockpicking", "skill", {"type": "utility", "effect": "unlock_any_lock", "difficulty": "master", "requirements": "dexterity_18"}),
            ("Ancient Lore", "knowledge", {"type": "academic", "effect": "historical_knowledge", "specialization": "archaeology", "requirements": "intelligence_16"}),
            ("Beast Speech", "ability", {"type": "communication", "effect": "talk_to_animals", "duration": "permanent", "requirements": "ranger_level_3"}),
        ]
        
        # Generate more skills
        skill_types = ["combat", "magic", "utility", "social", "crafting", "survival", "academic", "artistic", "athletic", "stealth"]
        for i in range(94):  # Generate 94 more skills
            skill_type = skill_types[i % len(skill_types)]
            name = f"{self._generate_skill_prefix()} {skill_type.title()}_{i}"
            
            attributes = {
                "type": skill_type,
                "difficulty": ["novice", "apprentice", "journeyman", "expert", "master"][i % 5],
                "requirements": f"level_{(i % 20) + 1}",
                "description": self._generate_skill_description(skill_type),
                "training_time": f"{(i % 12) + 1}_months"
            }
            skill_data.append((name, f"{skill_type}_skill", attributes))
        
        skill_ids = []
        for name, concept_type, attributes in skill_data:
            skill_id = memory_core.add_concept(name, concept_type, attributes)
            skill_ids.append(skill_id)
        
        return skill_ids
    
    def _create_modern_people(self, memory_core: MemoryCoreService) -> List[str]:
        """Create modern people with diverse backgrounds."""
        people_data = [
            # Tech Industry
            ("Dr. Sarah Chen", "tech_executive", {"company": "NeuroLink_Corp", "position": "CTO", "specialization": "brain_computer_interfaces", "education": "MIT_PhD", "age": 42}),
            ("Marcus Rodriguez", "software_engineer", {"company": "Quantum_Dynamics", "position": "senior_engineer", "specialization": "quantum_algorithms", "education": "Stanford_MS", "age": 29}),
            ("Elena Volkov", "ai_researcher", {"company": "DeepMind_Labs", "position": "research_scientist", "specialization": "neural_networks", "education": "Oxford_PhD", "age": 35}),
            
            # Healthcare
            ("Dr. James Patterson", "surgeon", {"hospital": "Metropolitan_General", "position": "chief_surgeon", "specialization": "cardiac_surgery", "education": "Harvard_Medical", "age": 51}),
            ("Lisa Thompson", "nurse_practitioner", {"hospital": "Community_Health", "position": "head_nurse", "specialization": "emergency_care", "education": "Johns_Hopkins", "age": 38}),
            
            # Academia
            ("Prof. David Kumar", "physicist", {"university": "Princeton", "position": "department_head", "specialization": "quantum_physics", "education": "Caltech_PhD", "age": 47}),
            ("Dr. Maria Santos", "historian", {"university": "Yale", "position": "professor", "specialization": "medieval_history", "education": "Cambridge_PhD", "age": 44}),
        ]
        
        # Generate more modern people
        professions = ["engineer", "doctor", "teacher", "lawyer", "architect", "designer", "journalist", "artist", "musician", "chef"]
        companies = ["TechCorp", "MediHealth", "EduSystems", "LegalMax", "DesignPro", "MediaGroup", "ArtStudio", "MusicLab", "GourmetPlus", "InnovateNow"]
        
        for i in range(143):  # Generate 143 more people
            profession = professions[i % len(professions)]
            company = companies[i % len(companies)]
            
            name = f"{self._generate_first_name()}_{i} {self._generate_last_name()}"
            attributes = {
                "profession": profession,
                "company": company,
                "experience_years": (i % 30) + 1,
                "education": self._generate_education(),
                "age": (i % 50) + 22,
                "location": self._generate_city(),
                "specialization": self._generate_specialization(profession),
                "achievements": self._generate_achievements()
            }
            people_data.append((name, f"modern_{profession}", attributes))
        
        person_ids = []
        for name, concept_type, attributes in people_data:
            person_id = memory_core.add_concept(name, concept_type, attributes)
            person_ids.append(person_id)
        
        return person_ids
    
    def _create_organizations(self, memory_core: MemoryCoreService) -> List[str]:
        """Create modern organizations."""
        org_data = [
            # Tech Companies
            ("QuantumTech Industries", "tech_company", {"industry": "quantum_computing", "employees": 15000, "revenue": "50B", "founded": 2015, "headquarters": "Silicon_Valley"}),
            ("BioGenesis Labs", "biotech_company", {"industry": "biotechnology", "employees": 8000, "revenue": "12B", "founded": 2010, "headquarters": "Boston"}),
            ("Neural Dynamics", "ai_company", {"industry": "artificial_intelligence", "employees": 3000, "revenue": "8B", "founded": 2018, "headquarters": "Seattle"}),
            
            # Healthcare
            ("Global Health Alliance", "healthcare_org", {"type": "nonprofit", "members": 500, "budget": "2B", "founded": 1995, "headquarters": "Geneva"}),
            ("Metropolitan Medical Center", "hospital", {"beds": 800, "staff": 5000, "specialization": "research_hospital", "founded": 1965, "location": "New_York"}),
            
            # Educational
            ("Future Learning Institute", "educational_org", {"type": "research_institute", "faculty": 200, "students": 5000, "focus": "emerging_technologies", "founded": 2005}),
            ("Global Education Foundation", "nonprofit", {"type": "educational_charity", "programs": 50, "beneficiaries": 1000000, "founded": 1985, "global_reach": True}),
        ]
        
        # Generate more organizations
        org_types = ["corporation", "nonprofit", "startup", "government", "university", "hospital", "laboratory", "foundation", "consortium", "cooperative"]
        industries = ["technology", "healthcare", "education", "finance", "energy", "manufacturing", "agriculture", "transportation", "entertainment", "consulting"]
        
        for i in range(93):  # Generate 93 more organizations
            org_type = org_types[i % len(org_types)]
            industry = industries[i % len(industries)]
            
            name = f"{self._generate_company_name()}_{i} {org_type.title()}"
            attributes = {
                "type": org_type,
                "industry": industry,
                "size": ["small", "medium", "large", "enterprise"][min(i // 11, 3)],
                "founded": 2024 - (i % 50),
                "employees": (i + 1) * 100,
                "headquarters": self._generate_city(),
                "mission": self._generate_mission(org_type, industry),
                "status": "active"
            }
            org_data.append((name, f"{industry}_{org_type}", attributes))
        
        org_ids = []
        for name, concept_type, attributes in org_data:
            org_id = memory_core.add_concept(name, concept_type, attributes)
            org_ids.append(org_id)
        
        return org_ids
    
    def _create_technologies(self, memory_core: MemoryCoreService) -> List[str]:
        """Create modern technologies."""
        tech_data = [
            # Cutting-edge Tech
            ("Quantum Neural Processor", "quantum_tech", {"type": "processor", "capability": "quantum_neural_hybrid", "development_stage": "prototype", "applications": "ai_acceleration"}),
            ("Bio-Integrated Display", "biotech", {"type": "interface", "capability": "retinal_projection", "development_stage": "testing", "applications": "augmented_reality"}),
            ("Fusion Energy Cell", "energy_tech", {"type": "power_source", "capability": "portable_fusion", "development_stage": "experimental", "applications": "clean_energy"}),
            
            # AI Technologies
            ("Conscious AI Framework", "ai_tech", {"type": "software", "capability": "artificial_consciousness", "development_stage": "research", "applications": "general_intelligence"}),
            ("Predictive Health Monitor", "medical_ai", {"type": "diagnostic", "capability": "disease_prediction", "development_stage": "clinical_trials", "applications": "preventive_medicine"}),
        ]
        
        # Generate more technologies
        tech_types = ["software", "hardware", "biotechnology", "nanotechnology", "robotics", "ai", "quantum", "energy", "materials", "communications"]
        capabilities = ["processing", "storage", "transmission", "analysis", "synthesis", "automation", "enhancement", "monitoring", "control", "optimization"]
        
        for i in range(95):  # Generate 95 more technologies
            tech_type = tech_types[i % len(tech_types)]
            capability = capabilities[i % len(capabilities)]
            
            name = f"{capability.title()} {tech_type.title()}_{i} System"
            attributes = {
                "type": tech_type,
                "capability": capability,
                "development_stage": ["concept", "research", "prototype", "testing", "production"][i % 5],
                "complexity": ["low", "medium", "high", "extreme"][i % 4],
                "cost": f"{(i + 1) * 1000000}",
                "timeline": f"{(i % 10) + 1}_years",
                "applications": self._generate_applications(tech_type),
                "challenges": self._generate_challenges(tech_type)
            }
            tech_data.append((name, f"{tech_type}_technology", attributes))
        
        tech_ids = []
        for name, concept_type, attributes in tech_data:
            tech_id = memory_core.add_concept(name, concept_type, attributes)
            tech_ids.append(tech_id)
        
        return tech_ids
    
    def _create_research_topics(self, memory_core: MemoryCoreService) -> List[str]:
        """Create scientific research topics."""
        research_data = [
            # Physics Research
            ("Quantum Consciousness Theory", "physics_research", {"field": "quantum_physics", "status": "active", "funding": "10M", "team_size": 12, "duration": "5_years"}),
            ("Dark Matter Detection", "physics_research", {"field": "particle_physics", "status": "ongoing", "funding": "50M", "team_size": 25, "duration": "10_years"}),
            ("Room Temperature Superconductivity", "physics_research", {"field": "materials_physics", "status": "breakthrough", "funding": "30M", "team_size": 18, "duration": "7_years"}),
            
            # Biology Research
            ("CRISPR Gene Therapy", "biology_research", {"field": "genetics", "status": "clinical_trials", "funding": "25M", "team_size": 20, "duration": "8_years"}),
            ("Synthetic Biology Applications", "biology_research", {"field": "synthetic_biology", "status": "development", "funding": "15M", "team_size": 14, "duration": "6_years"}),
            
            # Computer Science Research  
            ("Artificial General Intelligence", "cs_research", {"field": "artificial_intelligence", "status": "theoretical", "funding": "100M", "team_size": 50, "duration": "indefinite"}),
            ("Quantum Computing Algorithms", "cs_research", {"field": "quantum_computing", "status": "active", "funding": "40M", "team_size": 30, "duration": "12_years"}),
        ]
        
        # Generate more research topics
        fields = ["physics", "chemistry", "biology", "computer_science", "mathematics", "psychology", "neuroscience", "medicine", "engineering", "astronomy"]
        statuses = ["proposed", "active", "ongoing", "paused", "completed", "breakthrough", "theoretical", "experimental", "clinical", "applied"]
        
        for i in range(93):  # Generate 93 more research topics
            field = fields[i % len(fields)]
            status = statuses[i % len(statuses)]
            
            name = f"{self._generate_research_prefix()} {field.title()}_{i} Study"
            attributes = {
                "field": field,
                "status": status,
                "funding": f"{((i + 1) * 500000)}",
                "team_size": (i % 20) + 3,
                "duration": f"{(i % 15) + 1}_years",
                "methodology": self._generate_methodology(field),
                "objectives": self._generate_objectives(field),
                "challenges": self._generate_research_challenges(field)
            }
            research_data.append((name, f"{field}_research", attributes))
        
        research_ids = []
        for name, concept_type, attributes in research_data:
            research_id = memory_core.add_concept(name, concept_type, attributes)
            research_ids.append(research_id)
        
        return research_ids
    
    def _create_discoveries(self, memory_core: MemoryCoreService) -> List[str]:
        """Create scientific discoveries."""
        discovery_data = [
            # Major Discoveries
            ("Higgs Boson Detection", "physics_discovery", {"year": 2012, "field": "particle_physics", "significance": "fundamental", "impact": "confirms_standard_model"}),
            ("CRISPR-Cas9 Gene Editing", "biology_discovery", {"year": 2012, "field": "genetics", "significance": "revolutionary", "impact": "genetic_medicine"}),
            ("Gravitational Waves", "physics_discovery", {"year": 2015, "field": "astrophysics", "significance": "groundbreaking", "impact": "new_astronomy"}),
            ("Exoplanet Kepler-452b", "astronomy_discovery", {"year": 2015, "field": "astronomy", "significance": "important", "impact": "potentially_habitable"}),
        ]
        
        # Generate more discoveries
        fields = ["physics", "chemistry", "biology", "computer_science", "mathematics", "psychology", "neuroscience", "medicine", "engineering", "astronomy"]
        for i in range(76):  # Generate 76 more discoveries
            field = fields[i % len(fields)]
            year = 2024 - (i % 50)  # Discoveries from last 50 years
            
            name = f"{self._generate_discovery_name()}_{i} Discovery"
            attributes = {
                "year": year,
                "field": field,
                "significance": ["minor", "moderate", "major", "revolutionary"][i % 4],
                "impact": self._generate_impact(field),
                "discoverer": self._generate_scientist_name(),
                "method": self._generate_discovery_method(field),
                "verification": ["pending", "confirmed", "disputed", "accepted"][i % 4]
            }
            discovery_data.append((name, f"{field}_discovery", attributes))
        
        discovery_ids = []
        for name, concept_type, attributes in discovery_data:
            discovery_id = memory_core.add_concept(name, concept_type, attributes)
            discovery_ids.append(discovery_id)
        
        return discovery_ids
    
    def _create_theories(self, memory_core: MemoryCoreService) -> List[str]:
        """Create scientific theories."""
        theory_data = [
            # Established Theories
            ("Theory of Relativity", "physics_theory", {"proposer": "Einstein", "year": 1905, "status": "proven", "field": "physics", "applications": "gps_cosmology"}),
            ("Evolution by Natural Selection", "biology_theory", {"proposer": "Darwin", "year": 1859, "status": "proven", "field": "biology", "applications": "medicine_agriculture"}),
            ("Quantum Field Theory", "physics_theory", {"proposer": "Multiple", "year": 1940, "status": "proven", "field": "quantum_physics", "applications": "particle_physics"}),
            
            # Emerging Theories
            ("Many-Worlds Interpretation", "physics_theory", {"proposer": "Everett", "year": 1957, "status": "theoretical", "field": "quantum_mechanics", "applications": "quantum_computing"}),
            ("Integrated Information Theory", "neuroscience_theory", {"proposer": "Tononi", "year": 2008, "status": "developing", "field": "consciousness", "applications": "ai_research"}),
        ]
        
        # Generate more theories
        fields = ["physics", "chemistry", "biology", "computer_science", "mathematics", "psychology", "neuroscience", "medicine", "engineering", "astronomy"]
        for i in range(75):  # Generate 75 more theories
            field = fields[i % len(fields)]
            year = 1800 + (i * 8)  # Theories spanning 200+ years
            
            name = f"{self._generate_theory_prefix()} Theory_{i} of {field.title()}"
            attributes = {
                "proposer": self._generate_scientist_name(),
                "year": year,
                "status": ["theoretical", "developing", "testing", "accepted", "proven"][i % 5],
                "field": field,
                "applications": self._generate_theory_applications(field),
                "predictions": self._generate_predictions(field),
                "support_level": ["weak", "moderate", "strong", "overwhelming"][i % 4]
            }
            theory_data.append((name, f"{field}_theory", attributes))
        
        theory_ids = []
        for name, concept_type, attributes in theory_data:
            theory_id = memory_core.add_concept(name, concept_type, attributes)
            theory_ids.append(theory_id)
        
        return theory_ids
    
    def _create_historical_events(self, memory_core: MemoryCoreService) -> List[str]:
        """Create historical events."""
        event_data = [
            # Ancient Events
            ("Fall of the Roman Empire", "historical_event", {"year": 476, "period": "ancient", "region": "Europe", "significance": "civilization_collapse", "causes": "internal_external_pressures"}),
            ("Construction of the Great Wall", "historical_event", {"year": -220, "period": "ancient", "region": "China", "significance": "defensive_achievement", "causes": "border_protection"}),
            ("Founding of Rome", "historical_event", {"year": -753, "period": "ancient", "region": "Italy", "significance": "city_establishment", "causes": "settlement_expansion"}),
            
            # Medieval Events
            ("Norman Conquest of England", "historical_event", {"year": 1066, "period": "medieval", "region": "England", "significance": "political_transformation", "causes": "succession_dispute"}),
            ("Black Death Pandemic", "historical_event", {"year": 1347, "period": "medieval", "region": "Europe", "significance": "demographic_catastrophe", "causes": "disease_outbreak"}),
            
            # Modern Events
            ("Industrial Revolution", "historical_event", {"year": 1760, "period": "modern", "region": "Britain", "significance": "technological_transformation", "causes": "innovation_resources"}),
            ("World War I", "historical_event", {"year": 1914, "period": "modern", "region": "Global", "significance": "global_conflict", "causes": "alliance_tensions"}),
        ]
        
        # Generate more historical events
        periods = ["ancient", "medieval", "renaissance", "enlightenment", "industrial", "modern", "contemporary"]
        regions = ["Europe", "Asia", "Americas", "Africa", "Middle_East", "Oceania"]
        
        for i in range(93):  # Generate 93 more events
            period = periods[i % len(periods)]
            region = regions[i % len(regions)]
            year = -3000 + (i * 100)  # Events spanning 5000+ years
            
            name = f"{self._generate_event_name()}_{i} of {region}"
            attributes = {
                "year": year,
                "period": period,
                "region": region,
                "significance": self._generate_significance(),
                "causes": self._generate_causes(),
                "consequences": self._generate_consequences(),
                "duration": self._generate_duration(),
                "key_figures": self._generate_key_figures()
            }
            event_data.append((name, f"{period}_event", attributes))
        
        event_ids = []
        for name, concept_type, attributes in event_data:
            event_id = memory_core.add_concept(name, concept_type, attributes)
            event_ids.append(event_id)
        
        return event_ids
    
    def _create_historical_periods(self, memory_core: MemoryCoreService) -> List[str]:
        """Create historical periods."""
        period_data = [
            ("Stone Age", "prehistoric_period", {"start_year": -3000000, "end_year": -3000, "characteristics": "stone_tools", "regions": "global", "developments": "tool_making"}),
            ("Bronze Age", "ancient_period", {"start_year": -3000, "end_year": -1200, "characteristics": "bronze_metallurgy", "regions": "eurasia_africa", "developments": "metalworking"}),
            ("Classical Antiquity", "ancient_period", {"start_year": -800, "end_year": 500, "characteristics": "greek_roman_civilization", "regions": "mediterranean", "developments": "philosophy_politics"}),
            ("Middle Ages", "medieval_period", {"start_year": 500, "end_year": 1500, "characteristics": "feudalism", "regions": "europe", "developments": "christianity_islam"}),
            ("Renaissance", "early_modern_period", {"start_year": 1400, "end_year": 1600, "characteristics": "cultural_rebirth", "regions": "europe", "developments": "art_science"}),
            ("Industrial Age", "modern_period", {"start_year": 1760, "end_year": 1840, "characteristics": "mechanization", "regions": "britain_europe", "developments": "factories_steam"}),
            ("Information Age", "contemporary_period", {"start_year": 1950, "end_year": 2024, "characteristics": "digital_technology", "regions": "global", "developments": "computers_internet"}),
        ]
        
        # Generate more periods
        for i in range(63):  # Generate 63 more periods
            start_year = -2000 + (i * 300)
            end_year = start_year + 200 + (i * 50)
            
            name = f"{self._generate_period_name()}_{i} Period"
            period_type = ["ancient", "medieval", "renaissance", "modern", "contemporary"][min(i // 3, 4)]
            
            attributes = {
                "start_year": start_year,
                "end_year": end_year,
                "duration": end_year - start_year,
                "characteristics": self._generate_characteristics(),
                "regions": self._generate_regions(),
                "developments": self._generate_developments(),
                "key_technologies": self._generate_technologies_for_period(),
                "cultural_aspects": self._generate_cultural_aspects()
            }
            period_data.append((name, f"{period_type}_period", attributes))
        
        period_ids = []
        for name, concept_type, attributes in period_data:
            period_id = memory_core.add_concept(name, concept_type, attributes)
            period_ids.append(period_id)
        
        return period_ids
    
    def _create_historical_figures(self, memory_core: MemoryCoreService) -> List[str]:
        """Create historical figures."""
        figure_data = [
            # Ancient Figures
            ("Julius Caesar", "ancient_leader", {"birth_year": -100, "death_year": -44, "region": "Rome", "role": "emperor", "achievements": "gallic_wars_political_reform"}),
            ("Cleopatra VII", "ancient_leader", {"birth_year": -69, "death_year": -30, "region": "Egypt", "role": "pharaoh", "achievements": "diplomatic_alliances"}),
            ("Aristotle", "ancient_philosopher", {"birth_year": -384, "death_year": -322, "region": "Greece", "role": "philosopher", "achievements": "logic_natural_philosophy"}),
            
            # Medieval Figures
            ("Charlemagne", "medieval_leader", {"birth_year": 742, "death_year": 814, "region": "Europe", "role": "emperor", "achievements": "frankish_empire_expansion"}),
            ("Joan of Arc", "medieval_warrior", {"birth_year": 1412, "death_year": 1431, "region": "France", "role": "military_leader", "achievements": "orleans_victory"}),
            
            # Modern Figures
            ("Leonardo da Vinci", "renaissance_genius", {"birth_year": 1452, "death_year": 1519, "region": "Italy", "role": "polymath", "achievements": "art_invention_science"}),
            ("Isaac Newton", "scientific_revolutionary", {"birth_year": 1643, "death_year": 1727, "region": "England", "role": "scientist", "achievements": "laws_of_motion_calculus"}),
            ("Napoleon Bonaparte", "modern_leader", {"birth_year": 1769, "death_year": 1821, "region": "France", "role": "emperor", "achievements": "european_conquest_legal_code"}),
        ]
        
        # Generate more historical figures
        roles = ["leader", "warrior", "philosopher", "scientist", "artist", "explorer", "inventor", "religious_figure", "writer", "musician"]
        for i in range(92):  # Generate 92 more figures
            role = roles[i % len(roles)]
            birth_year = -500 + (i * 50)
            death_year = birth_year + (30 + (i % 60))
            
            name = f"{self._generate_historical_name()}_{i} the {self._generate_epithet()}"
            period = self._determine_historical_period(birth_year)
            
            regions = ["Europe", "Asia", "Americas", "Africa", "Middle_East", "Oceania"]
            attributes = {
                "birth_year": birth_year,
                "death_year": death_year,
                "lifespan": death_year - birth_year,
                "region": regions[i % len(regions)],
                "role": role,
                "achievements": self._generate_achievements_for_role(role),
                "legacy": self._generate_legacy(role),
                "contemporaries": self._generate_contemporaries()
            }
            figure_data.append((name, f"{period}_{role}", attributes))
        
        figure_ids = []
        for name, concept_type, attributes in figure_data:
            figure_id = memory_core.add_concept(name, concept_type, attributes)
            figure_ids.append(figure_id)
        
        return figure_ids
    
    def _create_fantasy_relationships(self, memory_core: MemoryCoreService, character_ids: List[str], 
                                    location_ids: List[str], item_ids: List[str], skill_ids: List[str]) -> int:
        """Create relationships between fantasy world concepts."""
        relations_created = 0
        
        # Character-Location relationships (lives_in, visits, rules)
        for i, char_id in enumerate(character_ids[:80]):
            loc_id = location_ids[i % len(location_ids)]
            relation_type = ["lives_in", "rules", "visits", "guards", "haunts"][i % 5]
            memory_core.add_relation(char_id, loc_id, relation_type, {"strength": "strong", "duration": "permanent"})
            relations_created += 1
        
        # Character-Item relationships (owns, wields, seeks)
        for i, char_id in enumerate(character_ids[:100]):
            item_id = item_ids[i % len(item_ids)]
            relation_type = ["owns", "wields", "seeks", "crafted", "lost"][i % 5]
            memory_core.add_relation(char_id, item_id, relation_type, {"acquisition_method": "quest", "importance": "high"})
            relations_created += 1
        
        # Character-Skill relationships (knows, teaches, learns)
        for i, char_id in enumerate(character_ids[:90]):
            skill_id = skill_ids[i % len(skill_ids)]
            relation_type = ["knows", "teaches", "masters", "develops"][i % 4]
            memory_core.add_relation(char_id, skill_id, relation_type, {"proficiency": "expert", "years_practiced": i + 5})
            relations_created += 1
        
        # Character-Character relationships (allies, enemies, mentors)
        for i in range(min(120, len(character_ids) - 1)):
            char1_id = character_ids[i]
            char2_id = character_ids[(i + 1) % len(character_ids)]
            relation_type = ["allies", "enemies", "mentor_student", "rivals", "companions"][i % 5]
            memory_core.add_relation(char1_id, char2_id, relation_type, {"history": "long_standing", "intensity": "strong"})
            relations_created += 1
        
        # Location-Item relationships (contains, hides)
        for i, loc_id in enumerate(location_ids[:60]):
            item_id = item_ids[i % len(item_ids)]
            relation_type = ["contains", "hides", "guards", "displays"][i % 4]
            memory_core.add_relation(loc_id, item_id, relation_type, {"accessibility": "restricted", "security": "high"})
            relations_created += 1
        
        # Additional Fantasy Relationships
        # Location-Location relationships (connected_to, part_of, near)
        for i in range(min(40, len(location_ids) - 1)):
            loc1_id = location_ids[i]
            loc2_id = location_ids[(i + 2) % len(location_ids)]
            relation_type = ["connected_to", "part_of", "near", "overlooks"][i % 4]
            memory_core.add_relation(loc1_id, loc2_id, relation_type, {"distance": f"{(i+1)*10}_miles", "travel_difficulty": "moderate"})
            relations_created += 1
        
        # Item-Item relationships (paired_with, requires, upgrades)
        for i in range(min(50, len(item_ids) - 1)):
            item1_id = item_ids[i]
            item2_id = item_ids[(i + 3) % len(item_ids)]
            relation_type = ["paired_with", "requires", "upgrades", "component_of"][i % 4]
            memory_core.add_relation(item1_id, item2_id, relation_type, {"synergy_bonus": "enhanced_power", "compatibility": "perfect"})
            relations_created += 1
        
        return relations_created
    
    def _create_modern_relationships(self, memory_core: MemoryCoreService, person_ids: List[str], 
                                   org_ids: List[str], tech_ids: List[str]) -> int:
        """Create relationships between modern world concepts."""
        relations_created = 0
        
        # Person-Organization relationships
        for i, person_id in enumerate(person_ids[:50]):
            org_id = org_ids[i % len(org_ids)]
            relation_type = ["works_for", "leads", "founded", "consults_for", "collaborates_with"][i % 5]
            memory_core.add_relation(person_id, org_id, relation_type, {"role": "key_contributor", "since": 2020 - (i % 10)})
            relations_created += 1
        
        # Person-Technology relationships
        for i, person_id in enumerate(person_ids[:40]):
            tech_id = tech_ids[i % len(tech_ids)]
            relation_type = ["develops", "uses", "researches", "invented", "improves"][i % 5]
            memory_core.add_relation(person_id, tech_id, relation_type, {"expertise_level": "expert", "contribution": "significant"})
            relations_created += 1
        
        # Organization-Technology relationships
        for i, org_id in enumerate(org_ids[:30]):
            tech_id = tech_ids[i % len(tech_ids)]
            relation_type = ["develops", "uses", "licenses", "patents", "commercializes"][i % 5]
            memory_core.add_relation(org_id, tech_id, relation_type, {"investment": f"{(i+1)*1000000}", "timeline": "2_years"})
            relations_created += 1
        
        # Person-Person professional relationships
        for i in range(min(40, len(person_ids) - 1)):
            person1_id = person_ids[i]
            person2_id = person_ids[(i + 3) % len(person_ids)]
            relation_type = ["colleague", "mentor", "competitor", "collaborator", "friend"][i % 5]
            memory_core.add_relation(person1_id, person2_id, relation_type, {"professional_context": True, "duration": f"{i+1}_years"})
            relations_created += 1
        
        return relations_created
    
    def _create_scientific_relationships(self, memory_core: MemoryCoreService, research_ids: List[str], 
                                       discovery_ids: List[str], theory_ids: List[str], person_ids: List[str]) -> int:
        """Create relationships between scientific concepts."""
        relations_created = 0
        
        # Research-Discovery relationships
        for i, research_id in enumerate(research_ids[:20]):
            discovery_id = discovery_ids[i % len(discovery_ids)]
            relation_type = ["led_to", "inspired_by", "validates", "contradicts"][i % 4]
            memory_core.add_relation(research_id, discovery_id, relation_type, {"confidence": "high", "year": 2020 + i})
            relations_created += 1
        
        # Discovery-Theory relationships
        for i, discovery_id in enumerate(discovery_ids[:15]):
            theory_id = theory_ids[i % len(theory_ids)]
            relation_type = ["supports", "challenges", "extends", "validates"][i % 4]
            memory_core.add_relation(discovery_id, theory_id, relation_type, {"significance": "major", "consensus": "growing"})
            relations_created += 1
        
        # Person-Research relationships
        for i, person_id in enumerate(person_ids[:30]):
            research_id = research_ids[i % len(research_ids)]
            relation_type = ["leads", "participates_in", "funds", "oversees"][i % 4]
            memory_core.add_relation(person_id, research_id, relation_type, {"role": "primary", "commitment": "full_time"})
            relations_created += 1
        
        # Theory-Theory relationships
        for i in range(min(15, len(theory_ids) - 1)):
            theory1_id = theory_ids[i]
            theory2_id = theory_ids[(i + 1) % len(theory_ids)]
            relation_type = ["builds_on", "contradicts", "unifies", "extends"][i % 4]
            memory_core.add_relation(theory1_id, theory2_id, relation_type, {"relationship_strength": "strong", "historical": True})
            relations_created += 1
        
        return relations_created
    
    def _create_historical_relationships(self, memory_core: MemoryCoreService, event_ids: List[str], 
                                       period_ids: List[str], figure_ids: List[str]) -> int:
        """Create relationships between historical concepts."""
        relations_created = 0
        
        # Figure-Event relationships
        for i, figure_id in enumerate(figure_ids[:30]):
            event_id = event_ids[i % len(event_ids)]
            relation_type = ["participated_in", "caused", "influenced", "witnessed", "led"][i % 5]
            memory_core.add_relation(figure_id, event_id, relation_type, {"role": "central", "impact": "significant"})
            relations_created += 1
        
        # Event-Period relationships
        for i, event_id in enumerate(event_ids[:25]):
            period_id = period_ids[i % len(period_ids)]
            relation_type = ["occurred_during", "defined", "ended", "began"][i % 4]
            memory_core.add_relation(event_id, period_id, relation_type, {"historical_significance": "major", "duration": "extended"})
            relations_created += 1
        
        # Figure-Period relationships
        for i, figure_id in enumerate(figure_ids[:35]):
            period_id = period_ids[i % len(period_ids)]
            relation_type = ["lived_during", "exemplified", "transcended", "shaped"][i % 4]
            memory_core.add_relation(figure_id, period_id, relation_type, {"influence": "profound", "legacy": "lasting"})
            relations_created += 1
        
        # Event-Event relationships
        for i in range(min(25, len(event_ids) - 1)):
            event1_id = event_ids[i]
            event2_id = event_ids[(i + 1) % len(event_ids)]
            relation_type = ["preceded", "caused", "influenced", "paralleled"][i % 4]
            memory_core.add_relation(event1_id, event2_id, relation_type, {"causal_strength": "strong", "time_gap": f"{i+1}_years"})
            relations_created += 1
        
        return relations_created
    
    def _create_cross_domain_relationships(self, memory_core: MemoryCoreService, all_concepts: Dict[str, List[str]]) -> int:
        """Create fascinating cross-domain relationships between different concept types."""
        relations_created = 0
        
        # Fantasy characters inspired by historical figures
        for i in range(min(15, len(all_concepts['characters']), len(all_concepts['figures']))):
            char_id = all_concepts['characters'][i]
            figure_id = all_concepts['figures'][i]
            memory_core.add_relation(char_id, figure_id, "inspired_by", {"creative_interpretation": "fantasy_adaptation", "traits_borrowed": "leadership_style"})
            relations_created += 1
        
        # Modern people studying historical events
        for i in range(min(20, len(all_concepts['people']), len(all_concepts['events']))):
            person_id = all_concepts['people'][i]
            event_id = all_concepts['events'][i]
            memory_core.add_relation(person_id, event_id, "researches", {"field": "historical_analysis", "focus": "socioeconomic_impact"})
            relations_created += 1
        
        # Technologies inspired by fantasy items
        for i in range(min(10, len(all_concepts['technologies']), len(all_concepts['items']))):
            tech_id = all_concepts['technologies'][i]
            item_id = all_concepts['items'][i]
            memory_core.add_relation(tech_id, item_id, "conceptually_similar", {"design_inspiration": "fictional_prototype", "functionality": "enhanced_realism"})
            relations_created += 1
        
        # Organizations located in fantasy-inspired locations
        for i in range(min(12, len(all_concepts['organizations']), len(all_concepts['locations']))):
            org_id = all_concepts['organizations'][i]
            loc_id = all_concepts['locations'][i]
            memory_core.add_relation(org_id, loc_id, "naming_inspiration", {"thematic_connection": "aspirational_branding", "cultural_reference": "fantasy_literature"})
            relations_created += 1
        
        # Research projects investigating historical discoveries
        for i in range(min(15, len(all_concepts['research']), len(all_concepts['discoveries']))):
            research_id = all_concepts['research'][i]
            discovery_id = all_concepts['discoveries'][i]
            memory_core.add_relation(research_id, discovery_id, "builds_upon", {"methodology": "modern_reexamination", "goal": "deeper_understanding"})
            relations_created += 1
        
        # Skills that bridge fantasy and modern worlds
        for i in range(min(8, len(all_concepts['skills']), len(all_concepts['people']))):
            skill_id = all_concepts['skills'][i]
            person_id = all_concepts['people'][i]
            memory_core.add_relation(person_id, skill_id, "modern_equivalent", {"adaptation": "professional_skill", "application": "career_development"})
            relations_created += 1
        
        return relations_created
    
    def _test_similarity_search(self, memory_core: MemoryCoreService):
        """Test similarity search functionality."""
        print("   🔍 Testing similarity search...")
        
        # Test character similarity
        char_results = memory_core.query_similar_concepts("brave warrior knight", limit=5)
        assert len(char_results) > 0, "Should find similar characters"
        
        # Test location similarity  
        loc_results = memory_core.query_similar_concepts("ancient magical forest", limit=5)
        assert len(loc_results) > 0, "Should find similar locations"
        
        # Test technology similarity
        tech_results = memory_core.query_similar_concepts("artificial intelligence computer system", limit=5)
        assert len(tech_results) > 0, "Should find similar technologies"
        
        print(f"      Found {len(char_results)} character matches")
        print(f"      Found {len(loc_results)} location matches")
        print(f"      Found {len(tech_results)} technology matches")
    
    def _test_relationship_traversal(self, memory_core: MemoryCoreService, character_ids: List[str], location_ids: List[str]):
        """Test multi-hop relationship traversal."""
        print("   🕸️  Testing relationship traversal...")
        
        if character_ids and location_ids:
            # Test single hop
            single_hop = memory_core.get_related_concepts(character_ids[0], hop=1)
            assert len(single_hop[0]) > 0 or len(single_hop[1]) > 0, "Should find single-hop relationships"
            
            # Test multi-hop
            multi_hop = memory_core.get_related_concepts(character_ids[0], hop=2)
            assert len(multi_hop[0]) >= len(single_hop[0]), "Multi-hop should find at least as many concepts"
            
            print(f"      Single hop: {len(single_hop[0])} concepts, {len(single_hop[1])} relations")
            print(f"      Multi hop: {len(multi_hop[0])} concepts, {len(multi_hop[1])} relations")
    
    def _test_complex_queries(self, memory_core: MemoryCoreService):
        """Test complex query scenarios."""
        print("   🧩 Testing complex queries...")
        
        # Search for specific concept types
        all_nodes = memory_core.graph_store.get_all_nodes()
        character_nodes = [node for node in all_nodes if "character" in node.get("node_type", "").lower() or 
                          "person" in node.get("node_type", "").lower() or
                          "figure" in node.get("node_type", "").lower()]
        
        location_nodes = [node for node in all_nodes if "location" in node.get("node_type", "").lower() or
                         "city" in node.get("node_type", "").lower() or
                         "place" in node.get("node_type", "").lower()]
        
        print(f"      Character-like nodes: {len(character_nodes)}")
        print(f"      Location-like nodes: {len(location_nodes)}")
        
        # Test concept retrieval by ID
        if all_nodes:
            sample_node = all_nodes[0]
            retrieved = memory_core.get_concept(sample_node["node_id"])
            assert retrieved is not None, "Should retrieve concept by ID"
            print(f"      Successfully retrieved concept: {retrieved.get('node_name', 'Unknown')}")
    
    def _test_graph_statistics(self, memory_core: MemoryCoreService):
        """Test graph statistics functionality."""
        print("   📈 Testing graph statistics...")
        
        stats = memory_core.get_graph_statistics()
        
        assert stats["total_nodes"] > 0, "Should have nodes"
        assert stats["total_edges"] > 0, "Should have edges"
        assert len(stats["node_types"]) > 0, "Should have node types"
        assert len(stats["edge_types"]) > 0, "Should have edge types"
        
        print(f"      Nodes: {stats['total_nodes']}")
        print(f"      Edges: {stats['total_edges']}")
        print(f"      Node types: {len(stats['node_types'])}")
        print(f"      Edge types: {len(stats['edge_types'])}")
    
    def _test_save_and_load_performance(self, memory_core: MemoryCoreService):
        """Test save/load performance."""
        print("   💾 Testing save/load performance...")
        
        start_time = time.time()
        memory_core.save_graph()
        save_time = time.time() - start_time
        
        print(f"      Save operation took: {save_time:.2f} seconds")
        
        # Verify save was successful by checking file sizes
        stats_after_save = memory_core.get_graph_statistics()
        assert stats_after_save["total_nodes"] > 0, "Graph should still have data after save"
    
    def _test_large_scale_operations(self, memory_core: MemoryCoreService):
        """Test performance with large-scale operations."""
        print("   ⚡ Testing large-scale operations...")
        
        # Test batch concept creation
        start_time = time.time()
        batch_ids = []
        for i in range(50):
            concept_id = memory_core.add_concept(
                f"Performance Test Concept {i}",
                "test_concept",
                {"index": i, "created_for": "performance_test"}
            )
            batch_ids.append(concept_id)
        batch_time = time.time() - start_time
        
        print(f"      Created 50 concepts in: {batch_time:.2f} seconds")
        print(f"      Average time per concept: {batch_time/50:.4f} seconds")
        
        # Test batch relationship creation
        start_time = time.time()
        for i in range(min(25, len(batch_ids) - 1)):
            memory_core.add_relation(
                batch_ids[i], 
                batch_ids[i + 1], 
                "test_relation",
                {"created_for": "performance_test"}
            )
        relation_time = time.time() - start_time
        
        print(f"      Created 25 relations in: {relation_time:.2f} seconds")
        print(f"      Average time per relation: {relation_time/25:.4f} seconds")
    
    def _get_directory_size(self, path: str) -> float:
        """Get total size of directory in MB."""
        total_size = 0
        try:
            for dirpath, _, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
        except OSError:
            return 0
        return total_size / (1024 * 1024)  # Convert to MB
    
    # Helper methods for generating realistic data
    def _generate_fantasy_name(self) -> str:
        first_names = ["Aeliana", "Thorgrim", "Zara", "Eldrin", "Lyra", "Kael", "Naia", "Draven", "Seraphina", "Gareth"]
        return first_names[hash(str(time.time())) % len(first_names)]
    
    def _generate_fantasy_surname(self) -> str:
        surnames = ["Starweaver", "Ironbeard", "Shadowstep", "Lightbringer", "Stormwind", "Nightfall", "Dawnbreaker", "Frostborn", "Emberhart", "Moonwhisper"]
        return surnames[hash(str(time.time()) + "surname") % len(surnames)]
    
    def _generate_place_name(self) -> str:
        places = ["Silverleaf", "Ironhold", "Stormhaven", "Goldspire", "Shadowmere", "Brightwater", "Darkwood", "Sunhaven", "Moonvale", "Starfall"]
        return places[hash(str(time.time()) + "place") % len(places)]
    
    def _generate_background(self) -> str:
        backgrounds = ["noble", "merchant", "soldier", "scholar", "artisan", "criminal", "hermit", "entertainer", "folk_hero", "outlander"]
        return backgrounds[hash(str(time.time()) + "background") % len(backgrounds)]
    
    def _generate_feature(self) -> str:
        features = ["scar_across_face", "heterochromia", "silver_hair", "unusual_height", "magical_tattoos", "distinctive_voice", "missing_finger", "birthmark", "elegant_bearing", "weathered_hands"]
        return features[hash(str(time.time()) + "feature") % len(features)]
    
    def _generate_location_suffix(self) -> str:
        suffixes = ["Hold", "Haven", "Spire", "Falls", "Grove", "Reach", "Gate", "Bridge", "Cross", "End"]
        return suffixes[hash(str(time.time()) + "suffix") % len(suffixes)]
    
    def _generate_location_feature(self) -> str:
        features = ["ancient_ruins", "natural_spring", "haunted_cemetery", "magical_grove", "hidden_cave", "watchtower", "stone_circle", "crystal_formation", "underground_river", "floating_rocks"]
        return features[hash(str(time.time()) + "loc_feature") % len(features)]
    
    def _generate_creator_name(self) -> str:
        creators = ["Master_Artificer_Aldric", "Enchantress_Morgana", "Dwarven_Smith_Thorek", "Elven_Mage_Silvius", "Ancient_Dragon_Pyraxis", "Lost_Civilization_Atlanteans", "Divine_Craftsman_Hephaestus", "Shadow_Guild_Artisans", "Celestial_Forge_Masters", "Primordial_Shapers"]
        return creators[hash(str(time.time()) + "creator") % len(creators)]
    
    def _generate_item_history(self) -> str:
        histories = ["forged_in_dragon_fire", "blessed_by_celestials", "cursed_by_dark_magic", "wielded_by_ancient_hero", "lost_for_centuries", "stolen_from_gods", "crafted_in_secret", "discovered_in_ruins", "inherited_through_generations", "created_by_accident"]
        return histories[hash(str(time.time()) + "history") % len(histories)]
    
    def _generate_skill_prefix(self) -> str:
        prefixes = ["Master", "Advanced", "Legendary", "Ancient", "Forbidden", "Sacred", "Primal", "Divine", "Shadow", "Elemental"]
        return prefixes[hash(str(time.time()) + "skill_prefix") % len(prefixes)]
    
    def _generate_skill_description(self, skill_type: str) -> str:
        descriptions = {
            "combat": "devastating_battle_technique",
            "magic": "reality_altering_spell",
            "utility": "practical_problem_solving",
            "social": "persuasive_communication",
            "crafting": "masterwork_creation"
        }
        return descriptions.get(skill_type, "specialized_ability")
    
    def _generate_first_name(self) -> str:
        names = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Avery", "Quinn", "Sage", "River"]
        return names[hash(str(time.time()) + "first") % len(names)]
    
    def _generate_last_name(self) -> str:
        names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
        return names[hash(str(time.time()) + "last") % len(names)]
    
    def _generate_education(self) -> str:
        educations = ["Harvard_PhD", "MIT_MS", "Stanford_BS", "Oxford_DPhil", "Cambridge_MA", "Caltech_PhD", "Yale_JD", "Princeton_PhD", "Berkeley_MS", "Columbia_MD"]
        return educations[hash(str(time.time()) + "education") % len(educations)]
    
    def _generate_city(self) -> str:
        cities = ["New_York", "San_Francisco", "Boston", "Seattle", "Austin", "Denver", "Portland", "Chicago", "Los_Angeles", "Miami"]
        return cities[hash(str(time.time()) + "city") % len(cities)]
    
    def _generate_specialization(self, profession: str) -> str:
        specializations = {
            "engineer": "software_architecture",
            "doctor": "internal_medicine", 
            "teacher": "secondary_education",
            "lawyer": "corporate_law"
        }
        return specializations.get(profession, "general_practice")
    
    def _generate_achievements(self) -> str:
        achievements = ["published_research", "industry_award", "patent_holder", "conference_speaker", "team_leader", "innovation_prize", "peer_recognition", "mentorship_excellence", "thought_leadership", "breakthrough_discovery"]
        return achievements[hash(str(time.time()) + "achievement") % len(achievements)]
    
    def _generate_company_name(self) -> str:
        prefixes = ["Tech", "Innova", "Global", "Advanced", "Future", "Smart", "Dynamic", "Strategic", "Creative", "Elite"]
        suffixes = ["Solutions", "Systems", "Dynamics", "Corp", "Industries", "Labs", "Works", "Group", "Partners", "Ventures"]
        prefix = prefixes[hash(str(time.time()) + "comp_prefix") % len(prefixes)]
        suffix = suffixes[hash(str(time.time()) + "comp_suffix") % len(suffixes)]
        return f"{prefix}_{suffix}"
    
    def _generate_mission(self, org_type: str, industry: str) -> str:
        missions = {
            "corporation": f"leading_{industry}_innovation",
            "nonprofit": f"advancing_{industry}_for_humanity",
            "startup": f"disrupting_{industry}_with_technology"
        }
        return missions.get(org_type, f"excellence_in_{industry}")
    
    def _generate_applications(self, tech_type: str) -> str:
        apps = {
            "software": "automation_optimization",
            "hardware": "performance_enhancement", 
            "ai": "intelligent_decision_making"
        }
        return apps.get(tech_type, "general_purpose")
    
    def _generate_challenges(self, tech_type: str) -> str:
        challenges = {
            "software": "scalability_security",
            "hardware": "miniaturization_power",
            "ai": "ethics_interpretability"
        }
        return challenges.get(tech_type, "technical_complexity")
    
    def _generate_research_prefix(self) -> str:
        prefixes = ["Comprehensive", "Advanced", "Longitudinal", "Cross-Sectional", "Experimental", "Theoretical", "Applied", "Interdisciplinary", "Collaborative", "Breakthrough"]
        return prefixes[hash(str(time.time()) + "research_prefix") % len(prefixes)]
    
    def _generate_methodology(self, field: str) -> str:
        methods = {
            "physics": "experimental_measurement",
            "biology": "controlled_trials",
            "computer_science": "algorithmic_analysis"
        }
        return methods.get(field, "empirical_investigation")
    
    def _generate_objectives(self, field: str) -> str:
        objectives = {
            "physics": "fundamental_understanding",
            "biology": "therapeutic_applications", 
            "computer_science": "computational_advancement"
        }
        return objectives.get(field, "knowledge_expansion")
    
    def _generate_research_challenges(self, field: str) -> str:
        challenges = {
            "physics": "measurement_precision",
            "biology": "ethical_considerations",
            "computer_science": "computational_complexity"
        }
        return challenges.get(field, "resource_limitations")
    
    def _generate_discovery_name(self) -> str:
        names = ["Nova", "Quantum", "Stellar", "Molecular", "Neural", "Cosmic", "Cellular", "Digital", "Genetic", "Atomic"]
        return names[hash(str(time.time()) + "discovery") % len(names)]
    
    def _generate_impact(self, field: str) -> str:
        impacts = {
            "physics": "fundamental_physics_revolution",
            "biology": "medical_breakthrough",
            "chemistry": "material_science_advancement"
        }
        return impacts.get(field, "scientific_progress")
    
    def _generate_scientist_name(self) -> str:
        names = ["Dr_Anderson", "Prof_Chen", "Dr_Patel", "Prof_Johnson", "Dr_Garcia", "Prof_Williams", "Dr_Brown", "Prof_Davis", "Dr_Miller", "Prof_Wilson"]
        return names[hash(str(time.time()) + "scientist") % len(names)]
    
    def _generate_discovery_method(self, field: str) -> str:
        methods = {
            "physics": "particle_acceleration",
            "biology": "dna_sequencing",
            "astronomy": "telescope_observation"
        }
        return methods.get(field, "systematic_investigation")
    
    def _generate_theory_prefix(self) -> str:
        prefixes = ["Unified", "Extended", "Quantum", "Relativistic", "Dynamic", "Stochastic", "Holistic", "Emergent", "Systemic", "Fundamental"]
        return prefixes[hash(str(time.time()) + "theory_prefix") % len(prefixes)]
    
    def _generate_theory_applications(self, field: str) -> str:
        apps = {
            "physics": "technology_development",
            "biology": "medical_applications",
            "psychology": "therapeutic_interventions"
        }
        return apps.get(field, "practical_implementation")
    
    def _generate_predictions(self, field: str) -> str:
        predictions = {
            "physics": "particle_behavior",
            "biology": "evolutionary_patterns",
            "chemistry": "reaction_outcomes"
        }
        return predictions.get(field, "future_observations")
    
    def _generate_event_name(self) -> str:
        events = ["Great Revolution", "Golden Age", "Dark Period", "Renaissance", "Transformation", "Uprising", "Discovery", "Conquest", "Unification", "Liberation"]
        return events[hash(str(time.time()) + "event") % len(events)]
    
    def _generate_significance(self) -> str:
        significances = ["world_changing", "culturally_defining", "politically_transformative", "economically_revolutionary", "socially_disruptive", "technologically_advancing", "religiously_significant", "militarily_decisive", "demographically_impactful", "intellectually_groundbreaking"]
        return significances[hash(str(time.time()) + "significance") % len(significances)]
    
    def _generate_causes(self) -> str:
        causes = ["political_tension", "economic_pressure", "social_unrest", "technological_change", "natural_disaster", "cultural_shift", "religious_conflict", "resource_scarcity", "leadership_crisis", "external_threat"]
        return causes[hash(str(time.time()) + "causes") % len(causes)]
    
    def _generate_consequences(self) -> str:
        consequences = ["political_restructuring", "economic_transformation", "social_evolution", "cultural_renaissance", "technological_acceleration", "demographic_shift", "territorial_changes", "institutional_reform", "ideological_influence", "historical_legacy"]
        return consequences[hash(str(time.time()) + "consequences") % len(consequences)]
    
    def _generate_duration(self) -> str:
        durations = ["brief_but_decisive", "extended_process", "gradual_change", "sudden_transformation", "cyclical_pattern", "one_time_event", "recurring_phenomenon", "long_term_trend", "momentary_catalyst", "era_defining"]
        return durations[hash(str(time.time()) + "duration") % len(durations)]
    
    def _generate_key_figures(self) -> str:
        return f"Leader_{hash(str(time.time()) + 'leader') % 100}, General_{hash(str(time.time()) + 'general') % 100}, Advisor_{hash(str(time.time()) + 'advisor') % 100}"
    
    def _generate_period_name(self) -> str:
        names = ["Golden", "Silver", "Bronze", "Iron", "Crystal", "Diamond", "Emerald", "Sapphire", "Ruby", "Platinum"]
        return names[hash(str(time.time()) + "period") % len(names)]
    
    def _generate_characteristics(self) -> str:
        chars = ["technological_advancement", "cultural_flourishing", "political_stability", "economic_prosperity", "artistic_renaissance", "scientific_revolution", "social_harmony", "military_prowess", "religious_devotion", "intellectual_growth"]
        return chars[hash(str(time.time()) + "characteristics") % len(chars)]
    
    def _generate_regions(self) -> str:
        regions = ["Mediterranean", "Northern_Europe", "Eastern_Asia", "Middle_East", "North_Africa", "Central_America", "South_America", "Oceania", "Arctic_Region", "Tropical_Zone"]
        return regions[hash(str(time.time()) + "regions") % len(regions)]
    
    def _generate_developments(self) -> str:
        devs = ["writing_systems", "agricultural_techniques", "metallurgy_advances", "architectural_innovations", "transportation_methods", "communication_systems", "trade_networks", "educational_institutions", "legal_frameworks", "artistic_expressions"]
        return devs[hash(str(time.time()) + "developments") % len(devs)]
    
    def _generate_technologies_for_period(self) -> str:
        techs = ["stone_tools", "bronze_working", "iron_smelting", "wheel_invention", "writing_development", "printing_press", "steam_engine", "electricity", "computers", "internet"]
        return techs[hash(str(time.time()) + "period_tech") % len(techs)]
    
    def _generate_cultural_aspects(self) -> str:
        aspects = ["religious_beliefs", "artistic_styles", "social_structures", "value_systems", "traditions", "ceremonies", "languages", "literature", "music", "philosophy"]
        return aspects[hash(str(time.time()) + "culture") % len(aspects)]
    
    def _generate_historical_name(self) -> str:
        names = ["Maximus", "Aurelius", "Constantine", "Justinian", "Charlemagne", "Frederick", "Richard", "William", "Henry", "Louis"]
        return names[hash(str(time.time()) + "hist_name") % len(names)]
    
    def _generate_epithet(self) -> str:
        epithets = ["Great", "Wise", "Bold", "Just", "Magnificent", "Conqueror", "Liberator", "Reformer", "Builder", "Unifier"]
        return epithets[hash(str(time.time()) + "epithet") % len(epithets)]
    
    def _determine_historical_period(self, birth_year: int) -> str:
        if birth_year < 500:
            return "ancient"
        elif birth_year < 1500:
            return "medieval"
        elif birth_year < 1800:
            return "renaissance"
        else:
            return "modern"
    
    def _generate_achievements_for_role(self, role: str) -> str:
        achievements = {
            "leader": "unified_territories",
            "warrior": "won_decisive_battles",
            "philosopher": "developed_new_theories",
            "scientist": "made_breakthrough_discoveries",
            "artist": "created_masterworks"
        }
        return achievements.get(role, "left_lasting_legacy")
    
    def _generate_legacy(self, role: str) -> str:
        legacies = {
            "leader": "political_transformation",
            "warrior": "military_tactics",
            "philosopher": "intellectual_framework",
            "scientist": "scientific_method",
            "artist": "cultural_influence"
        }
        return legacies.get(role, "historical_significance")
    
    def _generate_contemporaries(self) -> str:
        return f"Historical_Figure_{hash(str(time.time()) + 'contemp1') % 100}, Notable_Person_{hash(str(time.time()) + 'contemp2') % 100}, Contemporary_Leader_{hash(str(time.time()) + 'contemp3') % 100}"


if __name__ == "__main__":
    # Run the comprehensive test directly
    test = TestComprehensiveLargeScale()
    memory_core_path = test.test_comprehensive_large_memory_core()
    print(f"\nMemory core created at: {memory_core_path}")