# enhanced_schema_mapping.py
import json
import argparse
import logging
import os
import re
import torch
from typing import Dict, List, Any, Tuple, Optional
from pydantic import BaseModel, ValidationError, Field, field_validator
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EnhancedSchemaMapper")
logger.setLevel(logging.INFO)

class FieldDefinition(BaseModel):
    description: str
    data_type: str
    synonyms: List[str] = Field(default_factory=list)
    extraction_instructions: str = ""
    context_keywords: List[str] = Field(default_factory=list)

class SchemaModel(BaseModel):
    fields: Dict[str, FieldDefinition] = Field(..., description="Schema field definitions")

class BlockData(BaseModel):
    block_id: str
    label: str
    confidence: float
    bounding_box: List[float]
    text: str

    @field_validator('block_id', mode='before')
    def convert_block_id(cls, v):
        """Convert integer block IDs to strings"""
        if isinstance(v, int):
            return str(v)
        return v

class MappingResult(BaseModel):
    exists: bool
    extracted_value: str = ""
    confidence_score: float = 0.0

class LocalLLM:
    def __init__(self, model_path: str, model_type: str, max_new_tokens: int = 10000):
        """
        Initialize local LLM
        
        Args:
            model_path: Path to local model
            model_type: 'mistral' or 'tinyllama'
            max_new_tokens: Maximum tokens to generate
        """
        self.model_path = model_path
        self.model_type = model_type
        self.max_new_tokens = max_new_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()
        
    def load_model(self):
        """Load the model and tokenizer"""
        logger.info(f"Loading {self.model_type} model from: {self.model_path}")
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            
            # Load model with device_map="auto" for automatic device placement
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                load_in_4bit=True if torch.cuda.is_available() else False
            )

            # Create text generation pipeline
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=self.max_new_tokens
            )
            
            # Log device information
            device_info = "CPU"
            if torch.cuda.is_available():
                device_info = f"GPU ({torch.cuda.get_device_name(0)})"
            logger.info(f"Model loaded successfully on {device_info}")
            return True
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            return False

    def generate(self, prompt: str) -> str:
        """
        Generate text from prompt
        
        Args:
            prompt: Input prompt for the model
            
        Returns:
            Generated text
        """
        try:
            # Format prompt based on model type
            if self.model_type == "mistral":
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            else:  # TinyLlama
                formatted_prompt = f"<|system|>\n</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
            
            # Generate response
            outputs = self.pipe(
                formatted_prompt,
                do_sample=False,
                temperature=0.0,
                return_full_text=False
            )
            
            return outputs[0]['generated_text'].strip()
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            return ""

class SchemaMapper:
    def __init__(self, mapping_model_path: str, validator_model_path: str):
        """
        Initialize the SchemaMapper with enhanced field analysis
        
        Args:
            mapping_model_path: Path to Mistral model
            validator_model_path: Path to TinyLlama model
        """
        logger.info("Initializing enhanced schema mapper:")
        logger.info("- Mistral: Primary extraction and conflict resolution")
        logger.info("- TinyLlama: Field analysis and validation")
        
        # Mistral: Main extraction model
        self.mapper_model = LocalLLM(
            model_path=mapping_model_path,
            model_type="mistral",
            max_new_tokens=2048
        )
        
        # TinyLlama: Field analysis and validation
        self.validator_model = LocalLLM(
            model_path=validator_model_path,
            model_type="tinyllama",
            max_new_tokens=5000
        )
        
        # Validate models loaded successfully
        if not self.mapper_model.model or not self.validator_model.model:
            raise RuntimeError("Failed to load one or more models")
        
        # Tracking counters
        self.mistral_extraction_calls = 0
        self.mistral_conflict_resolution_calls = 0
        self.tinyllama_validation_calls = 0
        self.tinyllama_field_analysis_calls = 0
        self.conflicts_detected = 0
        self.conflicts_resolved = 0
        self.fields_with_candidates = 0

    def analyze_schema_fields(
        self,
        schema: SchemaModel
    ) -> SchemaModel:
        """
        Enhance schema with field analysis using TinyLlama:
        - Generate synonyms
        - Create extraction instructions
        - Identify context keywords
        - Validate data types
        """
        logger.info("Enhancing schema with field analysis...")
        enhanced_fields = {}
        
        for field_name, field_def in schema.fields.items():
            self.tinyllama_field_analysis_calls += 1
            
            # Generate field metadata
            prompt = f"""
You are a field analysis expert. Analyze the field definition and generate:
1. Up to 5 common synonyms that might appear in documents
2. Specific extraction instructions based on field characteristics
3. Context keywords that might appear near the value in text

FIELD: {field_name}
DESCRIPTION: {field_def.description}
DATA TYPE: {field_def.data_type}

RESPONSE FORMAT:
{{
  "synonyms": ["synonym1", "synonym2", ...],
  "extraction_instructions": "Detailed instructions for value extraction",
  "context_keywords": ["keyword1", "keyword2", ...]
}}

Respond with ONLY the JSON object, no explanations.
"""
            response = self.validator_model.generate(prompt)
            
            try:
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group(0))
                    
                    # Update field definition
                    enhanced_fields[field_name] = FieldDefinition(
                        description=field_def.description,
                        data_type=field_def.data_type,
                        synonyms=analysis.get("synonyms", []),
                        extraction_instructions=analysis.get("extraction_instructions", ""),
                        context_keywords=analysis.get("context_keywords", [])
                    )
                    logger.info(f"Enhanced field: {field_name} with {len(analysis.get('synonyms', []))} synonyms")
                else:
                    logger.warning(f"No JSON found in field analysis for {field_name}")
                    enhanced_fields[field_name] = field_def
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"Invalid JSON in field analysis for {field_name}: {str(e)}")
                enhanced_fields[field_name] = field_def
                
        return SchemaModel(fields=enhanced_fields)

    def normalize_field_name(self, name: str) -> str:
        """Normalize field name by removing punctuation and extra spaces"""
        # Remove any non-alphanumeric characters except spaces
        name = re.sub(r'[^\w\s]', '', name)
        # Collapse multiple spaces and trim
        name = re.sub(r'\s+', ' ', name).strip()
        return name.lower()

    def extract_fields_from_block(
        self,
        block_text: str,
        schema: SchemaModel,
        block_id: str
    ) -> Dict[str, MappingResult]:
        """
        Extract ALL schema fields from a single text block using Mistral
        with enhanced synonym and context handling
        """
        # Build field descriptions with metadata
        field_descriptions = []
        for field_name, field_def in schema.fields.items():
            syns = ", ".join(field_def.synonyms) if field_def.synonyms else "None"
            context_words = ", ".join(field_def.context_keywords) if field_def.context_keywords else "None"
            field_descriptions.append(
                f"### {field_name}:\n"
                f"- Description: {field_def.description}\n"
                f"- Data Type: {field_def.data_type}\n"
                f"- Synonyms: {syns}\n"
                f"- Context Keywords: {context_words}\n"
                f"- Extraction: {field_def.extraction_instructions or 'Extract exact value as it appears'}"
            )
        
        prompt = f"""
You are an expert document field extractor. Analyze the text block to identify field values using their descriptions and synonyms.

FIELD DEFINITIONS:
{chr(10).join(field_descriptions)}

TEXT BLOCK (ID: {block_id}):
{block_text}

EXTRACTION RULES:
1. Match fields using any known synonyms from the definitions
2. For numbers: extract digits only (e.g., "Age: 25 years" → "25")
3. For dates: preserve original format or convert to YYYY-MM-DD
4. For alphanumeric codes: preserve exact formatting (e.g., "4932Z")
5. If a field appears multiple times, extract each occurrence
6. Return JSON with ALL field names from the schema
7. For missing fields: set "exists" to false and "extracted_value" to empty string
8. Ignore punctuation differences between field names and text (e.g., "Field:" matches "Field")

REQUIRED RESPONSE FORMAT:
{{
  "field1": {{"exists": boolean, "extracted_value": "string"}},
  "field2": {{"exists": boolean, "extracted_value": "string"}},
  ... (ALL FIELDS MUST BE INCLUDED)
}}

Respond with ONLY the JSON object, no explanations.
"""

        try:
            response = self.mapper_model.generate(prompt)
            self.mistral_extraction_calls += 1
            
            # Parse JSON response
            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    result_data = json.loads(json_match.group(0))
                else:
                    raise json.JSONDecodeError("No JSON found", response, 0)
                
                # Validate and create results
                field_results = {}
                normalized_fields = {
                    self.normalize_field_name(f): f for f in schema.fields
                }
                
                for response_field, field_data in result_data.items():
                    # Normalize response field name for matching
                    normalized_name = self.normalize_field_name(response_field)
                    
                    if normalized_name in normalized_fields:
                        orig_name = normalized_fields[normalized_name]
                        if not isinstance(field_data, dict):
                            field_results[orig_name] = MappingResult(exists=False, extracted_value="")
                            continue
                            
                        # Ensure required fields exist
                        exists = field_data.get("exists", False)
                        value = field_data.get("extracted_value", "")
                        field_results[orig_name] = MappingResult(
                            exists=exists, 
                            extracted_value=value,
                            confidence_score=1.0 if exists else 0.0
                        )
                    else:
                        logger.warning(f"Unexpected field in response: {response_field}")
                
                # Ensure all schema fields are in results
                for field_name in schema.fields:
                    if field_name not in field_results:
                        field_results[field_name] = MappingResult(exists=False, extracted_value="")
                        
                return field_results
                
            except json.JSONDecodeError as e:
                logger.warning(f"Mistral returned invalid JSON: {response[:200]}... Error: {str(e)}")
                # Return all fields as not found
                return {
                    field_name: MappingResult(exists=False, extracted_value="") 
                    for field_name in schema.fields
                }
            
        except Exception as e:
            logger.error(f"Multi-field extraction failed: {str(e)}")
            return {
                field_name: MappingResult(exists=False, extracted_value="") 
                for field_name in schema.fields
            }

    def validate_candidates(
        self,
        field_name: str,
        candidates: List[Dict]
    ) -> bool:
        """
        Use TinyLlama to validate if candidate values represent a real conflict
        """
        if len(candidates) <= 1:
            return False
            
        # Check if all values are identical
        unique_values = set(candidate["value"] for candidate in candidates)
        if len(unique_values) == 1:
            return False
        
        # Multiple different values detected
        candidate_values = [c['value'] for c in candidates]
        
        prompt = f"""
You are a conflict validator. Analyze if these candidate values represent a real conflict.

FIELD: {field_name}
CANDIDATE VALUES:
{chr(10).join(f"- {val}" for val in candidate_values)}

VALIDATION RULES:
1. Check if values are truly different and conflicting
2. Consider if they might be equivalent representations
3. Ignore minor formatting differences
4. Return "CALLBACK_NEEDED" only for genuine conflicts

RESPONSE FORMAT:
- Return "CALLBACK_NEEDED" if values are conflicting
- Return "NO_CALLBACK" if values are equivalent
- Respond with ONLY the decision keyword

Decision:"""

        try:
            response = self.validator_model.generate(prompt)
            self.tinyllama_validation_calls += 1
            
            # Clean response
            response = response.strip().upper()
            
            if "CALLBACK_NEEDED" in response:
                logger.info(f"Conflict validated for '{field_name}'")
                return True
            else:
                logger.info(f"No real conflict for '{field_name}'")
                return False
                
        except Exception as e:
            logger.error(f"Validation failed for '{field_name}': {str(e)}")
            return len(unique_values) > 1

    def resolve_conflict(
        self,
        field_name: str,
        field_definition: FieldDefinition,
        candidates: List[Dict]
    ) -> str:
        """
        Resolve value conflicts using Mistral with enhanced context
        """
        candidate_values = [c['value'] for c in candidates]
        
        # Build candidate details with context
        candidate_details = []
        for i, candidate in enumerate(candidates, 1):
            candidate_details.append(
                f"OPTION {i}: '{candidate['value']}'"
                f"\n  - Source Block: {candidate['block_id']}"
                f"\n  - OCR Confidence: {candidate['confidence']:.2f}"
                f"\n  - Context: \"{candidate['block_text'][:150]}\""
            )
        
        prompt = f"""
FIELD: {field_name}
DESCRIPTION: {field_definition.description}
DATA TYPE: {field_definition.data_type}
SYNONYMS: {", ".join(field_definition.synonyms)}
CONTEXT KEYWORDS: {", ".join(field_definition.context_keywords)}

Resolve conflict between these candidate values:

{chr(10).join(candidate_details)}

SELECTION CRITERIA:
1. Match to field description: "{field_definition.description}"
2. Consider data type: {field_definition.data_type}
3. Higher OCR confidence is better
4. More complete context is better
5. Consistency with context keywords

MANDATORY: Select EXACTLY ONE candidate value from:
{chr(10).join(f"• {val}" for val in candidate_values)}

REQUIRED RESPONSE FORMAT:
{{"selected_value": "exact_candidate_value"}}

Respond with ONLY the JSON object.
"""

        try:
            response = self.mapper_model.generate(prompt)
            self.mistral_conflict_resolution_calls += 1
            
            # Parse JSON response
            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    result_data = json.loads(json_match.group(0))
                    selected_value = result_data.get("selected_value", "")
                else:
                    raise json.JSONDecodeError("No JSON found", response, 0)
            except json.JSONDecodeError as e:
                logger.warning(f"Conflict resolution returned invalid JSON: {response[:200]}... Error: {str(e)}")
                selected_value = ""
            
            # Validate selection
            if selected_value in candidate_values:
                self.conflicts_resolved += 1
                logger.info(f"Conflict resolved: Selected '{selected_value}' for '{field_name}'")
                return selected_value
            
            # Invalid selection - use fallback
            fallback_value = Counter(candidate_values).most_common(1)[0][0]
            logger.warning(
                f"Invalid selection '{selected_value}'. Using fallback: '{fallback_value}'"
            )
            return fallback_value
            
        except Exception as e:
            logger.error(f"Conflict resolution failed for '{field_name}': {str(e)}")
            return candidates[0]["value"]

    def process_document_blocks(
       self,
       blocks: List[BlockData],
        schema: SchemaModel,
        image_path: str = None
        
    ) -> Tuple[Dict[str, Any], Dict[str, List[Dict]]]:

        """
        Process all document blocks with enhanced schema analysis
        """
        # Enhance schema with field analysis
        schema = self.analyze_schema_fields(schema)
        logger.info(f"Schema enhancement complete: {self.tinyllama_field_analysis_calls} fields analyzed")
        
        # Log enhanced schema for debugging
        logger.info("Enhanced schema fields:")
        for field_name, field_def in schema.fields.items():
            logger.info(f" - {field_name}:")
            logger.info(f"   Description: {field_def.description}")
            logger.info(f"   Synonyms: {', '.join(field_def.synonyms)}")
            logger.info(f"   Context Keywords: {', '.join(field_def.context_keywords)}")
        
        # Initialize data structures
        all_candidates = {field_name: [] for field_name in schema.fields}
        
        logger.info(f"Processing {len(blocks)} blocks for {len(schema.fields)} fields")
        
        for block in blocks:
            # Extract ALL fields from this block
            field_results = self.extract_fields_from_block(
                block_text=block.text,
                schema=schema,
                block_id=block.block_id
            )
            
            # Process results
            for field_name, result in field_results.items():
                if result.exists and result.extracted_value.strip():
                    all_candidates[field_name].append({
                        "value": result.extracted_value,
                        "block_id": block.block_id,
                        "confidence": block.confidence,
                        "bounding_box": block.bounding_box,
                        "block_text": block.text
                    })
        
        # Count fields with candidates
        self.fields_with_candidates = sum(1 for candidates in all_candidates.values() if candidates)
        
        # Build initial values (last extracted value per field)
        initial_values = {}
        for field_name, candidates in all_candidates.items():
            initial_values[field_name] = candidates[-1]["value"] if candidates else None
        
        logger.info(f"Extraction complete - Mistral calls: {self.mistral_extraction_calls}")
        logger.info(f"Fields with candidates: {self.fields_with_candidates}/{len(schema.fields)}")
        return initial_values, all_candidates

    def resolve_value_conflicts(
        self,
        all_candidates: Dict[str, List[Dict]],
        schema: SchemaModel
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Validate and resolve value conflicts
        """
        resolved_values = {}
        source_mapping = {}
        
        logger.info("Starting conflict resolution...")
        
        for field_name, candidates in all_candidates.items():
            field_def = schema.fields[field_name]
            
            if not candidates:
                resolved_values[field_name] = None
                source_mapping[field_name] = None
                continue
                
            if len(candidates) == 1:
                resolved_values[field_name] = candidates[0]["value"]
                source_mapping[field_name] = candidates[0]["block_id"]
                continue
            
            # Multiple candidates - validate
            callback_needed = self.validate_candidates(
                field_name=field_name,
                candidates=candidates
            )
            
            if callback_needed:
                # Conflict confirmed - resolve
                self.conflicts_detected += 1
                chosen_value = self.resolve_conflict(
                    field_name=field_name,
                    field_definition=field_def,
                    candidates=candidates
                )
                
                # Find source block
                chosen_candidate = next(
                    (c for c in candidates if c["value"] == chosen_value),
                    candidates[0]
                )
                
                resolved_values[field_name] = chosen_value
                source_mapping[field_name] = chosen_candidate["block_id"]
            else:
                # Use the candidate with highest OCR confidence
                best_candidate = max(candidates, key=lambda x: x["confidence"])
                resolved_values[field_name] = best_candidate["value"]
                source_mapping[field_name] = best_candidate["block_id"]
        
        logger.info(f"Conflict resolution complete:")
        logger.info(f"- Fields processed: {len(all_candidates)}")
        logger.info(f"- Conflicts detected: {self.conflicts_detected}")
        logger.info(f"- Conflicts resolved: {self.conflicts_resolved}")
        
        return resolved_values, source_mapping

    def save_results(
        self,
        enhanced_schema: SchemaModel,
        initial_values: Dict[str, Any],
        all_candidates: Dict[str, List[Dict]],
        final_values: Dict[str, Any],
        source_mapping: Dict[str, str],
        output_path: str
    ):
        """Save comprehensive mapping results"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Enhanced results with full metadata
        final_results = {
            "enhanced_schema": enhanced_schema.model_dump(),
            "structured_data": final_values,
            "source_traceability": source_mapping,
            "processing_metadata": {
                "mistral_extractions": self.mistral_extraction_calls,
                "mistral_conflict_resolutions": self.mistral_conflict_resolution_calls,
                "tinyllama_validations": self.tinyllama_validation_calls,
                "tinyllama_field_analyses": self.tinyllama_field_analysis_calls,
                "fields_with_candidates": self.fields_with_candidates,
                "conflicts_detected": self.conflicts_detected,
                "conflicts_resolved": self.conflicts_resolved
            },
            "all_candidates": all_candidates
        }
        
        with open(f"{output_path}_full_results.json", "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        # Save values only
        with open(f"{output_path}_values.json", "w", encoding="utf-8") as f:
            json.dump(final_values, f, indent=2, ensure_ascii=False)
        
        # Save enhanced schema separately
        with open(f"{output_path}_enhanced_schema.json", "w", encoding="utf-8") as f:
            json.dump(enhanced_schema.model_dump(), f, indent=2, ensure_ascii=False)
        
        logger.info("Results saved:")
        logger.info(f"- Full results: {os.path.abspath(output_path)}_full_results.json")
        logger.info(f"- Values: {os.path.abspath(output_path)}_values.json")
        logger.info(f"- Enhanced schema: {os.path.abspath(output_path)}_enhanced_schema.json")

def load_schema(schema_path: str) -> SchemaModel:
    """Load and validate schema definition"""
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_data = json.load(f)
        
        # Clean field names by removing trailing punctuation
        cleaned_fields = {}
        for field_name, field_def in schema_data.items():
            # Remove trailing colons and spaces
            clean_name = re.sub(r'[: ]+$', '', field_name)
            cleaned_fields[clean_name] = field_def
        
        # Convert to enhanced field definitions
        fields = {}
        for field_name, field_def in cleaned_fields.items():
            fields[field_name] = FieldDefinition(
                description=field_def.get("description", ""),
                data_type=field_def.get("data_type", "string"),
                synonyms=field_def.get("synonyms", []),
                extraction_instructions=field_def.get("extraction_instructions", ""),
                context_keywords=field_def.get("context_keywords", [])
            )
            
        return SchemaModel(fields=fields)
    except (FileNotFoundError, json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Schema load error: {str(e)}")
        raise

def load_document_blocks(blocks_path: str) -> List[BlockData]:
    """Load OCR block data with preprocessing"""
    try:
        with open(blocks_path, "r", encoding="utf-8") as f:
            blocks_data = json.load(f)
        
        processed_blocks = []
        for block in blocks_data:
            # Ensure proper types
            if "block_id" in block and isinstance(block["block_id"], int):
                block["block_id"] = str(block["block_id"])
            if "confidence" in block and isinstance(block["confidence"], int):
                block["confidence"] = float(block["confidence"])
            processed_blocks.append(block)
        
        return [BlockData(**block) for block in processed_blocks]
    except (FileNotFoundError, json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Blocks load error: {str(e)}")
        raise

def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description="Enhanced Schema Mapping with Field Analysis"
    )
    parser.add_argument("--image", help="Source document image path (optional)")
    parser.add_argument("--blocks", required=True, help="Path to OCR block data JSON")
    parser.add_argument("--schema", required=True, help="Path to schema definition JSON")
    parser.add_argument("--output", default="schema_mapping_results", help="Output file base name")
    parser.add_argument("--mistral-model", required=True, help="Path to Mistral model")
    parser.add_argument("--tinyllama-model", required=True, help="Path to TinyLlama model")
    
    args = parser.parse_args()
    
    try:
        # Load input data
        logger.info("="*60)
        logger.info("LOADING INPUT DATA")
        logger.info("="*60)
        schema = load_schema(args.schema)
        blocks = load_document_blocks(args.blocks)
        
        logger.info(f"Loaded schema with {len(schema.fields)} fields")
        logger.info(f"Loaded {len(blocks)} document blocks")
        
        # Initialize mapper
        logger.info("="*60)
        logger.info("INITIALIZING ENHANCED SCHEMA MAPPER")
        logger.info("="*60)
        mapper = SchemaMapper(
            mapping_model_path=args.mistral_model,
            validator_model_path=args.tinyllama_model
        )
        
        # Process document
        logger.info("="*60)
        logger.info("PROCESSING DOCUMENT BLOCKS")
        logger.info("="*60)
        initial_values, all_candidates = mapper.process_document_blocks(
            blocks=blocks,
            schema=schema,
            image_path=args.image
        )
        
        # Resolve conflicts
        logger.info("="*60)
        logger.info("RESOLVING VALUE CONFLICTS")
        logger.info("="*60)
        final_values, source_mapping = mapper.resolve_value_conflicts(
            all_candidates=all_candidates,
            schema=schema
        )
        
        # Save results
        logger.info("="*60)
        logger.info("SAVING RESULTS")
        logger.info("="*60)
        mapper.save_results(
            enhanced_schema=schema,
            initial_values=initial_values,
            all_candidates=all_candidates,
            final_values=final_values,
            source_mapping=source_mapping,
            output_path=args.output
        )
        
        # Print summary
        print("\n" + "="*70)
        print("ENHANCED SCHEMA MAPPING COMPLETE")
        print("="*70)
        print(f"Document: {args.image or 'N/A'}")
        print(f"Blocks processed: {len(blocks)}")
        print(f"Schema fields: {len(schema.fields)}")
        print(f"Fields with candidates: {mapper.fields_with_candidates}/{len(schema.fields)}")
        print()
        print("PROCESSING METRICS:")
        print(f"- Mistral extraction calls: {mapper.mistral_extraction_calls}")
        print(f"- Mistral conflict resolutions: {mapper.mistral_conflict_resolution_calls}")
        print(f"- TinyLlama field analyses: {mapper.tinyllama_field_analysis_calls}")
        print(f"- TinyLlama validations: {mapper.tinyllama_validation_calls}")
        print(f"- Conflicts detected: {mapper.conflicts_detected}")
        print(f"- Conflicts resolved: {mapper.conflicts_resolved}")
        print()
        print("FINAL STRUCTURED DATA:")
        print(json.dumps(final_values, indent=2, ensure_ascii=False))
        print()
        print(f"Results saved to: {os.path.abspath(args.output)}_*")
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()