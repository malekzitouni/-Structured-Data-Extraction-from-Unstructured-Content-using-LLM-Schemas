import json
import logging
from typing import Dict, Any, List
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceEndpoint

logger = logging.getLogger("SchemaMapper")

class SchemaMapper:
    def __init__(self, llm_model: str = "mistralai/Mistral-7B-Instruct-v0.1"):
        self.llm = self._load_llm(llm_model)
        self.prompt_template = self._create_prompt_template()
        
    def _load_llm(self, model_name: str):
        return HuggingFaceEndpoint(
            repo_id=model_name,
            temperature=0.1,
            max_length=2048,
            top_p=0.95,
            repetition_penalty=1.15
        )
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create the prompt template for schema mapping"""
        return PromptTemplate(
            input_variables=["extracted_data", "schema_definition"],
            template="""You are an expert in structured data extraction. 
Map the extracted information to the required schema fields exactly as specified.

### EXTRACTED DATA:
{extracted_data}

### TARGET SCHEMA:
{schema_definition}

### INSTRUCTIONS:
1. Extract ONLY values present in the extracted data
2. Use null (None) for missing values
3. Preserve exact data formatting (case, punctuation, etc.)
4. Output MUST be valid JSON matching the schema structure
5. Do NOT infer or calculate values

### OUTPUT FORMAT:
<Valid JSON matching the schema>"""
        )
    
    def map_to_schema(self, extracted_data: List[Dict[str, Any]], schema_definition: Dict[str, str]) -> Dict[str, Any]:
        """Map extracted data to target schema using LLM"""
        try:
            # Prepare input for LLM
            extracted_str = json.dumps(extracted_data, indent=2)
            schema_str = json.dumps(schema_definition, indent=2)
            
            # Create and run LLM chain
            llm_chain = LLMChain(
                llm=self.llm,
                prompt=self.prompt_template
            )
            result = llm_chain.run({
                "extracted_data": extracted_str,
                "schema_definition": schema_str
            })
            
            # Clean and parse JSON response
            json_start = result.find('{')
            json_end = result.rfind('}') + 1
            json_str = result[json_start:json_end]
            return json.loads(json_str)
            
        except Exception as e:
            logger.error(f"Mapping error: {str(e)}")
            return {}

def main():
    """Test mapping functionality"""
    # Sample extracted OCR data
    extracted_data = [
        {
            "label": "title",
            "text": "ACCOUNT OPENING FORM",
            "confidence": 0.95
        },
        {
            "label": "section-header",
            "text": "Personal Information",
            "confidence": 0.92
        },
        {
            "label": "text",
            "text": "Full Name: John David Anderson\nDate of Birth: 1985-07-15",
            "confidence": 0.89
        },
        {
            "label": "table",
            "text": "+-----------------+-----------------+\n| Account Type    | Savings Account |\n+-----------------+-----------------+\n| Account Number  | 1234567890      |\n+-----------------+-----------------+",
            "confidence": 0.93
        }
    ]
    
    # Sample schema definition
    schema_definition = {
        "full_name": "string",
        "date_of_birth": "string",
        "account_type": "string",
        "account_number": "integer"
    }
    
    # Perform mapping
    mapper = SchemaMapper()
    mapped_data = mapper.map_to_schema(extracted_data, schema_definition)
    
    print("\nMapped Result:")
    print(json.dumps(mapped_data, indent=2))

if __name__ == "__main__":
    main()