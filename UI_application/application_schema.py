import json
from pydantic import BaseModel, ValidationError
from typing import Dict, Any

class BaseSchema:
    """Handles schema validation and normalization"""
    
    SUPPORTED_TYPES = ["string", "integer", "float", "boolean", "date", "datetime"]
    
    @classmethod
    def validate_schema(cls, schema: Dict[str, str]) -> bool:
        """Validate user-provided schema structure"""
        if not isinstance(schema, dict):
            raise ValueError("Schema must be a dictionary")
            
        if len(schema) == 0:
            raise ValueError("Schema cannot be empty")
            
        for field, dtype in schema.items():
            if not isinstance(field, str):
                raise ValueError(f"Field name must be string: {field}")
                
            dtype_str = dtype.lower()
            if dtype_str not in cls.SUPPORTED_TYPES:
                raise ValueError(
                    f"Unsupported data type '{dtype}' for field '{field}'. "
                    f"Supported types: {', '.join(cls.SUPPORTED_TYPES)}"
                )
                
        return True
    
    @classmethod
    def normalize_schema(cls, schema: Dict[str, Any]) -> Dict[str, str]:
        """Normalize schema to standard format"""
        normalized = {}
        for field, dtype in schema.items():
            dtype_str = str(dtype).lower()
            # Map common aliases to standard types
            if dtype_str in ["str", "text"]:
                normalized[field] = "string"
            elif dtype_str in ["int", "number"]:
                normalized[field] = "integer"
            elif dtype_str in ["bool"]:
                normalized[field] = "boolean"
            elif dtype_str in ["time"]:
                normalized[field] = "datetime"
            else:
                normalized[field] = dtype_str
        return normalized
    
    @classmethod
    def example_schema(cls) -> Dict[str, str]:
        """Return example schema for UI"""
        return {
            "full_name": "string",
            "account_number": "integer",
            "account_type": "string",
            "opening_date": "date",
            "initial_deposit": "float",
            "is_joint_account": "boolean"
        }