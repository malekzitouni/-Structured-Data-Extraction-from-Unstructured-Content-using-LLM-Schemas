# Structured Data Extraction from Unstructured Content using LLM Schemas

An intelligent document processing system that transforms complex, unstructured KYC documents into clean, structured data using Large Language Models and advanced computer vision techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)  
- [Solution Architecture](#solution-architecture)
- [Technical Approach](#technical-approach)
- [Key Features](#key-features)
- [Tools & Technologies](#tools--technologies)
- [Installation](#installation)
- [Usage](#usage)
- [Results & Impact](#results--impact)
- [Future Roadmap](#future-roadmap)
- [Contributing](#contributing)

## 🎯 Overview

This project addresses the challenge of extracting structured data from unstructured KYC (Know Your Customer) documents that come in countless formats - scanned PDFs, images, tables, handwritten notes. Traditional systems fail when layouts or labels change, making data extraction messy and inconsistent.

Our LLM-powered adaptive extraction system takes:
- An unstructured document (PDF/image)
- A dynamic schema of required fields

And outputs structured results with no manual reconfiguration, scalable for evolving formats.

## 🚩 Problem Statement

KYC documents for credit applications present several challenges:
- **Multiple formats**: Scanned PDFs, images, tables, handwritten notes
- **Inconsistent layouts**: Traditional systems fail when layouts or labels change
- **Manual processing**: Time-consuming and error-prone
- **Scalability issues**: Difficulty adapting to new document formats

## 🏗️ Solution Architecture

Our system transforms complex, unstructured KYC documents into clean, structured data through a two-phase approach:

### Phase 1: OCR Extraction Pipeline
1. **Document Layout Analysis (DLA)** - Structural element detection
2. **Table Structure Recognition (TSR)** - Table organization understanding  
3. **OCR Extraction** - Text extraction from detected regions

### Phase 2: Schema-Based Mapping Pipeline
4. **Schema Enhancement** - Field preparation and validation
5. **Field Extraction** - AI-powered content mapping
6. **Conflict Resolution** - Intelligent duplicate handling

## 🔧 Technical Approach

### Document Layout Analysis (DLA)

**Framework**: [Docling](https://github.com/DS4SD/docling) - Open-source Python framework for document structure analysis

**Models Used**:
- `ds4sd/docling-layout-old`
- `ds4sd/docling-layout-heron`
- `ds4sd/docling-layout-heron-101`
- `ds4sd/docling-layout-egret-medium`
- `ds4sd/docling-layout-egret-large`

**Purpose**:
- Identify text blocks, tables, headers, lists, and forms
- Provide structured layout metadata
- Ensure layout-agnostic extraction

**Output**: Clean, machine-readable document structure with bounding box coordinates

### Table Structure Recognition (TSR)

**Tool**: Microsoft's Table Transformer (TableFormer Model)

**Capabilities**:
- Detect table boundaries
- Identify rows, columns, and cell coordinates
- Preserve logical reading order
- Support multi-page and irregular table layouts

### OCR Extraction

**Engine**: Nanonets-OCR-s

**Key Features**:
- LaTeX equation recognition
- Intelligent image description with structured `<img>` tags
- Signature detection & isolation
- Watermark extraction
- Complex table extraction (Markdown & HTML output)

**Process**:
1. Load document image and layout JSON
2. Crop image regions using bounding boxes
3. Apply custom prompt templates per block type
4. Extract text with confidence scores

### Schema-Based Mapping

**AI Models**:
- **Primary**: Mistral-7B-Instruct-v0.2 (Main extraction engine)
- **Secondary**: TinyLlama-1.1B (Validation & analysis)

**Process**:
1. **Schema Enhancement**: Field preparation with regex patterns
2. **Field Extraction**: M×N grid processing (M schema fields, N text blocks)
3. **Conflict Resolution**: Multi-step scoring and validation system

## ✨ Key Features

- **Adaptive Layout Processing**: Handles various document formats automatically
- **Dynamic Schema Mapping**: Flexible field extraction based on user-defined schemas
- **Intelligent Conflict Resolution**: Advanced scoring system for duplicate detection
- **Multi-Modal Understanding**: Combines computer vision and NLP
- **GPU Acceleration**: CUDA support with 4-bit quantization for efficiency
- **Comprehensive Logging**: Full system monitoring and debugging

## 🛠️ Tools & Technologies

### Deep Learning Framework
- **PyTorch**: Core ML framework
- **GPU Acceleration**: CUDA support
- **4-bit Quantization**: Memory-efficient model loading

### Natural Language Processing  
- **Transformers Library**: HuggingFace transformers
- **Text Generation Pipeline**: Structured text generation
- **Chat Templates**: Proper prompt formatting

### Data Validation & Processing
- **Pydantic**: Data validation and serialization
- **JSON Schema**: Structured data definitions  
- **Regular Expressions**: Pattern matching and text cleaning

### Programming Libraries
- **Python 3.8+**: Main programming language
- **Collections.Counter**: Frequency analysis
- **difflib.SequenceMatcher**: Text similarity calculations
- **Logging**: System monitoring

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/data-extraction-llm.git
cd data-extraction-llm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support (optional, for GPU acceleration)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🚀 Usage

### Basic Usage

```python
from data_extraction import DocumentProcessor

# Initialize the processor
processor = DocumentProcessor(
    schema_path="path/to/schema.json",
    models_config="path/to/models_config.yaml"
)

# Process a document
result = processor.extract(
    document_path="path/to/document.pdf",
    output_format="json"
)

print(result)
```

### Schema Definition

```json
{
    "fields": [
        {
            "name": "customer_name",
            "type": "string",
            "required": true,
            "description": "Full name of the customer"
        },
        {
            "name": "account_number", 
            "type": "string",
            "pattern": "^[0-9]{10,12}$",
            "required": true
        },
        {
            "name": "balance",
            "type": "number",
            "required": false
        }
    ]
}
```

### Command Line Interface

```bash
# Process single document
python -m data_extraction.cli process --input document.pdf --schema schema.json --output result.json

# Batch processing
python -m data_extraction.cli batch --input-dir documents/ --schema schema.json --output-dir results/
```

## 📊 Results & Impact

- **Accuracy**: Achieved high precision in field extraction across diverse document formats
- **Scalability**: Successfully processes documents with varying layouts without reconfiguration
- **Efficiency**: Significant reduction in manual data entry time
- **Adaptability**: Dynamic schema support enables quick adaptation to new document types

## 🔮 Future Roadmap

### Phase 1: Core Enhancement
- [ ] Implement spatial relationship mapping
- [ ] Add confidence-based filtering
- [ ] Create pattern-matching database

### Phase 2: LLM Optimization  
- [ ] Implement batch processing for LLM calls
- [ ] Enhance conflict resolution logic
- [ ] Create validation pipeline

### Phase 3: Advanced Features
- [ ] Add schema learning capabilities
- [ ] Implement multi-modal understanding
- [ ] Create comprehensive testing framework
- [ ] Add support for more document formats
- [ ] Develop real-time processing capabilities

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Malek Zitouni**

## 🙏 Acknowledgments

- [Docling](https://github.com/DS4SD/docling) for document layout analysis
- Microsoft for Table Transformer
- Nanonets for OCR capabilities
- HuggingFace for transformer models
- The open-source community for various tools and libraries

---

⭐ If you find this project helpful, please consider giving it a star!

## 📞 Contact

For questions, suggestions, or collaboration opportunities, please open an issue or contact the maintainers.
