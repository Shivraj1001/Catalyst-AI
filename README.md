# CatalystAI — Chemistry AI Application

A full-featured chemistry analysis platform with 3D molecular visualization, reaction prediction, and molecular property analysis. Built for chemists, researchers, and chemistry students to explore molecules and reactions interactively.

## Problem Solved

Chemistry researchers need tools to visualize molecular structures, predict reactions, compare similar molecules, and analyze chemical properties. Existing tools are often expensive, complex, or require extensive installation. CatalystAI provides a simple web interface with powerful chemistry capabilities.

## Features

### 1. 3D Molecule Viewer
- Visualize molecular structures in interactive 3D
- Rotate, zoom, and pan molecular models
- Supports multiple molecular formats

### 2. Reaction Predictor
- Input reactants and predict products
- Uses Gemini API for intelligent prediction
- Shows reaction mechanism insights

### 3. Similar Molecules
- Search for structurally similar molecules
- Based on molecular fingerprints and structure matching
- Explore alternative compounds for research

### 4. 3D Comparison Viewer
- Side-by-side 3D visualization of multiple molecules
- Compare molecular structures and properties
- Identify structural similarities and differences

### 5. Chemical Pathway Engine
- Explore multi-step reaction pathways
- Understand reaction sequences
- Analyze synthetic routes

### 6. Molecular Property Analysis (RDKit)
- Calculate molecular weight, LogP, TPSA, etc.
- Predict drug-likeness (Lipinski's Rule of Five)
- Analyze molecular structure properties

## Tech Stack

- **Backend**: Python, FastAPI
- **Frontend**: HTML, CSS, JavaScript
- **Chemistry Libraries**: RDKit (molecular analysis), PubChem API (molecular data)
- **AI Integration**: Google Gemini API (reaction prediction)
- **3D Visualization**: Molecular viewer library (py3Dmol or similar)
- **Data Source**: PubChem database for chemical compounds

## Project Setup

### Prerequisites
- Python 3.9+
- Node.js (for frontend, if bundling)
- Gemini API key
- Internet connection (PubChem API calls)

### Installation

```bash
pip install fastapi uvicorn rdkit-pypi requests python-dotenv pydantic httpx
```

### Environment Variables

```
GEMINI_API_KEY=your_api_key
PUBCHEM_API_URL=https://pubchem.ncbi.nlm.nih.gov/rest/pug
```

### Running the Application

```bash
uvicorn main:app --reload
```

Visit `http://localhost:8000` in your browser.

## API Endpoints

```
GET  /api/molecule/{smiles}           - Get molecule data
POST /api/predict-reaction            - Predict reaction products
GET  /api/similar-molecules/{smiles}  - Find similar molecules
GET  /api/molecule-properties/{smiles} - Calculate properties
POST /api/pathway                     - Generate reaction pathway
```

## Key Learnings

- **FastAPI**: Building modern, high-performance Python APIs
- **Chemistry Libraries**: RDKit for molecular analysis and fingerprinting
- **API Integration**: Calling external APIs (PubChem, Gemini)
- **3D Visualization**: Rendering molecular structures in browser
- **LLM Integration**: Using Gemini for intelligent chemistry predictions
- **Frontend**: Building interactive UI for scientific applications

## Workflow Example

1. User enters a molecule SMILES string (e.g., `CCO` for ethanol)
2. System fetches molecule data from PubChem
3. 3D structure is rendered in the viewer
4. User can query similar molecules or predict reactions
5. Results displayed with visualizations and property data

## Libraries Used

- **RDKit**: Cheminformatics and molecular analysis
- **FastAPI**: Web framework
- **Requests**: HTTP requests to external APIs
- **Pydantic**: Data validation
- **PubChem API**: Molecular structure and property data
- **Gemini API**: Reaction prediction and insights

## Use Cases

- Chemistry education and learning
- Molecular research and drug discovery
- Organic synthesis planning
- Molecular property prediction
- Reaction mechanism exploration
- Academic research

## Limitations & Future Work

- Prediction accuracy depends on Gemini API quality
- Limited to small molecules (not polymers/proteins)
- No real-time reaction data validation
- Could add: molecule drawing interface, batch operations, export options, literature integration

## Author

Shivraj | 2nd Year B.Tech CSE/AIML | College of Engineering, Pune
GitHub: github.com/Shivraj1001/catalystai
