from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="CatalystAI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


# ─── Request Models ────────────────────────────────────────────────────────────

class ReactionRequest(BaseModel):
    reactant1: str
    reactant2: str


class SafetyRequest(BaseModel):
    chemical: str


class DrugDiscoveryRequest(BaseModel):
    target: str          # disease name, protein target, or biological pathway
    constraints: str = ""  # optional: e.g. "oral bioavailability, CNS penetration"


class PathwayRequest(BaseModel):
    start: str           # starting material
    target: str          # desired product
    constraints: str = ""  # optional: e.g. "avoid toxic reagents, green chemistry"


# ─── Gemini Helper ─────────────────────────────────────────────────────────────

async def ask_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set in environment.")

    async with httpx.AsyncClient(timeout=45) as client:
        resp = await client.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 1200, "temperature": 0.3}
            }
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Gemini API error: {resp.text}")

        data = resp.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            raise HTTPException(status_code=500, detail="Unexpected Gemini response format.")


# ─── Existing Endpoints ────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse("index.html")


@app.get("/molecule/{name}")
async def get_molecule(name: str):
    async with httpx.AsyncClient(timeout=15) as client:
        cid_resp = await client.get(
            f"{PUBCHEM_BASE}/compound/name/{name}/cids/JSON"
        )
        if cid_resp.status_code != 200:
            raise HTTPException(status_code=404, detail=f"Molecule '{name}' not found in PubChem.")

        cid = cid_resp.json()["IdentifierList"]["CID"][0]

        props_resp = await client.get(
            f"{PUBCHEM_BASE}/compound/cid/{cid}/property/"
            "MolecularFormula,MolecularWeight,IUPACName,IsomericSMILES,"
            "XLogP,HBondAcceptorCount,HBondDonorCount/JSON"
        )
        if props_resp.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to fetch molecule properties.")

        props = props_resp.json()["PropertyTable"]["Properties"][0]

        sdf_data = None
        for record_type in ["3d", "2d"]:
            sdf_resp = await client.get(
                f"{PUBCHEM_BASE}/compound/cid/{cid}/SDF?record_type={record_type}"
            )
            if sdf_resp.status_code == 200 and "V2000" in sdf_resp.text:
                sdf_data = sdf_resp.text
                break

        return {
            "cid": cid,
            "name": props.get("IUPACName", name),
            "formula": props.get("MolecularFormula"),
            "molecular_weight": props.get("MolecularWeight"),
            "smiles": props.get("IsomericSMILES"),
            "logp": props.get("XLogP"),
            "hb_acceptors": props.get("HBondAcceptorCount"),
            "hb_donors": props.get("HBondDonorCount"),
            "sdf": sdf_data,
        }


@app.get("/cid-to-name/{cid}")
async def cid_to_name(cid: int):
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"{PUBCHEM_BASE}/compound/cid/{cid}/property/IUPACName/JSON"
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=404, detail=f"CID {cid} not found.")
        props = resp.json()["PropertyTable"]["Properties"][0]
        return {"cid": cid, "name": props.get("IUPACName", str(cid))}


@app.get("/similar/{name}")
async def similar_molecules(name: str):
    async with httpx.AsyncClient(timeout=30) as client:
        # 1. Resolve name → CID
        cid_resp = await client.get(
            f"{PUBCHEM_BASE}/compound/name/{name}/cids/JSON"
        )
        if cid_resp.status_code != 200:
            raise HTTPException(status_code=404, detail=f"Molecule '{name}' not found.")
        source_cid = cid_resp.json()["IdentifierList"]["CID"][0]

        # 2. Get source name for display
        src_name_resp = await client.get(
            f"{PUBCHEM_BASE}/compound/cid/{source_cid}/property/IUPACName/JSON"
        )
        source_name = name
        if src_name_resp.status_code == 200:
            source_name = src_name_resp.json()["PropertyTable"]["Properties"][0].get("IUPACName", name)

        # 3. Find similar CIDs (Tanimoto ≥ 0.8, limit 8)
        sim_resp = await client.get(
            f"{PUBCHEM_BASE}/compound/fastsimilarity_2d/cid/{source_cid}/cids/JSON"
            "?Threshold=80&MaxRecords=9"
        )
        if sim_resp.status_code != 200:
            raise HTTPException(status_code=500, detail="PubChem similarity search failed.")
        cids = sim_resp.json().get("IdentifierList", {}).get("CID", [])
        # exclude the source itself
        cids = [c for c in cids if c != source_cid][:8]

        if not cids:
            return {"source": source_name, "similar": []}

        # 4. Fetch properties for all similar CIDs in one call
        cid_str = ",".join(map(str, cids))
        props_resp = await client.get(
            f"{PUBCHEM_BASE}/compound/cid/{cid_str}/property/"
            "MolecularFormula,MolecularWeight,IUPACName,IsomericSMILES/JSON"
        )
        if props_resp.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to fetch properties for similar compounds.")

        props_list = props_resp.json()["PropertyTable"]["Properties"]

        # 5. Ask Gemini for a one-line description of each
        names_for_gemini = [p.get("IUPACName") or p.get("IsomericSMILES", "?") for p in props_list]
        desc_prompt = (
            f"For each of these chemistry compounds, write ONE short sentence (max 15 words) "
            f"describing its primary use or biological role. "
            f"Respond as a numbered list matching the order given, nothing else.\n\n"
            + "\n".join(f"{i+1}. {n}" for i, n in enumerate(names_for_gemini))
        )
        descriptions = []
        try:
            desc_text = await ask_gemini(desc_prompt)
            for line in desc_text.strip().split("\n"):
                line = line.strip()
                if line and line[0].isdigit():
                    descriptions.append(line.split(".", 1)[-1].strip())
                elif line:
                    descriptions.append(line)
        except Exception:
            descriptions = [""] * len(props_list)

        # pad if needed
        while len(descriptions) < len(props_list):
            descriptions.append("")

        results = []
        for i, p in enumerate(props_list):
            results.append({
                "cid": p.get("CID"),
                "name": p.get("IUPACName", "Unknown"),
                "formula": p.get("MolecularFormula"),
                "molecular_weight": p.get("MolecularWeight"),
                "smiles": p.get("IsomericSMILES"),
                "description": descriptions[i],
            })

        return {"source": source_name, "similar": results}


@app.post("/reaction")
async def predict_reaction(req: ReactionRequest):
    prompt = f"""You are a chemistry expert. Analyze the reaction between these two reactants.

Reactant 1: {req.reactant1}
Reactant 2: {req.reactant2}

Respond in EXACTLY this format with no extra text before or after:

PRODUCTS: [chemical products formed]
TYPE: [reaction type: neutralization / combustion / redox / precipitation / esterification / etc]
CONDITIONS: [temperature, catalyst, pressure needed — or "Room temperature, no catalyst"]
EQUATION: [balanced chemical equation using proper symbols]
OBSERVATIONS: [visible signs: color change, gas evolved, precipitate formed, heat/light released]
MECHANISM: [2-3 sentences explaining what happens at the molecular/ionic level]

Be accurate, concise, and use proper chemistry terminology."""

    text = await ask_gemini(prompt)
    return {"analysis": text}


@app.post("/safety")
async def chemical_safety(req: SafetyRequest):
    prompt = f"""You are a chemical safety expert. Provide a complete safety and properties report.

Chemical: {req.chemical}

Respond in EXACTLY this format with no extra text before or after:

HAZARDS: [comma-separated list from: Flammable, Toxic, Corrosive, Irritant, Oxidizer, Explosive, Carcinogen — only applicable ones]
FIRST AID: [brief first aid for skin contact, eye contact, inhalation, ingestion]
STORAGE: [safe storage conditions]
PPE: [required protective equipment]
PHYSICAL PROPERTIES: [boiling point, melting point, color, odor, state at room temp]
CHEMICAL PROPERTIES: [pH if applicable, solubility, key reactivity notes]
COMMON USES: [top 3-4 uses]
DISPOSAL: [safe disposal method]

Be accurate and concise."""

    text = await ask_gemini(prompt)
    return {"report": text}


# ─── New Endpoint 1: Drug Discovery ───────────────────────────────────────────

@app.post("/drug-discovery")
async def drug_discovery(req: DrugDiscoveryRequest):
    """
    Given a disease/target, suggest candidate drug molecules with
    rationale, Lipinski rule-of-5 assessment, and mechanism of action.
    """
    constraints_line = f"\nAdditional constraints: {req.constraints}" if req.constraints.strip() else ""

    prompt = f"""You are a medicinal chemist and drug discovery expert.

Target disease / biological target: {req.target}{constraints_line}

Suggest 3 candidate drug molecules (can be existing drugs repurposed or novel scaffolds).
For each candidate, respond in EXACTLY this format (repeat the block 3 times, separated by ---):

CANDIDATE: [drug or compound name]
SMILES: [valid SMILES string]
MECHANISM: [how it acts on the target — 2 sentences]
LIPINSKI_MW: [molecular weight in Da — number only]
LIPINSKI_LOGP: [LogP value — number only]
LIPINSKI_HBD: [H-bond donors — number only]
LIPINSKI_HBA: [H-bond acceptors — number only]
LIPINSKI_PASS: [Yes or No — does it pass Lipinski rule of 5?]
DRUG_LIKENESS: [brief note on bioavailability, toxicity concerns, selectivity]
DEVELOPMENT_STAGE: [Approved / Clinical Trial / Preclinical / Conceptual]

---

Be scientifically accurate. Use real drugs where applicable."""

    text = await ask_gemini(prompt)
    return {"candidates": text}


# ─── New Endpoint 2: Reaction Pathway Engine ──────────────────────────────────

@app.post("/pathway")
async def reaction_pathway(req: PathwayRequest):
    """
    Generate a multi-step synthesis pathway from a starting material
    to a target product, with conditions, intermediates, and yield notes.
    """
    constraints_line = f"\nConstraints/preferences: {req.constraints}" if req.constraints.strip() else ""

    prompt = f"""You are an expert synthetic organic chemist.

Starting material: {req.start}
Target product: {req.target}{constraints_line}

Design a multi-step synthesis pathway. Respond in EXACTLY this format with no extra text:

FEASIBILITY: [High / Medium / Low — one word + 1-sentence reason]
TOTAL_STEPS: [number]
OVERALL_YIELD_EST: [estimated overall yield as a percentage range, e.g. 35-50%]

Then for each step, use this block (repeat as needed):

STEP: [step number]
FROM: [starting compound for this step]
REAGENTS: [reagents and solvents used]
CONDITIONS: [temperature, time, atmosphere e.g. N2, pressure]
TO: [product of this step / intermediate name]
YIELD_EST: [estimated yield for this step, e.g. 80-90%]
NOTES: [key considerations: selectivity, side reactions, purification needed]

After all steps:

ALTERNATIVES: [1-2 alternative routes or key variations worth considering]
GREEN_SCORE: [rate the route's environmental friendliness: Poor / Fair / Good — with brief reason]

Be accurate. Use real named reactions where applicable (e.g. Grignard, Wittig, Diels-Alder)."""

    text = await ask_gemini(prompt)
    return {"pathway": text}


# ─── New Endpoint 3: RDKit Property Predictor ─────────────────────────────────

@app.get("/rdkit-props")
async def rdkit_properties(smiles: str):
    """
    Compute cheminformatics descriptors for a SMILES string using RDKit.
    Returns MW, LogP, TPSA, rotatable bonds, ring count, aromatic rings,
    H-bond donors/acceptors, and Lipinski/Veber rule assessments.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, QED
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="RDKit is not installed. Run: pip install rdkit"
        )

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES string. Could not parse molecule.")

    mw          = round(Descriptors.ExactMolWt(mol), 3)
    logp        = round(Crippen.MolLogP(mol), 3)
    tpsa        = round(rdMolDescriptors.CalcTPSA(mol), 3)
    hbd         = rdMolDescriptors.CalcNumHBD(mol)
    hba         = rdMolDescriptors.CalcNumHBA(mol)
    rot_bonds   = rdMolDescriptors.CalcNumRotatableBonds(mol)
    ring_count  = rdMolDescriptors.CalcNumRings(mol)
    arom_rings  = rdMolDescriptors.CalcNumAromaticRings(mol)
    heavy_atoms = mol.GetNumHeavyAtoms()
    fsp3        = round(rdMolDescriptors.CalcFractionCSP3(mol), 3)
    qed_score   = round(QED.qed(mol), 3)

    # Lipinski Rule of Five
    lipinski_violations = sum([
        mw > 500,
        logp > 5,
        hbd > 5,
        hba > 10,
    ])
    lipinski_pass = lipinski_violations <= 1  # allow max 1 violation

    # Veber Rules (oral bioavailability)
    veber_pass = tpsa <= 140 and rot_bonds <= 10

    # Ghose filter
    ghose_pass = (
        160 <= mw <= 480 and
        -0.4 <= logp <= 5.6 and
        40 <= heavy_atoms <= 70
    )

    return {
        "smiles": smiles,
        "descriptors": {
            "molecular_weight":     mw,
            "logp":                 logp,
            "tpsa":                 tpsa,
            "hb_donors":            hbd,
            "hb_acceptors":         hba,
            "rotatable_bonds":      rot_bonds,
            "ring_count":           ring_count,
            "aromatic_rings":       arom_rings,
            "heavy_atom_count":     heavy_atoms,
            "fraction_csp3":        fsp3,
            "qed_score":            qed_score,
        },
        "drug_likeness": {
            "lipinski_pass":        lipinski_pass,
            "lipinski_violations":  lipinski_violations,
            "veber_pass":           veber_pass,
            "ghose_pass":           ghose_pass,
        }
    }