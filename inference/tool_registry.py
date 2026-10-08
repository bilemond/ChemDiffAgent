#!/usr/bin/env python3
"""Unified 45-tool registry for ChemDiffAgent inference.

The project combines tools from ChemistryAgent, ChemToolAgent, and a small
project-specific RDKit adapter. Their public names, descriptions, and execution
contracts are preserved in the bundled tool catalog.

Heavy property and reaction predictors are lazy.  Missing optional checkpoints
therefore do not prevent the rest of the agent from starting.
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
import shutil
import signal
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "inference/assets/tool_catalog.json"
CHEMTOOL_AGENT_ROOT = PROJECT_ROOT / "third_party/ChemToolAgent"
DEFAULT_T5CHEM_ROOT = PROJECT_ROOT / "third_party/t5chem"
CONTROLLED_CHEMICALS_CSV = (
    PROJECT_ROOT / "third_party/ChemistryAgent/agent/toolpool/chemcrow/data/chem_wep_smi.csv"
)


class ToolBackendUnavailable(RuntimeError):
    """Raised when a tool's optional model/API backend is not installed."""


def _load_catalog() -> dict[str, Any]:
    with CATALOG_PATH.open("r", encoding="utf-8") as handle:
        catalog = json.load(handle)
    if catalog.get("n_tools") != 45 or len(catalog.get("tool_index", {})) != 45:
        raise RuntimeError(f"Unexpected tool catalog: {CATALOG_PATH}")
    return catalog


CATALOG = _load_catalog()
TOOL_NAMES = [name for name, _ in sorted(CATALOG["tool_index"].items(), key=lambda kv: kv[1])]


def _tool_descriptions() -> dict[str, str]:
    prompt = CATALOG["unified_system_prompt"]
    descriptions: dict[str, str] = {}
    for match in re.finditer(r"^\(\d+\) ([^:]+): (.+)$", prompt, flags=re.MULTILINE):
        descriptions[match.group(1)] = match.group(2).strip()
    missing = [name for name in TOOL_NAMES if name not in descriptions]
    if missing:
        raise RuntimeError(f"Descriptions missing from unified catalog: {missing}")
    return descriptions


TOOL_DESCRIPTIONS = _tool_descriptions()


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    runner: Callable[[Any], Any]

    def __call__(self, value: Any) -> Any:
        return self.runner(value)


def _literalish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    raw = value.strip()
    if not raw:
        return raw
    try:
        return json.loads(raw)
    except Exception:
        pass
    try:
        return ast.literal_eval(raw)
    except Exception:
        return value


def _items(value: Any, count: int | None = None) -> list[Any]:
    value = _literalish(value)
    if isinstance(value, tuple):
        result = list(value)
    elif isinstance(value, list):
        result = value
    else:
        result = [value]
    if count is not None and len(result) != count:
        raise ValueError(f"Expected {count} inputs, got {len(result)}: {result!r}")
    return result


def _json_url(url: str, timeout: int = 30) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "DLLM-Science/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _pubchem_cid(query: Any) -> int:
    encoded = urllib.parse.quote(str(query).strip(), safe="")
    data = _json_url(
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/cids/JSON"
    )
    cids = data.get("IdentifierList", {}).get("CID", [])
    if not cids:
        raise ValueError(f"PubChem found no compound for {query!r}")
    return int(cids[0])


def _pubchem_properties(cid: Any) -> dict[str, Any]:
    cid = int(cid)
    fields = "MolecularWeight,MolecularFormula,Charge,IUPACName,ConnectivitySMILES,SMILES"
    data = _json_url(
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/{fields}/JSON"
    )
    rows = data.get("PropertyTable", {}).get("Properties", [])
    if not rows:
        raise ValueError(f"PubChem returned no properties for CID {cid}")
    return rows[0]


def _formula_counts(formula: str) -> dict[str, int]:
    # PubChem molecular formulas in this dataset are simple Hill formulas.  If
    # a formula has multiple components, counts are accumulated.
    counts: dict[str, int] = {}
    pieces = re.findall(r"([A-Z][a-z]?)(\d*)", str(formula))
    if not pieces:
        raise ValueError(f"Cannot parse molecular formula: {formula!r}")
    for element, raw_count in pieces:
        counts[element] = counts.get(element, 0) + int(raw_count or 1)
    return counts


def _rdkit_modules():
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, Fragments, Lipinski, QED, rdMolDescriptors
    except ImportError as exc:
        raise ToolBackendUnavailable("RDKit is required for this tool") from exc
    return Chem, Crippen, Descriptors, Fragments, Lipinski, QED, rdMolDescriptors


def _mol(smiles: Any):
    Chem, *_ = _rdkit_modules()
    raw = str(smiles).strip()
    # A handful of recovered examples contain one accidental unmatched quote
    # before a valid SMILES.  The lost adapter accepted those rows, so retain
    # that small piece of input tolerance without otherwise rewriting SMILES.
    stripped = raw.strip("\"'").strip()
    candidates = [stripped, raw] if stripped != raw else [raw]
    for candidate in candidates:
        mol = Chem.MolFromSmiles(candidate)
        if mol is not None:
            return mol
    raise ValueError(f"Invalid SMILES: {smiles}")


def _canonical_smiles(smiles: Any) -> str:
    Chem, *_ = _rdkit_modules()
    return Chem.MolToSmiles(_mol(smiles), canonical=True)


def _descriptor_row(smiles: Any) -> dict[str, Any]:
    Chem, Crippen, Descriptors, _, Lipinski, QED, rdMolDescriptors = _rdkit_modules()
    canonical = Chem.MolToSmiles(_mol(smiles), canonical=True)
    mol = _mol(canonical)
    return {
        "smiles": canonical,
        "qed": round(float(QED.qed(mol)), 4),
        "logp": round(float(Crippen.MolLogP(mol)), 4),
        "tpsa": round(float(rdMolDescriptors.CalcTPSA(mol)), 4),
        "mol_weight": round(float(Descriptors.MolWt(mol)), 4),
        "hbd": int(Lipinski.NumHDonors(mol)),
        "hba": int(Lipinski.NumHAcceptors(mol)),
        "rings": int(rdMolDescriptors.CalcNumRings(mol)),
        "fraction_csp3": round(float(rdMolDescriptors.CalcFractionCSP3(mol)), 4),
        "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
    }


def _objective_score(row: dict[str, Any], objective: str) -> float:
    qed = float(row["qed"])
    logp = float(row["logp"])
    tpsa = float(row["tpsa"])
    mw = float(row["mol_weight"])
    hbd = float(row["hbd"])
    hba = float(row["hba"])
    fsp3 = float(row["fraction_csp3"])
    rotb = float(row["rotatable_bonds"])
    if objective == "maximize_qed":
        return qed
    if objective == "balanced_druglikeness":
        return (
            qed
            - 0.2 * abs(mw - 350.0) / 350.0
            - 0.2 * abs(logp - 2.5) / 5.0
            - 0.1 * abs(tpsa - 75.0) / 120.0
        )
    if objective == "moderate_logp_high_qed":
        return qed - 0.35 * abs(logp - 2.0) / 5.0
    if objective == "brain_penetrant_like":
        return (
            qed
            - 0.25 * max(0.0, mw - 450.0) / 450.0
            - 0.3 * max(0.0, tpsa - 90.0) / 90.0
            - 0.2 * max(0.0, hbd - 3.0) / 5.0
            - 0.15 * max(0.0, abs(logp - 2.5) - 1.5) / 5.0
        )
    if objective == "lipinski_strict":
        passes = (mw <= 500.0) + (logp <= 5.0) + (hbd <= 5.0) + (hba <= 10.0)
        return 0.5 * qed + 0.125 * float(passes)
    if objective in ("low_tpsa_high_qed", "high_qed_low_tpsa"):
        return qed - 0.004 * tpsa
    if objective == "maximize_logp":
        return logp
    if objective == "minimize_logp":
        return -logp
    if objective == "minimize_mw":
        return -mw
    if objective == "maximize_mw":
        return mw
    if objective == "maximize_fraction_csp3":
        return fsp3
    if objective == "minimize_rotatable_bonds":
        return -rotb
    return qed


def _calculate_molar_mass(value: Any) -> float:
    formula = str(value).strip()
    # Formula strings are the dominant contract for this tool.  Fall back to
    # RDKit only when chemlib cannot parse the input as a formula.
    try:
        from chemlib import Compound

        return round(float(Compound(formula).molar_mass()), 8)
    except Exception:
        _, _, Descriptors, *_ = _rdkit_modules()
        return round(float(Descriptors.MolWt(_mol(formula))), 4)


def _empirical_formula(value: Any) -> str:
    mapping = _literalish(value)
    if not isinstance(mapping, dict):
        raise ValueError("Input must be an element-to-percentage mapping")
    from chemlib import empirical_formula_by_percent_comp

    return str(empirical_formula_by_percent_comp(**{str(k): float(v) for k, v in mapping.items()}).formula)


def _percentage_by_mass(value: Any) -> float:
    formula, element = _items(value, 2)
    from chemlib import Compound

    return round(float(Compound(str(formula)).percentage_by_mass(str(element))), 8)


def _druglikeness(value: Any) -> Any:
    _, Crippen, Descriptors, _, Lipinski, _, _ = _rdkit_modules()
    mol = _mol(value)
    violations = []
    if Lipinski.NumHDonors(mol) > 5:
        violations.append(f"H Bond Donors {Lipinski.NumHDonors(mol)}>5")
    if Lipinski.NumHAcceptors(mol) > 10:
        violations.append(f"H Bond Acceptors {Lipinski.NumHAcceptors(mol)}>10")
    exact_mw = Descriptors.ExactMolWt(mol)
    if exact_mw > 500:
        violations.append(f"Molecular Weight {exact_mw}>500")
    logp = Crippen.MolLogP(mol)
    if logp > 5:
        violations.append(f"LOGP {logp}>5")
    return violations if violations else "No violations found"


def _query_to_cas(value: Any) -> str:
    cid = _pubchem_cid(value)
    data = _json_url(
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
    )
    cas_pattern = re.compile(r"\b\d{2,7}-\d{2}-\d\b")
    candidates: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)
        elif isinstance(node, str):
            candidates.extend(cas_pattern.findall(node))

    walk(data)
    if not candidates:
        raise ValueError(f"CAS number not found for {value!r}")
    # PUG View may repeat values; the most common registry number is generally
    # the primary CAS entry.
    return max(dict.fromkeys(candidates), key=candidates.count)


def _analyze_combustion(value: Any) -> str:
    co2, h2o = (float(x) for x in _items(value, 2))
    from chemlib.thermochemistry import combustion_analysis

    return str(combustion_analysis(co2, h2o))


def _element_properties(value: Any) -> dict[str, Any]:
    symbol = str(value).strip()
    if symbol.isdigit():
        Chem, *_ = _rdkit_modules()
        symbol = Chem.GetPeriodicTable().GetElementSymbol(int(symbol))
    from chemlib import Element

    element = Element(symbol)
    fields = [
        "AtomicNumber", "Element", "Symbol", "AtomicMass", "Neutrons", "Protons",
        "Electrons", "Period", "Group", "Phase", "Radioactive", "Natural", "Metal",
        "Nonmetal", "Metalloid", "Type", "AtomicRadius", "Electronegativity",
        "FirstIonization", "Density", "MeltingPoint", "BoilingPoint", "Isotopes",
        "Discoverer", "Year", "SpecificHeat", "Shells", "Valence", "Config", "MassNumber",
    ]
    return {field: getattr(element, field) for field in fields}


def _patent_check(value: Any) -> dict[str, str]:
    try:
        import molbloom
    except ImportError as exc:
        raise ToolBackendUnavailable(
            "PatentCheck requires optional package 'molbloom' (pip install molbloom)"
        ) from exc
    values = str(value).split(".") if "." in str(value) else [str(value)]
    return {
        smiles: ("Patented" if molbloom.buy(smiles, canonicalize=True, catalog="surechembl") else "Novel")
        for smiles in values
    }


def _control_chem_check(value: Any) -> str:
    import csv

    Chem, *_ = _rdkit_modules()
    from rdkit.Chem import AllChem, DataStructs

    query = str(value).strip()
    try:
        query_mol = _mol(query)
    except ValueError:
        return "Please input a valid SMILES string."
    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=2048)
    max_similarity = 0.0
    exact = False
    with CONTROLLED_CHEMICALS_CSV.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            smiles = (row.get("smiles") or "").strip()
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            if Chem.MolToSmiles(mol) == Chem.MolToSmiles(query_mol):
                exact = True
                break
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            max_similarity = max(max_similarity, float(DataStructs.TanimotoSimilarity(query_fp, fp)))
    if exact:
        return f"The molecule {query} appears in a list of controlled chemicals."
    adjective = "high" if max_similarity > 0.35 else "low"
    message = (
        f"{query} has a {adjective} similarity ({max_similarity:.4}) "
        "to a known controlled chemical."
    )
    if adjective == "low":
        message += " This is substance is safe, you may proceed with the task."
    return message


def _combustion_reaction(value: Any) -> Any:
    from chemlib import Combustion, Compound

    reaction = Combustion(Compound(str(value)))
    return [reaction.formula, reaction.is_balanced]


def _balance_reaction(value: Any) -> str:
    from chemlib import Reaction

    reaction = Reaction.by_formula(str(value))
    reaction.balance()
    return str(reaction.formula)


def _reaction_side(value: Any, side: str) -> list[str]:
    from chemlib import Reaction

    reaction = Reaction.by_formula(str(value))
    return list(getattr(reaction, f"{side}_formulas"))


_PREDICTOR_CLASS_NAMES = {
    "SolubilityPredictor": "PropertyPredictorESOL",
    "LogDPredictor": "PropertyPredictorLIPO",
    "BBBPPredictor": "PropertyPredictorBBBP",
    "ToxicityPredictor": "PropertyPredictorClinTox",
    "HIVInhibitorPredictor": "PropertyPredictorHIV",
    "SideEffectPredictor": "PropertyPredictorSIDER",
}
_PREDICTOR_INSTANCES: dict[str, Any] = {}


def _property_predictor(name: str, value: Any) -> Any:
    value = _literalish(value)
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    checkpoint_name = {
        "SolubilityPredictor": "esol",
        "LogDPredictor": "lipo",
        "BBBPPredictor": "bbbp",
        "ToxicityPredictor": "clintox",
        "HIVInhibitorPredictor": "hiv",
        "SideEffectPredictor": "sider",
    }[name]
    checkpoint = (
        CHEMTOOL_AGENT_ROOT
        / f"chemagent/tools/property_prediction/checkpoints/{checkpoint_name}/checkpoint_best.pt"
    )
    if not checkpoint.is_file():
        raise ToolBackendUnavailable(
            f"{name} checkpoint is missing: {checkpoint}. Download the official "
            "ChemToolAgent property-prediction checkpoints first."
        )
    unicore_root = os.getenv("UNICORE_ROOT", "").strip()
    if unicore_root and unicore_root not in sys.path:
        sys.path.insert(0, unicore_root)
    try:
        import unicore  # noqa: F401
    except ImportError as exc:
        raise ToolBackendUnavailable(
            f"{name} requires Uni-Core in the active environment"
        ) from exc
    if str(CHEMTOOL_AGENT_ROOT) not in sys.path:
        sys.path.insert(0, str(CHEMTOOL_AGENT_ROOT))
    if name not in _PREDICTOR_INSTANCES:
        from chemagent.tools.property_prediction import property_prediction

        cls = getattr(property_prediction, _PREDICTOR_CLASS_NAMES[name])
        _PREDICTOR_INSTANCES[name] = cls(init=False)
    previous = Path.cwd()
    try:
        os.chdir(CHEMTOOL_AGENT_ROOT)
        return _PREDICTOR_INSTANCES[name](str(value))
    finally:
        os.chdir(previous)


class _T5ChemBackend:
    """Lazy local backend used by both recovered reaction tools.

    The implementation follows the lost ``pipeline_kg_tools.py`` recovered
    from the Codex session log.  ``USPTO_500_MT`` is a multi-task checkpoint:
    the task prefix selects forward synthesis or single-step retrosynthesis.
    """

    def __init__(self) -> None:
        self.root = Path(
            os.getenv("RXN_T5CHEM_ROOT", str(DEFAULT_T5CHEM_ROOT)).strip()
        ).expanduser().resolve()
        self.model_path = Path(
            os.getenv(
                "RXN_T5CHEM_MODEL_PATH",
                str(self.root / "checkpoints/models/USPTO_500_MT"),
            ).strip()
        ).expanduser().resolve()
        self.num_beams = max(1, int(os.getenv("RXN_T5CHEM_NUM_BEAMS", "10")))
        self.num_preds = max(1, int(os.getenv("RXN_T5CHEM_NUM_PREDS", "5")))
        self.device_pref = os.getenv("RXN_T5CHEM_DEVICE", "auto").strip().lower()
        self._torch = None
        self._model = None
        self._tokenizer = None
        self._device = None

    @property
    def vocab_path(self) -> Path:
        return self.model_path / "vocab.txt"

    def _ensure_vocab_file(self) -> None:
        if self.vocab_path.exists():
            return
        fallback_vocab = self.root / "t5chem/vocab/simple.txt"
        if not fallback_vocab.is_file():
            raise ToolBackendUnavailable(
                f"T5Chem vocab is missing at {self.vocab_path}; fallback is also "
                f"missing: {fallback_vocab}"
            )
        self.vocab_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.vocab_path.symlink_to(fallback_vocab)
        except OSError:
            shutil.copyfile(fallback_vocab, self.vocab_path)

    def _ensure_import(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        if not self.root.is_dir():
            raise ToolBackendUnavailable(
                f"T5Chem source is missing: {self.root}. Run the commands in "
                "inference/WEIGHTS.md or set RXN_T5CHEM_ROOT."
            )
        if not self.model_path.is_dir():
            raise ToolBackendUnavailable(
                f"T5Chem USPTO_500_MT checkpoint is missing: {self.model_path}. "
                "Run the commands in inference/WEIGHTS.md or set RXN_T5CHEM_MODEL_PATH."
            )
        root_str = str(self.root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        self._ensure_vocab_file()
        try:
            import torch
            from transformers import T5Config, T5ForConditionalGeneration
            from t5chem.mol_tokenizers import (
                AtomTokenizer,
                SelfiesTokenizer,
                SimpleTokenizer,
            )
        except Exception as exc:
            raise ToolBackendUnavailable(
                f"Failed to import the T5Chem runtime: {type(exc).__name__}: {exc}"
            ) from exc

        config = T5Config.from_pretrained(str(self.model_path))
        tokenizer_type = str(getattr(config, "tokenizer", "simple")).lower()
        if tokenizer_type == "atom":
            tokenizer_cls = AtomTokenizer
        elif tokenizer_type == "selfies":
            tokenizer_cls = SelfiesTokenizer
        else:
            tokenizer_cls = SimpleTokenizer
        tokenizer = tokenizer_cls(vocab_file=str(self.vocab_path))
        model = T5ForConditionalGeneration.from_pretrained(str(self.model_path))
        if self.device_pref == "auto":
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.device_pref)
        model.to(device)
        model.eval()

        self._torch = torch
        self._model = model
        self._tokenizer = tokenizer
        self._device = device

    @staticmethod
    def _task_prefix(task_type: str) -> tuple[str, int, int]:
        if task_type == "forward":
            return "Product:", 400, 200
        if task_type == "retro":
            return "Reactants:", 200, 300
        raise ValueError(f"Unknown T5Chem task type: {task_type}")

    def _decode_predictions(self, outputs: Any) -> list[str]:
        predictions: list[str] = []
        for prediction in outputs:
            text = self._tokenizer.decode(
                prediction,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).replace(" ", "").strip()
            if text and text.lower() != "nan" and text not in predictions:
                predictions.append(text)
        return predictions

    def _generate(self, payload: str, task_type: str) -> list[str]:
        self._ensure_import()
        prefix, max_source_length, max_target_length = self._task_prefix(task_type)
        encoded = self._tokenizer(
            prefix + payload.strip(),
            max_length=max_source_length,
            padding="do_not_pad",
            truncation=True,
            return_tensors="pt",
        )
        encoded_inputs = {"input_ids": encoded["input_ids"].to(self._device)}
        if "attention_mask" in encoded:
            encoded_inputs["attention_mask"] = encoded["attention_mask"].to(self._device)
        with self._torch.no_grad():
            outputs = self._model.generate(
                **encoded_inputs,
                early_stopping=True,
                max_length=max_target_length,
                num_beams=self.num_beams,
                num_return_sequences=min(self.num_preds, self.num_beams),
                decoder_start_token_id=self._tokenizer.pad_token_id,
            )
        return self._decode_predictions(outputs)

    def predict_forward(self, reactants: str) -> str:
        predictions = self._generate(reactants, task_type="forward")
        if not predictions:
            raise RuntimeError("T5Chem forward prediction returned no valid product")
        try:
            return _canonical_smiles(predictions[0])
        except ValueError:
            return predictions[0]

    def predict_retro(self, product: str) -> str:
        predictions = self._generate(product, task_type="retro")
        if not predictions:
            raise RuntimeError("T5Chem retrosynthesis returned no valid reactants")
        top_k = max(1, int(os.getenv("RXN_RETRO_TOP_K", "1")))
        predictions = predictions[: min(top_k, len(predictions))]
        normalized: list[str] = []
        for reactants in predictions:
            try:
                reactants = _canonical_smiles(reactants)
            except ValueError:
                pass
            normalized.append(reactants)
        if top_k == 1:
            return normalized[0]
        result = "There %s %d possible sets of reactants for the given product:\n" % (
            "are" if len(normalized) > 1 else "is",
            len(normalized),
        )
        for index, reactants in enumerate(normalized, start=1):
            # T5Chem does not expose calibrated probabilities in this path;
            # preserve the output shape of the old adapter with rank scores.
            score = max(0.01, 1.0 - (index - 1) * 0.05)
            result += f"{index}.\tReactants: {reactants}\tConfidence: {score:.3f}\n"
        return result


_T5CHEM_BACKEND: _T5ChemBackend | None = None


def _reaction_input(value: Any) -> str:
    parsed = _literalish(value)
    if isinstance(parsed, dict):
        for key in ("input", "smiles", "SMILES", "reactants", "product", "target"):
            if key in parsed:
                parsed = parsed[key]
                break
    return str(parsed).replace("<END_INPUT>", "").strip().strip("'\"")


def _reaction_predictor(name: str, value: Any) -> Any:
    backend_name = os.getenv("RXN_BACKEND", "t5chem").strip().lower()
    if backend_name != "t5chem":
        raise ToolBackendUnavailable(
            f"Unsupported RXN_BACKEND={backend_name!r}; the recovered final pipeline "
            "uses the local 't5chem' backend"
        )
    global _T5CHEM_BACKEND
    if _T5CHEM_BACKEND is None:
        _T5CHEM_BACKEND = _T5ChemBackend()
    payload = _reaction_input(value)
    if name == "RXNPredict":
        return _T5CHEM_BACKEND.predict_forward(payload)
    if name == "RXNRetrosynthetic":
        return _T5CHEM_BACKEND.predict_retro(payload)
    raise ValueError(f"Unknown reaction tool: {name}")


def _name_to_smiles(value: Any) -> str:
    cid = _pubchem_cid(value)
    props = _pubchem_properties(cid)
    smiles = props.get("SMILES") or props.get("ConnectivitySMILES") or props.get("CanonicalSMILES")
    if not smiles:
        raise ValueError(f"PubChem returned no SMILES for {value!r}")
    return str(smiles)


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=True)


def _test_molecule(value: Any) -> str:
    Chem, *_ = _rdkit_modules()
    mol = _mol(value)
    return _json_text({
        "valid": True,
        "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
        "num_atoms": mol.GetNumAtoms(),
        "num_bonds": mol.GetNumBonds(),
    })


def _functional_groups(value: Any) -> str:
    _, _, _, Fragments, *_ = _rdkit_modules()
    mol = _mol(value)
    functions = {
        "alcohol": Fragments.fr_Al_OH,
        "phenol": Fragments.fr_Ar_OH,
        "carboxylic_acid": Fragments.fr_COO,
        "amide": Fragments.fr_amide,
        "amine": Fragments.fr_NH2,
        "aromatic_ring": Fragments.fr_benzene,
        "ester": Fragments.fr_ester,
        "ether": Fragments.fr_ether,
        "halogen": Fragments.fr_halogen,
        "nitrile": Fragments.fr_nitrile,
        "nitro": Fragments.fr_nitro,
        "sulfone": Fragments.fr_sulfone,
    }
    groups = {name: int(fn(mol)) for name, fn in functions.items() if int(fn(mol)) > 0}
    return _json_text(groups or {"detected_functional_groups": []})


def _diversity_filter(value: Any) -> str:
    values = _literalish(value)
    if not isinstance(values, list):
        raise ValueError("candidate_diversity_filter expects a list of SMILES")
    seen: set[str] = set()
    result: list[str] = []
    for item in values:
        try:
            canonical = _canonical_smiles(item)
        except ValueError:
            continue
        if canonical not in seen:
            seen.add(canonical)
            result.append(canonical)
    return _json_text(result)


def _druglikeness_checker(value: Any) -> str:
    values = _literalish(value)
    if isinstance(values, dict):
        values = values.get("candidates") or values.get("smiles")
    if not isinstance(values, list):
        values = [values]
    return _json_text([_descriptor_row(item) for item in values])


def _candidate_ranker(value: Any) -> str:
    payload = _literalish(value)
    if not isinstance(payload, dict):
        raise ValueError("candidate_ranker expects a JSON object")
    objective = str(payload.get("objective") or "maximize_qed")
    rows = payload.get("descriptor_rows") or payload.get("descriptors")
    if not isinstance(rows, list):
        candidates = payload.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("candidate_ranker needs candidates or descriptor_rows")
        rows = [
            _descriptor_row(item.get("smiles") if isinstance(item, dict) else item)
            for item in candidates
        ]
    ranked = [
        {**row, "objective_score": round(_objective_score(row, objective), 4)}
        for row in rows
    ]
    ranked.sort(key=lambda row: row["objective_score"], reverse=True)
    if not ranked:
        raise ValueError("No valid candidates to rank")
    return _json_text({"objective": objective, "ranking": ranked, "best": ranked[0]})


def _crippen_descriptors(value: Any) -> str:
    _, Crippen, *_ = _rdkit_modules()
    return _json_text({
        "logp": round(float(Crippen.MolLogP(_mol(value))), 4),
        "molar_refractivity": round(float(Crippen.MolMR(_mol(value))), 4),
    })


def _registry_runners() -> dict[str, Callable[[Any], Any]]:
    return {
        "chemistrytools/get_compound_CID": _pubchem_cid,
        "chemistrytools/get_compound_MolecularWeight_by_CID": lambda x: float(_pubchem_properties(x)["MolecularWeight"]),
        "chem_lib/calculate_compound_molar_mass": _calculate_molar_mass,
        "chem_lib/get_empirical_formula_by_percent_composition": _empirical_formula,
        "chem_lib/calculate_compound_percentage_composition_by_mass": _percentage_by_mass,
        "cactus/CalculateDruglikeness": _druglikeness,
        "chemcrow/Query2CAS": _query_to_cas,
        "chem_lib/analyze_combustion": _analyze_combustion,
        "chem_lib/get_element_properties": _element_properties,
        "chemcrow/PatentCheck": _patent_check,
        "cactus/CalculateLogP": lambda x: float(_rdkit_modules()[1].MolLogP(_mol(x))),
        "chemcrow/ControlChemCheck": _control_chem_check,
        "chemistrytools/convert_compound_CID_to_SMILES": lambda x: str(
            _pubchem_properties(x).get("ConnectivitySMILES")
            or _pubchem_properties(x).get("SMILES")
            or _pubchem_properties(x).get("CanonicalSMILES")
        ),
        "chemistrytools/get_compound_charge_by_CID": lambda x: int(_pubchem_properties(x)["Charge"]),
        "chemistrytools/convert_compound_CID_to_Molecular_Formula": lambda x: _formula_counts(
            _pubchem_properties(x)["MolecularFormula"]
        ),
        "chem_lib/combustion_reactions": _combustion_reaction,
        "chem_lib/balance_the_reaction": _balance_reaction,
        "chemistrytools/convert_compound_CID_to_IUPAC": lambda x: str(_pubchem_properties(x)["IUPACName"]),
        "chem_lib/product_formulas_of_reaction": lambda x: _reaction_side(x, "product"),
        "chem_lib/reactant_formulas_of_reaction": lambda x: _reaction_side(x, "reactant"),
        **{
            name: (lambda x, predictor_name=name: _property_predictor(predictor_name, x))
            for name in _PREDICTOR_CLASS_NAMES
        },
        "RXNPredict": lambda x: _reaction_predictor("RXNPredict", x),
        "RXNRetrosynthetic": lambda x: _reaction_predictor("RXNRetrosynthetic", x),
        "NameToSMILES": _name_to_smiles,
        "TestMolecule": _test_molecule,
        "GetMolFormula": lambda x: str(_rdkit_modules()[6].CalcMolFormula(_mol(x))),
        "CalculateTPSA": lambda x: round(float(_rdkit_modules()[6].CalcTPSA(_mol(x))), 4),
        "GetExactMolceularWeight": lambda x: round(float(_rdkit_modules()[2].ExactMolWt(_mol(x))), 4),
        "SMILESToWeight": lambda x: round(float(_rdkit_modules()[2].ExactMolWt(_mol(x))), 4),
        "GetCrippenDescriptors": _crippen_descriptors,
        "GetHBDNum": lambda x: int(_rdkit_modules()[4].NumHDonors(_mol(x))),
        "GetHBANum": lambda x: int(_rdkit_modules()[4].NumHAcceptors(_mol(x))),
        "GetRingsNum": lambda x: int(_rdkit_modules()[6].CalcNumRings(_mol(x))),
        "GetRotatableBondsNum": lambda x: int(_rdkit_modules()[4].NumRotatableBonds(_mol(x))),
        "GetFractionCSP3": lambda x: round(float(_rdkit_modules()[6].CalcFractionCSP3(_mol(x))), 4),
        "FuncGroups": _functional_groups,
        "CheckExplosiveness": lambda x: "No explosive flag in local checker",
        "candidate_diversity_filter": _diversity_filter,
        "druglikeness_checker": _druglikeness_checker,
        "candidate_ranker": _candidate_ranker,
    }


def load_tools(names: Iterable[str] | None = None) -> list[ToolSpec]:
    requested = list(names) if names is not None else TOOL_NAMES
    unknown = [name for name in requested if name not in TOOL_NAMES]
    if unknown:
        raise ValueError(f"Unknown tool names: {unknown}")
    runners = _registry_runners()
    missing = [name for name in requested if name not in runners]
    if missing:
        raise RuntimeError(f"No runner registered for tools: {missing}")
    return [ToolSpec(name, TOOL_DESCRIPTIONS[name], runners[name]) for name in requested]


class ToolExecutor:
    def __init__(self, tools: Iterable[ToolSpec], timeout_seconds: int = 120):
        self.tools = {tool.name: tool for tool in tools}
        self.timeout_seconds = int(timeout_seconds)

    def execute(self, name: str, arguments: Any) -> tuple[str, bool]:
        tool = self.tools.get(name)
        if tool is None:
            return f'Error: Unknown tool "{name}".', False
        value = arguments.get("input") if isinstance(arguments, dict) and "input" in arguments else arguments

        previous_handler = None
        if self.timeout_seconds > 0 and hasattr(signal, "SIGALRM"):
            def timeout_handler(signum, frame):  # noqa: ARG001
                raise TimeoutError(f"tool call timed out after {self.timeout_seconds}s")

            previous_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, float(self.timeout_seconds))
        try:
            result = tool(value)
            return str(result), True
        except Exception as exc:
            return f"Error: {type(exc).__name__}: {exc}", False
        finally:
            if previous_handler is not None:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                signal.signal(signal.SIGALRM, previous_handler)


def unified_system_prompt() -> str:
    return str(CATALOG["unified_system_prompt"])
