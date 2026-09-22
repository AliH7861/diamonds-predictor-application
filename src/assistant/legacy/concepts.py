"""Map user language to the project's existing diamond concepts."""

from dataclasses import asdict, dataclass
import re


@dataclass(frozen=True)
class ConceptEvidence:
    concept: str
    confidence: float
    source: str
    evidence: str


CONCEPT_PATTERNS = {
    "PRICE": r"\b(?:price|cost|expensive|cheap|money)\b|\$",
    "VALUE": r"\b(?:value|worth|overpriced|underpriced|deal)\b|bang for (?:the )?buck|for my money",
    "BUDGET": r"\b(?:budget|under|below|up to|less than|spend|ceiling|cap)\b|\$",
    "CARAT_SIZE": r"\bcarat|carats|ct|size|large|small|medium|bigger\b",
    "DIMENSIONS": r"\bdepth|table|dimensions?|\bx\b|\by\b|\bz\b",
    "CUT": r"\bcut|ideal|premium|very good|fair\b",
    "COLOR": r"\bcolou?r(?: grade)?\b",
    "CLARITY": r"\bclarity|clean(?:liness| looking)?|\b(?:I1|SI1|SI2|VS1|VS2|VVS1|VVS2|IF)\b",
    "QUALITY": r"\b(?:quality|clean|cleanliness|eye-clean)\b",
    "SEARCH_FILTER": r"\b(?:find|show|search|looking for|options?|matches)\b|\b(?:want|need)\b.{0,60}(?:under|below|\$|carat|clarity|cut|diamond)",
    "FILTER": r"\b(?:under|below|at least|at most|no more than|or better|between)\b",
    "RANK_RECOMMENDATION": r"\b(?:recommend|shortlist|rank|top options?|sort (?:the )?matches)\b|\bbest\s+(?:value|deal|option|diamond|match)|bang for (?:the )?buck",
    "COMPARE": r"\bcompare|side by side|difference between|versus|\bvs\.\b",
    "PRICE_PREDICTION": r"\bpredict price|price estimate|what would this cost\b",
    "CLARITY_PREDICTION": r"\bpredict clarity|classify clarity|clarity family\b",
    "RELATIONSHIP": r"\b(?:relationship|correlation|correlate|relate|affect|changes? with|move together|against)\b",
    "TREND": r"\b(?:trend|group(?:ed)? by|break .* down by|across .{0,20} groups?|higher/lower)\b|\bby (?:cut|color|clarity|carat band)\b",
    "GROUP": r"\b(?:group|grouped|across)\b|\bby (?:cut|color|clarity|carat band)\b",
    "STATISTIC": r"\b(?:average|avg|mean|median|count|how many|minimum|min|maximum|max|q25|q75|percentile|standard deviation|std|range)\b",
    "EXPLANATION": r"\bexplain|what is|what does|why|meaning\b",
    "CUSTOMER_SEGMENT": r"\bcustomer profile|buyer profile|segment|archetype\b",
    "MODEL_PREDICTION": r"\b(?:predict|prediction|classify|model estimate|price estimate)\b",
    "CORRECTION": r"\b(?:actually|instead|change|make it|switch|correction|same budget|keep)\b",
    "REMOVAL": r"\b(?:remove|forget|clear|drop|do not care about|don't care about)\b",
    "ENGINEERED_FEATURE": r"\b(?:volume proxy|face area proxy|xy asymmetry|length width ratio|depth ratio xyz|carat squared|price per carat|weight per volume|normalized dimensions?)\b",
}


def map_concepts(text: str) -> list[ConceptEvidence]:
    """Return every explicitly supported concept with auditable evidence."""
    found = []
    for concept, pattern in CONCEPT_PATTERNS.items():
        match = re.search(pattern, text, re.I)
        if match:
            found.append(ConceptEvidence(concept, 0.99, "deterministic_pattern", match.group(0)))
    return found


def concepts_as_dicts(text: str) -> list[dict]:
    return [asdict(item) for item in map_concepts(text)]
