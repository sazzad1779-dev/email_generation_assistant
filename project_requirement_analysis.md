
# EMAIL GENERATION ASSISTANT — COMPLETE PROJECT ANALYSIS SHEET
## AI Engineer Candidate Assessment | LangChain + LangGraph + FastAPI + Python

---

## 1. EXECUTIVE SUMMARY

**Project:** Email Generation Assistant with Custom Evaluation Metrics & Model Comparison  
**Tech Stack:** Python, FastAPI, LangChain, LangGraph, Pydantic, AsyncIO  
**Primary LLM (Model A):** Google Gemini 2.5 Flash (Free Tier: 1,500 req/day, 10 RPM, 1M context)  
**Comparison LLM (Model B):** Groq Llama 3.3 70B (Free Tier: 30 RPM, 1,000 req/day, 128K context)  
**Judge LLM (for LLM-as-a-Judge):** Gemini 2.5 Flash (separate instance to avoid self-evaluation bias)  
**Deployment:** FastAPI service with async endpoints, batch evaluation pipeline  
**Output Format:** Markdown emails (best practice for email generation — clean, portable, renderable)

---

## 2. SYSTEM ARCHITECTURE

### 2.1 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                       │
│  │  Web UI      │  │  CLI Tool    │  │  Test Runner │                       │
│  │  (Optional)  │  │  (Dev/Debug) │  │  (Evaluation)│                       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                       │
└─────────┼─────────────────┼─────────────────┼───────────────────────────────┘
          │                 │                 │
          └─────────────────┴─────────────────┘
                            │
┌───────────────────────────▼────────────────────────────────────────────────┐
│                         FASTAPI LAYER                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  /generate          POST → Generate email (Model A or B)            │   │
│  │  /evaluate          POST → Run custom metrics on single email       │   │
│  │  /batch-evaluate    POST → Run full 10-scenario evaluation          │   │
│  │  /compare           POST → Run A/B model comparison                 │   │
│  │  /health            GET  → Service health + rate limit status       │   │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                              │                                             │
│  ┌───────────────────────────▼──────────────────────────────────────────┐   │
│  │  MIDDLEWARE: Rate Limiting, Request Validation, Error Handling       │   │
│  │  - Pydantic v2 models for strict input validation                  │   │
│  │  - Custom exception handlers for LLM failures                      │   │
│  │  - Request ID tracing for debugging                                │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────────────┐
│                      LANGGRAPH ORCHESTRATION LAYER                           │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STATE GRAPH: EmailGenerationState                                  │    │
│  │                                                                     │    │
│  │  [START] ──► [validate_input] ──► [construct_prompt] ──► [call_llm]│    │
│  │                                              │                      │    │
│  │  ┌───────────────────────────────────────────┘                      │    │
│  │  │                                                                  │    │
│  │  ▼                                                                  │    │
│  │  [post_process] ──► [quality_check] ──► [END]                      │    │
│  │                                                                     │    │
│  │  Fallback path: [call_llm] ──► [retry_with_backup] ──► [call_llm]  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  KEY DESIGN DECISION: LangGraph provides:                                    │
│  • State persistence across steps                                            │
│  • Retry/fallback logic (critical for free tier rate limits)                 │
│  • Conditional edges (e.g., if output too short → retry)                     │
│  • Parallel execution for evaluation metrics                                 │
└───────────────────────────┬─────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────────────┐
│                      PROMPT ENGINEERING LAYER                                │
│                                                                              │
│  TECHNIQUE: Chain-of-Thought + Few-Shot Examples + Role-Playing (Combined)  │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  PROMPT TEMPLATE STRUCTURE:                                         │    │
│  │                                                                     │    │
│  │  SYSTEM PROMPT (Role-Playing):                                      │    │
│  │  "You are an expert business communication specialist with 15 years │    │
│  │   of experience writing professional emails across Fortune 500     │    │
│  │   companies. You understand nuanced tone adaptation and fact       │    │
│  │   integration without sounding robotic."                           │    │
│  │                                                                     │    │
│  │  FEW-SHOT EXAMPLES (3 examples):                                    │    │
│  │  Example 1: Intent="Follow up after meeting"                       │    │
│  │             Facts=["Met on June 10", "Discussed Q3 roadmap"]       │    │
│  │             Tone="Formal"                                          │    │
│  │             Output: [Full email example]                           │    │
│  │                                                                     │    │
│  │  CHAIN-OF-THOUGHT INSTRUCTION:                                      │    │
│  │  "Before writing the email, think through:                         │    │
│  │   1. What is the relationship between sender and recipient?        │    │
│  │   2. Which facts are most important and where should they appear?  │    │
│  │   3. What tone markers (word choice, sentence length) achieve      │
│  │      the requested tone?                                           │    │
│  │   4. How should the email open and close for maximum impact?       │    │
│  │   Write your thinking in <thinking> tags, then the email."        │    │
│  │                                                                     │    │
│  │  DYNAMIC USER INPUT:                                                │    │
│  │  Intent: {intent}                                                   │    │
│  │  Key Facts: {facts}                                                 │    │
│  │  Tone: {tone}                                                       │    │
│  │  Constraints: Must include all facts, match tone exactly,          │    │
│  │               150-400 words, professional signature                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└───────────────────────────┬─────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────────────┐
│                         LLM PROVIDER LAYER                                   │
│                                                                              │
│  ┌─────────────────────────┐    ┌─────────────────────────┐                 │
│  │  MODEL A: Gemini Flash  │    │  MODEL B: Groq Llama    │                 │
│  │  (Primary Generator)    │    │  (Comparison Generator) │                 │
│  │  - 1,500 req/day        │    │  - 1,000 req/day        │                 │
│  │  - 10 RPM               │    │  - 30 RPM               │                 │
│  │  - 1M context           │    │  - 128K context         │                 │
│  │  - Best for: quality    │    │  - Best for: speed      │                 │
│  │  - Native SDK (Google)  │    │  - OpenAI-compatible    │                 │
│  └─────────────────────────┘    └─────────────────────────┘                 │
│                                                                              │
│  FALLBACK CHAIN (if primary fails):                                          │
│  Gemini Flash → Groq Llama → OpenRouter (free tier) → Error                 │
│                                                                              │
│  RATE LIMIT MANAGEMENT:                                                      │
│  • Token bucket algorithm per provider                                       │
│  • Exponential backoff with jitter                                           │
│  • Request queuing for batch evaluation                                      │
│  • Pre-flight quota checks before batch jobs                                 │
└───────────────────────────┬─────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────────────┐
│                      EVALUATION ENGINE LAYER                                 │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  CUSTOM METRIC 1: FACT RECALL SCORE (FRS)                           │    │
│  │  ├─ Type: Automated (Python + LLM-as-Judge hybrid)                  │    │
│  │  ├─ Logic: Extract facts from generated email, compare with input   │    │
│  │  ├─ Scoring: 0-100 (weighted by fact importance)                    │    │
│  │  └─ Implementation: NER + semantic similarity + LLM verification    │    │
│  │                                                                     │    │
│  │  CUSTOM METRIC 2: TONE FIDELITY INDEX (TFI)                         │    │
│  │  ├─ Type: Automated (Python NLP + LLM-as-Judge)                     │    │
│  │  ├─ Logic: Analyze word choice, sentence complexity, formality      │    │
│  │  ├─ Scoring: 0-100 (distance from target tone profile)              │    │
│  │  └─ Implementation: LIWC-style features + embedding similarity      │    │
│  │                                                                     │    │
│  │  CUSTOM METRIC 3: STRUCTURAL COHERENCE SCORE (SCS)                  │    │
│  │  ├─ Type: Automated (Python + LLM-as-Judge)                         │    │
│  │  ├─ Logic: Grammar, fluency, conciseness, intro/closing quality     │    │
│  │  ├─ Scoring: 0-100 (composite of sub-metrics)                       │    │
│  │  └─ Implementation: LanguageTool API + readability + LLM judge      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  OUTPUT: JSON/CSV report with per-scenario and aggregate scores              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow Diagram

```
User Request
    │
    ▼
┌─────────────────┐
│ Input Validation│ ← Pydantic models enforce types, ranges, forbidden words
│ (Pydantic v2)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ LangGraph State │────►│ Prompt Assembly │ ← Few-shot examples loaded from
│ Initialization  │     │ (Jinja2/Dynamic)│   JSON registry based on intent
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ LLM Call        │────►│ Rate Limit Check│ ← Token bucket, quota pre-check
│ (Async)         │     │ (Failover ready)│
└────────┬────────┘     └─────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
Success   Failure
    │         │
    ▼         ▼
┌─────────┐ ┌─────────────┐
│ Parsing │ │ Fallback to │
│ (Regex/ │ │ Backup LLM  │
│  XML)   │ │ (Max 2 retries)│
└────┬────┘ └──────┬──────┘
     │              │
     └──────┬───────┘
            ▼
    ┌───────────────┐
    │ Post-Process  │ ← Strip thinking tags, validate length,
    │ (Clean output)│   ensure signature, check word count
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │ Quality Gate  │ ← Minimum thresholds (FRS≥70, TFI≥60, SCS≥65)
    │ (Auto-retry   │   If fail → retry with stronger prompt
    │  if below)    │
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │ Return Email  │ ← Markdown format with metadata
    │ + Metadata    │   (model used, tokens, latency, scores if eval)
    └───────────────┘
```

---

## 3. DETAILED COMPONENT ANALYSIS

### 3.1 Input Schema & Validation

```python
# Pydantic v2 Models — STRICT VALIDATION

class ToneEnum(str, Enum):
    FORMAL = "formal"
    CASUAL = "casual"
    URGENT = "urgent"
    EMPATHETIC = "empathetic"
    PERSUASIVE = "persuasive"
    APOLOGETIC = "apologetic"

class EmailRequest(BaseModel):
    intent: str = Field(..., min_length=10, max_length=500, 
                        description="Core purpose of the email")
    key_facts: List[str] = Field(..., min_length=1, max_length=10,
                                 description="Bullet points to include")
    tone: ToneEnum = Field(..., description="Desired email tone")
    recipient_name: Optional[str] = Field(None, max_length=100)
    sender_name: Optional[str] = Field(None, max_length=100)
    max_words: int = Field(300, ge=50, le=800)
    
    @field_validator('key_facts')
    @classmethod
    def validate_facts_not_empty(cls, v):
        for fact in v:
            if len(fact.strip()) < 5:
                raise ValueError("Each fact must be at least 5 characters")
        return v
    
    @field_validator('intent')
    @classmethod
    def validate_intent_specific(cls, v):
        forbidden = ["email", "write", "compose"]  # Too vague
        if any(word in v.lower() for word in forbidden) and len(v) < 30:
            raise ValueError("Intent must be specific, not generic")
        return v
```

**Corner Cases Handled:**
- Empty facts list → ValidationError with clear message
- Facts with only whitespace → Stripped and validated
- Intent too vague (e.g., "write an email") → Rejected with suggestion
- Tone not in enum → 422 Unprocessable Entity
- max_words out of range → Clamped or rejected based on config
- Special characters in names → Sanitized (XSS prevention)
- Very long facts (>500 chars) → Warning logged, truncated with ellipsis

### 3.2 LangGraph State Machine

```python
# State Definition
class EmailGenerationState(TypedDict):
    request: EmailRequest
    prompt: str
    raw_output: str
    cleaned_email: str
    metadata: Dict[str, Any]
    retry_count: int
    error: Optional[str]
    scores: Optional[Dict[str, float]]

# Graph Nodes

async def validate_input(state: EmailGenerationState) -> EmailGenerationState:
    """Validate and enrich input data."""
    # Enrich: detect intent category for few-shot selection
    # Enrich: estimate complexity score
    return state

async def construct_prompt(state: EmailGenerationState) -> EmailGenerationState:
    """Build dynamic prompt with few-shot examples."""
    # Load examples from registry based on intent similarity
    # Inject CoT instructions
    # Add constraints (word count, tone markers)
    return state

async def call_llm(state: EmailGenerationState) -> EmailGenerationState:
    """Async LLM call with rate limit handling."""
    # Try primary provider
    # If rate limited → fallback provider
    # If fail after 2 fallbacks → raise with context
    return state

async def post_process(state: EmailGenerationState) -> EmailGenerationState:
    """Clean and format output."""
    # Remove <thinking> tags
    # Ensure proper greeting and closing
    # Validate word count
    # Add signature if missing
    return state

async def quality_check(state: EmailGenerationState) -> EmailGenerationState:
    """Run lightweight quality gate."""
    # Quick heuristic checks (has greeting? has closing? word count ok?)
    # If fail AND retry_count < 2 → route back to construct_prompt with stronger instructions
    # If fail AND retry_count >= 2 → return with warning flag
    return state

# Conditional Edges
def route_after_quality(state: EmailGenerationState) -> str:
    if state.get("error"):
        return "error_handler"
    if state["retry_count"] < 2 and not state.get("quality_passed", True):
        return "construct_prompt"  # Retry with stronger prompt
    return "end"

# Compile Graph
builder = StateGraph(EmailGenerationState)
builder.add_node("validate", validate_input)
builder.add_node("construct_prompt", construct_prompt)
builder.add_node("call_llm", call_llm)
builder.add_node("post_process", post_process)
builder.add_node("quality_check", quality_check)

builder.set_entry_point("validate")
builder.add_edge("validate", "construct_prompt")
builder.add_edge("construct_prompt", "call_llm")
builder.add_edge("call_llm", "post_process")
builder.add_edge("post_process", "quality_check")
builder.add_conditional_edges("quality_check", route_after_quality)

graph = builder.compile()
```

### 3.3 Prompt Engineering Strategy (ADVANCED)

**Technique: "Structured Chain-of-Thought with Dynamic Few-Shot Retrieval"**

This combines three techniques:
1. **Role-Playing** — Establishes expertise context
2. **Few-Shot Examples** — Retrieved dynamically based on intent similarity
3. **Chain-of-Thought** — Forces reasoning before generation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PROMPT TEMPLATE v1.0                                 │
└─────────────────────────────────────────────────────────────────────────────┘

SYSTEM:
You are Dr. Elena Voss, a senior executive communication coach who has trained 
C-suite leaders at Fortune 500 companies for 18 years. You specialize in 
translating business intent into precisely-toned, fact-rich professional emails.

Your emails are known for:
- Seamless fact integration (never list-like, always narrative)
- Exact tone matching (formal = structured, no contractions; casual = warm, 
  conversational; urgent = direct, action-oriented; empathetic = supportive, 
  acknowledging feelings)
- Strategic structure (hook → context → facts → call-to-action → close)

FEW-SHOT EXAMPLES (Retrieved dynamically — top 3 by intent similarity):

[Example 1 — Formal / Follow-up]
Intent: Follow up after quarterly business review
Facts: ["Q3 revenue up 12%", "New hire Sarah starts Monday", "Budget approved"]
Tone: formal

<thinking>
1. Relationship: CEO to Board member — high formality, respect
2. Key facts placement: Revenue in opening (establishes credibility), 
   Sarah in body (operational detail), budget as closing CTA
3. Tone markers: No contractions, precise terminology, "I am pleased to report"
4. Structure: Direct subject line, formal greeting, structured paragraphs
</thinking>

Subject: Q3 Performance Update and Q4 Preparations

Dear Ms. Richardson,

I am pleased to report that our third-quarter revenue has increased by 12 percent 
compared to the same period last year, reflecting the strategic initiatives we 
discussed during our quarterly business review.

In preparation for the fourth quarter, I would like to bring several operational 
updates to your attention. Sarah Chen will be joining our team on Monday as the 
new Director of Digital Transformation. Her expertise will be instrumental as we 
implement the approved budget for technology infrastructure upgrades.

Please let me know if you require any additional documentation before our next 
scheduled meeting.

Respectfully,

[Name]
[Title]

---

[Example 2 — Casual / Request]
[Example 3 — Urgent / Escalation]

---

TASK:
Generate a professional email with the following specifications:

INTENT: {intent}
KEY FACTS (MUST ALL BE INCLUDED):
{formatted_facts}

TONE: {tone}
TONE PROFILE:
{tone_specific_guidance}  ← Loaded from tone registry

CONSTRAINTS:
- Word count: Between {min_words} and {max_words} words
- Must include all key facts naturally (not as a list)
- Must match the {tone} tone precisely
- Include appropriate greeting and professional closing
- Subject line should be clear and relevant

OUTPUT FORMAT:
<thinking>
[Your reasoning process here]
</thinking>

---

Subject: [Subject line]

[Email body]

[Closing]
[Name]
```

**Tone-Specific Guidance Registry:**

| Tone | Markers | Sentence Structure | Word Choice |
|------|---------|-------------------|-------------|
| Formal | No contractions, full titles, "would like", "please advise" | Complex, varied length | Precise, technical |
| Casual | Contractions OK, first names, "hey", "thanks" | Short to medium, conversational | Warm, accessible |
| Urgent | Direct imperatives, "immediately", "action required" | Short, punchy | Forceful, clear |
| Empathetic | "I understand", "I know this is difficult" | Gentle pacing, pauses | Supportive, validating |

---

## 4. CUSTOM EVALUATION METRICS — DEEP DIVE

### 4.1 Metric 1: FACT RECALL SCORE (FRS)

**Definition:** Measures the completeness and accuracy of fact integration in the generated email.

**Why Custom:** Generic metrics (BLEU, ROUGE) can't verify if specific business facts were included. FRS checks semantic equivalence, not just string matching.

**Scoring Logic (0-100):**

```
FRS = Σ(fact_score_i) / N * 100

Where for each fact i:
  fact_score_i = 1.0  if exact or near-exact match (semantic similarity > 0.85)
  fact_score_i = 0.7  if paraphrased but accurate (similarity 0.60-0.85)
  fact_score_i = 0.4  if partially mentioned but distorted
  fact_score_i = 0.0  if completely missing

Bonus: +5 points if all facts are integrated naturally (not listed)
Penalty: -10 points if any fact is hallucinated (not in input)
```

**Implementation Pipeline:**

```python
class FactRecallScorer:
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embedding_model)
        self.llm_judge = GeminiJudge()  # For edge cases
    
    async def score(self, generated_email: str, key_facts: List[str]) -> Dict:
        # Step 1: Extract fact-like statements from email (NER + pattern matching)
        extracted = self._extract_statements(generated_email)
        
        # Step 2: Semantic matching
        scores = []
        for fact in key_facts:
            fact_emb = self.embedder.encode(fact)
            best_match = max(
                cosine_similarity(fact_emb, self.embedder.encode(ex))
                for ex in extracted
            )
            scores.append(self._score_to_bucket(best_match))
        
        # Step 3: Hallucination check (LLM-as-Judge)
        hallucination_penalty = await self._check_hallucinations(
            generated_email, key_facts
        )
        
        # Step 4: Natural integration check
        integration_bonus = self._check_natural_integration(generated_email, key_facts)
        
        raw_score = sum(scores) / len(scores) * 100
        final_score = max(0, raw_score + integration_bonus - hallucination_penalty)
        
        return {
            "score": round(final_score, 2),
            "per_fact_breakdown": [...],
            "hallucinations_found": hallucination_penalty > 0,
            "natural_integration": integration_bonus > 0
        }
```

**Corner Cases:**
- Fact expressed as date range vs exact date → semantic match should catch
- Fact split across sentences → aggregate extraction needed
- Fact mentioned in subject line → should count
- Numerical fact with unit conversion ("12%" vs "twelve percent") → normalize before matching
- Fact implied but not stated → LLM judge decides (conservative: no credit)

### 4.2 Metric 2: TONE FIDELITY INDEX (TFI)

**Definition:** Measures how accurately the generated email matches the requested tone across linguistic dimensions.

**Why Custom:** Standard sentiment analysis can't distinguish "formal" from "casual". TFI uses a multi-dimensional tone profile.

**Scoring Logic (0-100):**

```
TFI = 0.3 * Lexical_Score + 0.25 * Syntactic_Score + 0.25 * Pragmatic_Score + 0.2 * Emotional_Score

Lexical Score (word choice analysis):
  - Contraction ratio (formal: <5%, casual: >20%)
  - Modal verb usage (formal: "would", "could"; urgent: "must", "need")
  - Politeness markers (formal: "please advise"; casual: "thanks", "cheers")
  - Jargon density (formal: high; casual: low)

Syntactic Score (sentence structure):
  - Average sentence length (formal: 20+ words; urgent: <15 words)
  - Subordination complexity (formal: high; casual: low)
  - Imperative ratio (urgent: high; empathetic: low)

Pragmatic Score (communicative function):
  - Directness level (urgent: very direct; empathetic: indirect)
  - Hedging frequency (formal: moderate; urgent: low)
  - Question ratio (empathetic: higher; formal: lower)

Emotional Score (affective markers):
  - Positive/negative word ratio (empathetic: balanced acknowledgment)
  - Intensity modifiers (urgent: strong; formal: neutral)
```

**Implementation:**

```python
class ToneFidelityScorer:
    def __init__(self):
        self.tone_profiles = self._load_tone_profiles()  # Statistical baselines
        self.liwc_analyzer = LIWCAnalyzer()  # Or custom regex-based
    
    async def score(self, email: str, target_tone: str) -> Dict:
        features = self._extract_linguistic_features(email)
        target_profile = self.tone_profiles[target_tone]
        
        # Calculate distances for each dimension
        lexical_dist = self._lexical_distance(features, target_profile)
        syntactic_dist = self._syntactic_distance(features, target_profile)
        pragmatic_dist = self._pragmatic_distance(features, target_profile)
        emotional_dist = self._emotional_distance(features, target_profile)
        
        # Convert distances to scores (closer = higher score)
        scores = {
            "lexical": max(0, 100 - lexical_dist * 100),
            "syntactic": max(0, 100 - syntactic_dist * 100),
            "pragmatic": max(0, 100 - pragmatic_dist * 100),
            "emotional": max(0, 100 - emotional_dist * 100)
        }
        
        final = 0.3 * scores["lexical"] + 0.25 * scores["syntactic"] + \
                0.25 * scores["pragmatic"] + 0.2 * scores["emotional"]
        
        return {
            "score": round(final, 2),
            "dimension_breakdown": scores,
            "detected_tone": self._classify_tone(features),
            "confidence": self._confidence_score(features)
        }
```

**Corner Cases:**
- Mixed tone (e.g., starts formal, ends casual) → Penalize inconsistency
- Tone drift in long emails → Segment analysis, worst segment dominates
- Cultural tone variations (UK vs US formal) → Normalize to standard business English
- Sarcasm or irony → Rare in business emails, but LLM judge catches it
- Tone requested is "empathetic" but facts are negative → Must balance empathy with professionalism

### 4.3 Metric 3: STRUCTURAL COHERENCE SCORE (SCS)

**Definition:** Measures the grammatical correctness, fluency, conciseness, and structural effectiveness of the email.

**Why Custom:** Grammar checkers alone don't measure email-specific effectiveness (hook quality, CTA clarity). SCS combines technical correctness with rhetorical structure.

**Scoring Logic (0-100):**

```
SCS = 0.25 * Grammar_Score + 0.25 * Fluency_Score + 0.20 * Conciseness_Score + 
      0.15 * Structure_Score + 0.15 * Professionalism_Score

Grammar Score (0-100):
  - Spelling errors: -5 per error
  - Subject-verb agreement: -10 per error
  - Pronoun reference clarity: -5 per ambiguity
  - Punctuation correctness: -3 per error
  (Base: 100, floor at 0)

Fluency Score (0-100):
  - Sentence transition smoothness (coherence metrics)
  - Repetition penalty (unique bigram ratio)
  - Readability (Flesch Reading Ease adapted for business context)
  - Flow score (sentence length variation coefficient)

Conciseness Score (0-100):
  - Word count vs target (optimal: within ±10% of requested)
  - Redundancy detection ("in order to" → "to", "at this point in time" → "now")
  - Information density (facts per sentence ratio)

Structure Score (0-100):
  - Has clear subject line: +10
  - Has appropriate greeting: +10
  - Opening hook quality (engages reader in first 2 sentences): +20
  - Body paragraph coherence (one idea per paragraph): +20
  - Clear call-to-action: +20
  - Professional closing: +10
  - Signature present: +10

Professionalism Score (0-100):
  - No inappropriate humor or slang
  - Proper capitalization
  - Consistent formatting
  - No ALL CAPS (except acronyms)
  - Appropriate use of bullet points (if any)
```

**Implementation:**

```python
class StructuralCoherenceScorer:
    def __init__(self):
        self.language_tool = language_tool_python.LanguageTool('en-US')
        self.readability = textstat
    
    async def score(self, email: str, request: EmailRequest) -> Dict:
        # Grammar check
        matches = self.language_tool.check(email)
        grammar_score = max(0, 100 - len(matches) * 5)
        
        # Fluency
        fluency = self._calculate_fluency(email)
        
        # Conciseness
        word_count = len(email.split())
        target = request.max_words
        conciseness = self._calculate_conciseness(email, word_count, target)
        
        # Structure
        structure = self._evaluate_structure(email)
        
        # Professionalism
        professionalism = self._check_professionalism(email)
        
        final = (0.25 * grammar_score + 0.25 * fluency + 0.20 * conciseness + 
                 0.15 * structure + 0.15 * professionalism)
        
        return {
            "score": round(final, 2),
            "grammar": round(grammar_score, 2),
            "fluency": round(fluency, 2),
            "conciseness": round(conciseness, 2),
            "structure": round(structure, 2),
            "professionalism": round(professionalism, 2),
            "word_count": word_count,
            "grammar_issues": [m.message for m in matches[:5]]
        }
```

**Corner Cases:**
- Email has perfect grammar but terrible structure → SCS catches via Structure score
- Very short email (under min_words) → Heavy conciseness penalty
- Very long email (over max_words) → Heavy conciseness penalty
- Missing subject line → Structure score hit
- Informal closing in formal email → Professionalism + Tone penalty
- Run-on sentences → Fluency + Grammar penalty

---

## 5. TEST DATA: 10 SCENARIOS WITH HUMAN REFERENCES

| # | Intent | Key Facts | Tone | Complexity | Human Reference Length |
|---|--------|-----------|------|------------|----------------------|
| 1 | Follow up after product demo meeting | ["Demo held on June 15", "Client liked AI feature", "Pricing discussion pending", "Next steps: technical deep-dive"] | Formal | Medium | 180 words |
| 2 | Request deadline extension for project deliverable | ["Original deadline: July 1", "Team member illness", "Need 2-week extension", "Quality will improve"] | Empathetic | Medium | 200 words |
| 3 | Inform team about office relocation | ["Moving to Building C", "Effective August 1", "New address: 450 Market St", "Parking info attached"] | Casual | Low | 150 words |
| 4 | Escalate unresolved security incident | ["Incident #4421", "48 hours no response", "Customer data potentially exposed", "Immediate action required"] | Urgent | High | 170 words |
| 5 | Congratulate colleague on promotion | ["Promoted to VP Engineering", "Effective immediately", "10 years at company", "Team celebration Friday"] | Casual | Low | 140 words |
| 6 | Request proposal details from vendor | ["RFP submitted May 20", "Need breakdown by module", "Budget cap: $150K", "Decision by June 30"] | Formal | Medium | 190 words |
| 7 | Apologize for delayed payment to contractor | ["Invoice #8832", "Payment 15 days late", "Bank transfer processing", "Will include late fee"] | Apologetic | Medium | 160 words |
| 8 | Announce new company policy on remote work | ["Effective Q3", "3 days office, 2 days remote", "Core collaboration days: Tue-Thu", "Hot-desking system"] | Formal | High | 220 words |
| 9 | Persuade stakeholder to approve new tool purchase | ["Current tool has 99.5% uptime", "New tool reduces cost 30%", "Migration takes 2 weeks", "ROI: 6 months"] | Persuasive | High | 210 words |
| 10 | Express condolences to team on loss of colleague | ["Sarah passed away June 10", "15 years of service", "Memorial service June 20", "Counseling available"] | Empathetic | High | 180 words |

**Human Reference Generation Strategy:**
- Write each reference as "gold standard" — what a skilled professional would write
- Ensure all facts are naturally integrated (not bulleted)
- Match tone precisely with linguistic markers
- Keep within word count constraints
- Save as Markdown files in `/data/references/`

---

## 6. MODEL COMPARISON FRAMEWORK

### 6.1 Comparison Setup

| Aspect | Model A (Gemini Flash) | Model B (Groq Llama 3.3 70B) |
|--------|----------------------|------------------------------|
| Provider | Google AI Studio | Groq |
| Model ID | gemini-2.5-flash | llama-3.3-70b-versatile |
| API | Native (google-genai) | OpenAI-compatible |
| Context | 1M tokens | 128K tokens |
| Speed | Medium | Very Fast (320 tok/s) |
| Free Quota | 1,500 req/day | 1,000 req/day |
| Rate Limit | 10 RPM | 30 RPM |
| Strength | Reasoning, long context | Speed, instruction following |
| Weakness | Slower, Google-specific SDK | Shorter context, occasional verbosity |

### 6.2 Evaluation Execution Plan

```python
async def run_comparison():
    results = []
    
    for scenario in TEST_SCENARIOS:
        # Run Model A
        email_a = await generate_email(scenario, model="gemini")
        scores_a = await evaluate_email(email_a, scenario)
        
        # Run Model B
        email_b = await generate_email(scenario, model="groq")
        scores_b = await evaluate_email(email_b, scenario)
        
        results.append({
            "scenario_id": scenario.id,
            "model_a": {"email": email_a, **scores_a},
            "model_b": {"email": email_b, **scores_b},
            "winner": determine_winner(scores_a, scores_b)
        })
    
    # Aggregate analysis
    report = generate_comparison_report(results)
    return report
```

### 6.3 Analysis Questions to Answer

1. **Which performed better?** → Compare aggregate FRS, TFI, SCS across 10 scenarios
2. **Biggest failure mode?** → Identify pattern in lowest scores (e.g., "Model B consistently hallucinated dates" or "Model A failed urgent tone")
3. **Production recommendation?** → Justify with data: "Model A wins on FRS (+12%) and TFI (+8%) despite 2x latency. Model B's speed advantage doesn't offset 23% fact hallucination rate."

---

## 7. CORNER CASES & FAILURE MODES

### 7.1 Input-Level Corner Cases

| Case | Handling Strategy |
|------|------------------|
| Intent is ambiguous ("email about the thing") | Reject with suggestion: "Please specify: follow-up, request, announcement, etc." |
| Facts contradict each other | Flag warning, proceed with LLM (may resolve or highlight contradiction) |
| Too many facts (>10) | Reject: "Maximum 10 facts. Please prioritize." |
| Facts contain PII (emails, phone numbers) | Sanitize in logs, warn user |
| Tone is "sarcastic" or "angry" | Reject: "Professional tones only: formal, casual, urgent, empathetic, persuasive, apologetic" |
| Non-English input | Detect language, reject or translate based on config |
| Empty intent with only facts | Reject: "Intent is required to guide email structure" |

### 7.2 Generation-Level Corner Cases

| Case | Handling Strategy |
|------|------------------|
| LLM output is cut off (max_tokens hit) | Retry with higher token limit, flag as partial |
| LLM refuses (safety filter) | Catch refusal pattern, retry with slightly modified prompt |
| LLM outputs thinking tags in final email | Regex strip `<thinking>...</thinking>` |
| LLM includes placeholder text like [Name] | Detect pattern, flag for user to fill in |
| LLM generates email in wrong language | Language detection, retry with explicit "Write in English" |
| LLM adds facts not in input (hallucination) | FRS hallucination penalty, flag in output |
| LLM misses critical facts | Quality gate catches, retry with emphasis |
| LLM tone drifts mid-email | Segment-level TFI analysis, flag inconsistency |
| LLM is too verbose (2x word limit) | Truncate with warning, or retry with stronger constraint |
| LLM is too brief (<50 words) | Retry with "expand on context" instruction |

### 7.3 API-Level Corner Cases

| Case | Handling Strategy |
|------|------------------|
| Rate limit hit (429) | Exponential backoff, switch to fallback LLM |
| API timeout | Retry once, then fallback |
| Invalid API key | Immediate error, don't retry |
| Model deprecated | Alert in health check, auto-switch to new model ID |
| Free quota exhausted | Graceful degradation: queue for tomorrow or suggest paid tier |
| Network partition | Circuit breaker pattern, return cached response if available |
| Provider outage | Automatic failover to backup provider |

### 7.4 Evaluation-Level Corner Cases

| Case | Handling Strategy |
|------|------------------|
| Human reference has different fact order | FRS uses semantic matching, order-independent |
| Human reference is shorter than generated | Conciseness score uses target, not reference length |
| Generated email is better than human reference | LLM-as-Judge may score higher than 100 → cap at 100 |
| Metric calculation fails | Return partial scores with error flag |
| LLM judge is inconsistent | Run 3 times, take median; document variance |
| Circular dependency (judge = generator) | Always use separate model instance for judging |

---

## 8. FASTAPI ENDPOINT SPECIFICATION

### 8.1 Endpoint Definitions

```python
# POST /generate
# Generate a single email

Request Body (EmailRequest):
{
    "intent": "Follow up after product demo meeting",
    "key_facts": [
        "Demo held on June 15",
        "Client liked AI feature",
        "Pricing discussion pending",
        "Next steps: technical deep-dive"
    ],
    "tone": "formal",
    "recipient_name": "Alice Johnson",
    "sender_name": "Bob Smith",
    "max_words": 250
}

Response (200 OK):
{
    "email": "Subject: Follow-Up: Product Demo Discussion\\n\\nDear Ms. Johnson,...",
    "metadata": {
        "model_used": "gemini-2.5-flash",
        "prompt_tokens": 450,
        "completion_tokens": 180,
        "latency_ms": 1200,
        "retry_count": 0,
        "word_count": 178
    },
    "quality_flags": []
}

# POST /evaluate
# Evaluate a single email against metrics

Request Body:
{
    "email": "Subject: ...\\n\\n...",
    "reference_email": "Subject: ...\\n\\n...",  # Optional
    "request": { /* EmailRequest object */ }
}

Response:
{
    "fact_recall_score": 92.5,
    "tone_fidelity_index": 88.0,
    "structural_coherence_score": 95.0,
    "overall_score": 91.8,
    "breakdown": { /* detailed per-metric breakdown */ }
}

# POST /batch-evaluate
# Run full evaluation on all 10 scenarios

Response:
{
    "scenarios_evaluated": 10,
    "results": [ /* per-scenario results */ ],
    "aggregate": {
        "avg_frs": 89.2,
        "avg_tfi": 85.5,
        "avg_scs": 91.0,
        "overall_avg": 88.6
    },
    "report_path": "/reports/evaluation_2026-06-18.json"
}

# POST /compare
# A/B model comparison

Request Body:
{
    "model_a": "gemini",
    "model_b": "groq",
    "scenarios": "all"  # or list of IDs
}

Response:
{
    "comparison_id": "cmp_20260618_001",
    "model_a_results": { /* aggregate */ },
    "model_b_results": { /* aggregate */ },
    "winner": "model_a",
    "margin": 8.5,
    "failure_analysis": "Model B showed 23% hallucination rate on dates...",
    "recommendation": "Model A recommended for production due to superior FRS..."
}

# GET /health
# Service health + provider status

Response:
{
    "status": "healthy",
    "providers": {
        "gemini": {"status": "up", "quota_remaining": 1450, "rpm_remaining": 8},
        "groq": {"status": "up", "quota_remaining": 980, "rpm_remaining": 25}
    },
    "version": "1.0.0"
}
```

### 8.2 Error Response Schema

```python
class ErrorResponse(BaseModel):
    error_code: str  # "RATE_LIMITED", "VALIDATION_ERROR", "LLM_FAILURE", etc.
    message: str
    details: Optional[Dict] = None
    request_id: str  # For tracing
    retry_after: Optional[int] = None  # Seconds, for 429
```

---

## 9. PROJECT STRUCTURE

```
email-generation-assistant/
├── README.md                          # Setup, run, architecture overview
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Modern Python packaging
├── .env.example                       # Template for API keys
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── main.py                        # FastAPI app entry point
│   ├── config.py                      # Pydantic Settings, env vars
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py                  # FastAPI routers
│   │   ├── models.py                  # Pydantic request/response models
│   │   ├── dependencies.py            # Auth, rate limiting, logging
│   │   └── middleware.py              # Error handling, CORS, tracing
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── email_generator.py         # Main orchestration logic
│   │   ├── state.py                   # LangGraph state definitions
│   │   ├── graph.py                   # LangGraph compilation
│   │   └── exceptions.py              # Custom exceptions
│   │
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── templates/
│   │   │   ├── base.j2                # Jinja2 base template
│   │   │   ├── system_role.j2         # Role-playing system prompt
│   │   │   ├── few_shot.j2            # Few-shot example injection
│   │   │   └── cot_instructions.j2    # Chain-of-thought guidance
│   │   ├── examples/
│   │   │   ├── formal_followup.json   # Example scenarios
│   │   │   ├── casual_request.json
│   │   │   └── ... (6 more)
│   │   ├── tone_profiles/
│   │   │   ├── formal.yaml            # Tone-specific guidance
│   │   │   ├── casual.yaml
│   │   │   └── ... (4 more)
│   │   └── registry.py                # Dynamic example retrieval
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── providers/
│   │   │   ├── base.py                # Abstract provider interface
│   │   │   ├── gemini_provider.py     # Google Gemini integration
│   │   │   ├── groq_provider.py       # Groq/OpenAI-compatible
│   │   │   └── router.py              # Failover logic
│   │   ├── rate_limiter.py            # Token bucket, quota tracking
│   │   └── judge.py                   # LLM-as-a-Judge wrapper
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics/
│   │   │   ├── base.py                # Abstract metric interface
│   │   │   ├── fact_recall.py         # FRS implementation
│   │   │   ├── tone_fidelity.py       # TFI implementation
│   │   │   └── structural_coherence.py # SCS implementation
│   │   ├── runner.py                  # Batch evaluation orchestration
│   │   ├── reporter.py                # Report generation (JSON/CSV/PDF)
│   │   └── comparison.py              # A/B comparison logic
│   │
│   └── utils/
│       ├── __init__.py
│       ├── text_processing.py         # Cleaning, normalization
│       ├── validators.py              # Custom validators
│       └── logging_config.py          # Structured logging
│
├── data/
│   ├── test_scenarios.json            # 10 test scenarios
│   ├── human_references/              # Gold-standard emails
│   │   ├── scenario_01.md
│   │   └── ... (9 more)
│   └── few_shot_examples/             # Prompt examples
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                    # Pytest fixtures
│   ├── test_api.py                    # API endpoint tests
│   ├── test_metrics.py                # Metric unit tests
│   ├── test_graph.py                  # LangGraph flow tests
│   └── test_integration.py            # End-to-end tests
│
├── reports/                           # Generated evaluation reports
│   └── .gitkeep
│
├── scripts/
│   ├── run_server.py                  # `python scripts/run_server.py`
│   ├── run_evaluation.py              # `python scripts/run_evaluation.py`
│   └── run_comparison.py              # `python scripts/run_comparison.py`
│
└── docs/
    ├── architecture.md                # This analysis sheet
    ├── prompt_template.md             # Documented prompt template
    ├── metrics_spec.md                # Metric definitions and logic
    └── comparison_report.md           # Final analysis summary
```

---

## 10. DEPENDENCIES

```txt
# Core Framework
fastapi>=0.111.0
uvicorn[standard]>=0.30.0
pydantic>=2.7.0
pydantic-settings>=2.3.0

# LangChain Ecosystem
langchain>=0.2.0
langchain-google-genai>=1.0.0
langchain-groq>=0.1.0
langgraph>=0.1.0

# LLM Providers
google-generativeai>=0.7.0
groq>=0.9.0
openai>=1.30.0  # For Groq compatibility

# NLP & Evaluation
sentence-transformers>=3.0.0
spacy>=3.7.0
language-tool-python>=2.8.0
textstat>=0.7.0
nltk>=3.8.0

# Data Processing
jinja2>=3.1.0
pyyaml>=6.0.0
pandas>=2.2.0

# Async & HTTP
httpx>=0.27.0
aiohttp>=3.9.0
aiolimiter>=1.1.0

# Testing & Quality
pytest>=8.2.0
pytest-asyncio>=0.23.0
pytest-cov>=5.0.0
httpx>=0.27.0  # For TestClient

# Utilities
python-dotenv>=1.0.0
structlog>=24.1.0
orjson>=3.10.0  # Fast JSON
```

---

## 11. SETUP & EXECUTION GUIDE

### 11.1 Environment Setup

```bash
# 1. Clone and setup
git clone <repo-url>
cd email-generation-assistant
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
pip install -r requirements.txt

# 2. Download NLP models
python -m spacy download en_core_web_md
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# 3. Configure environment
cp .env.example .env
# Edit .env:
#   GEMINI_API_KEY=your_google_key
#   GROQ_API_KEY=your_groq_key
```

### 11.2 Running the Service

```bash
# Development
python scripts/run_server.py --reload --port 8000

# Production
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4

# API docs at: http://localhost:8000/docs
```

### 11.3 Running Evaluation

```bash
# Full evaluation (Model A)
python scripts/run_evaluation.py --model gemini --output reports/eval_gemini.json

# Full evaluation (Model B)
python scripts/run_evaluation.py --model groq --output reports/eval_groq.json

# Comparison
python scripts/run_comparison.py --model-a gemini --model-b groq --output reports/comparison.json
```

### 11.4 Testing

```bash
# Unit tests
pytest tests/ -v --cov=src

# Integration tests (requires API keys)
pytest tests/test_integration.py -v --run-integration

# Specific metric tests
pytest tests/test_metrics.py::test_fact_recall -v
```

---

## 12. RISK MITIGATION & BEST PRACTICES

### 12.1 Free Tier Management

| Risk | Mitigation |
|------|-----------|
| Quota exhaustion mid-evaluation | Pre-check quota before batch jobs; implement request queuing |
| RPM throttling | Token bucket per provider; stagger requests with jitter |
| Provider changes free tier | Monitor health endpoint; alert on quota < 100 requests |
| API key rotation | Support multiple keys per provider; failover logic |

### 12.2 Quality Assurance

| Risk | Mitigation |
|------|-----------|
| Prompt injection | Input sanitization; no user text in system prompt |
| Hallucination | FRS metric catches it; human review for critical emails |
| Tone mismatch | TFI metric + quality gate retry |
| Inconsistent outputs | Temperature=0 for reproducibility; seed fixed |
| PII leakage | Sanitize logs; don't persist user data beyond session |

### 12.3 Production Readiness

| Aspect | Status | Notes |
|--------|--------|-------|
| Structured logging | Required | Use structlog with JSON format |
| Request tracing | Required | UUID per request, propagated to LLM calls |
| Metrics export | Recommended | Prometheus endpoint for latency, scores |
| Circuit breaker | Required | For LLM provider failover |
| Rate limiting | Required | Per-client + per-provider |
| Health checks | Required | /health with provider status |
| Graceful shutdown | Required | Finish in-flight requests |
| API versioning | Recommended | /v1/ prefix |

---

## 13. DELIVERABLES CHECKLIST

### Code Repository (GitHub)
- [ ] Complete source code with modular structure
- [ ] README with setup instructions
- [ ] requirements.txt and pyproject.toml
- [ ] .env.example (no real keys)
- [ ] All 10 test scenarios in data/
- [ ] Human reference emails for all scenarios
- [ ] Working FastAPI service
- [ ] LangGraph state machine implementation
- [ ] Advanced prompt template with documentation
- [ ] 3 custom metrics fully implemented
- [ ] Batch evaluation script
- [ ] Model comparison script
- [ ] Unit and integration tests
- [ ] CI/CD workflow (GitHub Actions)

### Final Report (PDF/Markdown)
- [ ] Prompt Template (documented with rationale)
- [ ] 3 Custom Metric Definitions + Logic
- [ ] Raw Evaluation Data (CSV/JSON)
- [ ] Comparative Analysis (single page)
  - [ ] Which model performed better
  - [ ] Biggest failure mode of lower performer
  - [ ] Production recommendation with justification

---

## 14. TIMELINE ESTIMATE

| Phase | Tasks | Duration |
|-------|-------|----------|
| **Setup** | Repo, env, dependencies, basic FastAPI | 2-3 hours |
| **Core Generation** | LangGraph, prompt engineering, LLM integration | 4-6 hours |
| **Test Data** | Write 10 scenarios + human references | 2-3 hours |
| **Metrics** | Implement FRS, TFI, SCS | 6-8 hours |
| **Evaluation** | Batch runner, reporter, comparison logic | 3-4 hours |
| **Testing** | Unit tests, integration tests, bug fixes | 3-4 hours |
| **Documentation** | README, report, architecture docs | 2-3 hours |
| **Total** | | **22-31 hours** |

---

## 15. KEY DECISIONS SUMMARY

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Primary LLM | Gemini 2.5 Flash | Best free tier (1,500 req/day), strong reasoning, 1M context |
| Comparison LLM | Groq Llama 3.3 70B | Fastest free tier, OpenAI-compatible, good instruction following |
| Judge LLM | Gemini 2.5 Flash (separate) | Avoids self-evaluation bias, same quality as generator |
| Prompt Technique | CoT + Few-Shot + Role-Playing | Maximizes quality via reasoning + examples + expertise context |
| Output Format | Markdown | Best practice: clean, renderable, portable |
| Evaluation | Hybrid (Python + LLM-as-Judge) | Automated for scale, LLM for nuanced judgment |
| Architecture | LangGraph + FastAPI | State persistence, retry logic, async API |
| Rate Limiting | Token bucket + failover | Essential for free tier reliability |

---

**Document Version:** 1.0  
**Last Updated:** 2026-06-18  
**Author:** Senior Software Engineer (10YOE, Ex-Google)  
**Status:** Ready for Implementation
