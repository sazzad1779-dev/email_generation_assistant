# 📋 EMAIL GENERATION ASSISTANT — TASK EXECUTION PLAN

> **Project:** Email Generation Assistant with Custom Evaluation Metrics & Model Comparison  
> **Tech Stack:** Python, FastAPI, LangChain, LangGraph, Pydantic, AsyncIO  
> **Primary LLM:** Google Gemini 2.5 Flash | **Comparison LLM:** Groq Llama 3.3 70B | **Judge LLM:** Gemini 2.5 Flash (separate instance)  
> **Total Estimated Effort:** 22–31 hours  
> **Version:** 1.0 | **Last Updated:** 2026-06-19  
> **Status:** ✅ Ready for Execution

---

## HOW TO USE THIS PLAN

1. **Phases** must be completed **sequentially** (each depends on prior phases).
2. **Tasks within a phase** can often be **parallelized** (noted per phase).
3. Each task has a **checklist** of sub-steps. Mark `[ ]` → `[x]` as you complete.
4. After each phase, run relevant tests to validate before moving on.
5. Use the **scripts/** directory to run evaluation, comparison, and server tasks.
6. Refer to `project_requirement_analysis.md` for full technical specifications.

---

## DEPENDENCY GRAPH

```
Phase 0 (Scaffolding) ─────────────────────────────────────────────────
    │
    ▼
Phase 1 (FastAPI Setup) ──────────────────────────────────────────────
    │
    ├────────────────────────────────────────────────────┐
    ▼                                                    ▼
Phase 2 (LangGraph Core)                          Phase 5 (Test Data)
    │                                                    │
    ├──────────┬──────────┐                              │
    ▼          ▼          ▼                              │
Phase 3a    Phase 3b    Phase 4                          │
(Prompts)   (LLM Prov.) (Rate Limit)                     │
    │          │          │                               │
    └──────────┼──────────┘                               │
               ▼                                          │
         Phase 6 (Metrics) ◄──────────────────────────────┘
               │
               ▼
         Phase 7 (Evaluation Engine)
               │
               ▼
         Phase 8 (Scripts & CLI)
               │
               ▼
         Phase 9 (Testing)
               │
               ▼
         Phase 10 (Documentation & Finalization)
```

> **Parallelization Tip:** Phases 3 (Prompts), 4 (LLM Providers), and 5 (Test Data) can be developed concurrently since they have no strict dependency on each other.

---

## PHASE 0: PROJECT SCAFFOLDING & INITIALIZATION ⏱️ 2–3 hrs

> **Goal:** Set up the entire project skeleton, environment, and all dependencies.

### Task 0.1 — Create Project Directory Structure

**Files to Create:**

```
email-generation-assistant/
├── src/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app entry point
│   ├── config.py                  # Pydantic Settings, env vars
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py              # FastAPI routers
│   │   ├── models.py              # Pydantic request/response models
│   │   ├── dependencies.py        # Auth, rate limiting, logging
│   │   └── middleware.py          # Error handling, CORS, tracing
│   ├── core/
│   │   ├── __init__.py
│   │   ├── email_generator.py     # Main orchestration logic
│   │   ├── state.py               # LangGraph state definitions
│   │   ├── graph.py               # LangGraph compilation
│   │   └── exceptions.py          # Custom exceptions
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── templates/
│   │   │   ├── base.j2            # Jinja2 base template
│   │   │   ├── system_role.j2     # Role-playing system prompt
│   │   │   ├── few_shot.j2        # Few-shot example injection
│   │   │   └── cot_instructions.j2 # Chain-of-thought guidance
│   │   ├── examples/
│   │   │   ├── formal_followup.json
│   │   │   ├── casual_request.json
│   │   │   └── ... (6 more)
│   │   ├── tone_profiles/
│   │   │   ├── formal.yaml
│   │   │   ├── casual.yaml
│   │   │   ├── urgent.yaml
│   │   │   ├── empathetic.yaml
│   │   │   ├── persuasive.yaml
│   │   │   └── apologetic.yaml
│   │   └── registry.py            # Dynamic example retrieval
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── providers/
│   │   │   ├── __init__.py
│   │   │   ├── base.py            # Abstract provider interface
│   │   │   ├── gemini_provider.py # Google Gemini integration
│   │   │   ├── groq_provider.py   # Groq/OpenAI-compatible
│   │   │   └── router.py          # Failover logic
│   │   ├── rate_limiter.py        # Token bucket, quota tracking
│   │   └── judge.py               # LLM-as-a-Judge wrapper
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics/
│   │   │   ├── __init__.py
│   │   │   ├── base.py            # Abstract metric interface
│   │   │   ├── fact_recall.py     # FRS implementation
│   │   │   ├── tone_fidelity.py   # TFI implementation
│   │   │   └── structural_coherence.py # SCS implementation
│   │   ├── runner.py              # Batch evaluation orchestration
│   │   ├── reporter.py            # Report generation (JSON/CSV/MD)
│   │   └── comparison.py          # A/B comparison logic
│   └── utils/
│       ├── __init__.py
│       ├── text_processing.py     # Cleaning, normalization
│       ├── validators.py          # Custom validators
│       └── logging_config.py      # Structured logging
├── data/
│   ├── test_scenarios.json        # 10 test scenarios
│   ├── human_references/          # Gold-standard emails
│   │   ├── scenario_01.md
│   │   ├── scenario_02.md
│   │   └── ... (up to scenario_10.md)
│   └── few_shot_examples/         # Prompt examples
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Pytest fixtures
│   ├── test_api.py                # API endpoint tests
│   ├── test_metrics.py            # Metric unit tests
│   ├── test_graph.py              # LangGraph flow tests
│   └── test_integration.py        # End-to-end tests
├── reports/                       # Generated evaluation reports
│   └── .gitkeep
├── scripts/
│   ├── run_server.py              # `python scripts/run_server.py`
│   ├── run_evaluation.py          # `python scripts/run_evaluation.py`
│   └── run_comparison.py          # `python scripts/run_comparison.py`
└── docs/
    ├── architecture.md            # Architecture documentation
    ├── prompt_template.md         # Documented prompt template
    ├── metrics_spec.md            # Metric definitions and logic
    └── comparison_report.md       # Final analysis summary
```

**Checklist:**

- [x] Create all directories as per structure above
- [x] Add `__init__.py` to every Python package directory
- [x] Create `.gitkeep` in `reports/` and `data/human_references/`
- [x] Create `.gitignore` (Python + venv + .env + __pycache__ + *.pyc + reports/*.json)
- [x] Verify tree structure matches specification

---

### Task 0.2 — Create Configuration & Environment Files

**Checklist:**

- [x] Create `requirements.txt` with all dependencies from §10 of requirements doc
- [x] Create `pyproject.toml` for modern Python packaging
- [x] Create `.env.example` with template variables (NO real keys):

```env
# LLM Provider API Keys
GEMINI_API_KEY=
GROQ_API_KEY=

# Application Settings
ENVIRONMENT=development
LOG_LEVEL=INFO
PORT=8000
HOST=0.0.0.0

# Generation Defaults
MAX_WORDS_DEFAULT=300
TEMPERATURE=0.0
TOP_P=0.95

# Provider Rate Limits
GEMINI_RPM=10
GEMINI_DAILY_LIMIT=1500
GROQ_RPM=30
GROQ_DAILY_LIMIT=1000

# Judge LLM Config
JUDGE_MODEL=gemini-2.5-flash
JUDGE_TEMPERATURE=0.0
JUDGE_RUNS=3
```

- [x] Create `src/config.py` using Pydantic Settings with all above variables
- [x] Add validation for required API keys raising clear error if missing
- [x] Add `Settings` singleton instance

---

### Task 0.3 — Setup Virtual Environment & Install Dependencies (with `uv`)

> **Using `uv`** — a fast Python package installer, resolver, and virtual environment manager (Rust-based alternative to pip/venv).

**Checklist:**

- [x] Install uv (if not already installed): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
- [x] Create virtual environment: `uv venv .venv`
- [ ] Activate venv: `.venv\Scripts\activate` (Windows) — *execution policy blocks activation, used `.venv\Scripts\python.exe` directly instead*
- [x] Install all dependencies from requirements.txt: `uv pip install -r requirements.txt`
- [ ] (Alternative) Or use pyproject.toml directly: `uv sync`
- [x] Download spaCy model: `uv run python -m spacy download en_core_web_md`
- [x] Download NLTK data: `uv run python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"`
- [x] Verify all packages: `uv run python -c "import fastapi, langchain, langgraph, pydantic, sentence_transformers, spacy, nltk"` → no errors
- [x] Check uv is installed: `uv --version`

---

## PHASE 1: CORE FRAMEWORK & FASTAPI SETUP ⏱️ 2–3 hrs

> **Goal:** Set up FastAPI application with all endpoints, middleware, and error handling.
> **Parallelizable:** Tasks 1.1, 1.2, and 1.3 can be done together.

### Task 1.1 — FastAPI Application Entry Point (`src/main.py`)

**Checklist:**

- [x] Initialize FastAPI app with title="Email Generation Assistant", version="1.0.0"
- [x] Add lifespan handler for startup (init providers, check health) / shutdown (cleanup)
- [x] Include routers from `src.api.routes` with `/v1` prefix
- [x] Add CORS middleware (allow all origins for development)
- [x] Add request ID tracing middleware (UUID per request)
- [x] Add structured logging middleware (structlog)
- [x] Global exception handler → returns standard `ErrorResponse`
- [x] Register exception handlers: RateLimitException (429), LLMFailureException (503), ValidationError (422), ProviderQuotaExhausted (429), Generic Exception (500)
- [x] Configure OpenAPI/Swagger docs with proper metadata

---

### Task 1.2 — Pydantic Models (`src/api/models.py`)

**Models to Define:**

| Model | Fields | Validators | Used By |
|-------|--------|------------|---------|
| `ToneEnum` | FORMAL, CASUAL, URGENT, EMPATHETIC, PERSUASIVE, APOLOGETIC | — | All endpoints |
| `EmailRequest` | intent, key_facts, tone, recipient_name?, sender_name?, max_words | intent≥10 chars, facts 1-10 items each≥5 chars, reject vague intent, clamp max_words 50-800 | POST /generate |
| `EmailResponse` | email (str), metadata (Dict), quality_flags (List[str]) | — | POST /generate |
| `EvaluateRequest` | email, reference_email?, request (EmailRequest) | — | POST /evaluate |
| `EvaluateResponse` | fact_recall_score, tone_fidelity_index, structural_coherence_score, overall_score, breakdown | — | POST /evaluate |
| `BatchEvaluateRequest` | model (str), scenarios (List[int]?) | — | POST /batch-evaluate |
| `BatchEvaluateResponse` | scenarios_evaluated, results, aggregate, report_path | — | POST /batch-evaluate |
| `CompareRequest` | model_a, model_b, scenarios | — | POST /compare |
| `CompareResponse` | comparison_id, model_a_results, model_b_results, winner, margin, failure_analysis, recommendation | — | POST /compare |
| `HealthResponse` | status, providers (dict), version | — | GET /health |
| `ErrorResponse` | error_code, message, details?, request_id, retry_after? | — | All errors |

**Validation Rules for `EmailRequest`:**

```python
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
    forbidden = ["email", "write", "compose"]
    if any(word in v.lower() for word in forbidden) and len(v) < 30:
        raise ValueError("Intent must be specific, not generic")
    return v
```

**Checklist:**

- [x] `ToneEnum` defined with all 6 tones
- [x] `EmailRequest` with all fields + validators for all corner cases (§3.1, §7.1)
- [x] `EmailResponse` with email + metadata + quality_flags
- [x] `EvaluateRequest` / `EvaluateResponse`
- [x] `BatchEvaluateRequest` / `BatchEvaluateResponse`
- [x] `CompareRequest` / `CompareResponse`
- [x] `HealthResponse` with provider status dict
- [x] `ErrorResponse` standard schema for all errors

---

### Task 1.3 — API Routes (`src/api/routes.py`) & Dependencies

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/generate` | Generate single email (Model A or B) |
| `POST` | `/evaluate` | Run custom metrics on single email |
| `POST` | `/batch-evaluate` | Run full 10-scenario evaluation |
| `POST` | `/compare` | Run A/B model comparison |
| `GET` | `/health` | Service health + rate limit status |

**Dependencies (`src/api/dependencies.py`):**

- Rate limiting dependency (per-client token bucket)
- Provider availability check
- Request ID generation and logging context

**Checklist:**

- [x] `POST /generate` — validate request → call `generate_email()` → return response
- [x] `POST /evaluate` — run all 3 metrics → return scores
- [x] `POST /batch-evaluate` — run 10-scenario evaluation → return aggregate
- [x] `POST /compare` — run A/B comparison → return winner + analysis
- [x] `GET /health` — check providers + quota → return status
- [x] Proper HTTP status codes: 200, 201, 400, 422, 429, 500, 503
- [x] All errors returned as standard `ErrorResponse`
- [x] Rate limit info in response headers (`X-RateLimit-Remaining`, `X-Request-ID`)

---

### Task 1.4 — Middleware & Custom Exceptions

**Files to Create:**

- `src/api/middleware.py`
- `src/core/exceptions.py`

**`src/core/exceptions.py`:**

```python
class RateLimitException(HTTPException):
    def __init__(self, retry_after: int):
        super().__init__(status_code=429, detail="Rate limit exceeded")
        self.retry_after = retry_after

class LLMFailureException(HTTPException):
    def __init__(self, provider: str, detail: str):
        super().__init__(status_code=503, detail=f"LLM provider {provider} failed: {detail}")

class ProviderQuotaExhausted(HTTPException):
    def __init__(self, provider: str, reset_at: str):
        super().__init__(status_code=429, detail=f"Provider {provider} quota exhausted")
        self.reset_at = reset_at

class ValidationError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=422, detail=detail)
```

**`src/api/middleware.py` Checklist:**

- [x] Request ID middleware (UUID per request, add to request state + response headers)
- [x] CORS middleware (allow all origins in dev, configurable in prod)
- [x] Logging middleware (structlog: method, path, status, duration, request_id)
- [x] Global exception handler for unhandled errors
- [x] Rate limit error handler → 429 with retry_after

---

### Task 1.5 — Structured Logging (`src/utils/logging_config.py`)

**Checklist:**

- [x] Configure `structlog` with JSON formatter
- [x] Add console handler (colored for dev, JSON for prod)
- [x] Add file handler with rotation
- [x] Configure log levels per environment (DEBUG for dev, INFO for prod)
- [x] Add request_id to all log entries automatically
- [x] Create convenience functions: `get_logger(name)`

---

## PHASE 2: LANGGRAPH STATE MACHINE & EMAIL GENERATION ⏱️ 4–6 hrs

> **Goal:** Build the LangGraph-powered email generation pipeline with retry/fallback.
> **Cannot parallelize:** Tasks 2.1 → 2.2 → 2.3 must be sequential.

### Task 2.1 — State Definition (`src/core/state.py`)

**Checklist:**

- [ ] Define `EmailGenerationState` TypedDict:

```python
class EmailGenerationState(TypedDict):
    request: EmailRequest           # Original request (immutable)
    prompt: str                     # Assembled prompt
    raw_output: str                 # Raw LLM output
    cleaned_email: str              # Post-processed email
    metadata: Dict[str, Any]        # model_used, tokens, latency
    retry_count: int                # Current retry attempt
    error: Optional[str]            # Error message if failed
    scores: Optional[Dict[str, float]]  # Quality scores
    quality_passed: bool            # Quality gate result
```

- [ ] Define default values for state initialization

---

### Task 2.2 — Graph Nodes Implementation (`src/core/graph.py`)

**Node 1: `validate_input`**

- [ ] Validate and enrich input data
- [ ] Detect intent category for few-shot selection (follow-up, request, announcement, escalation, etc.)
- [ ] Estimate complexity score based on facts + tone
- [ ] Check for PII in facts → sanitize logs
- [ ] Handle corner cases: ambiguous intent, fact contradictions, PII detection

**Node 2: `construct_prompt`**

- [ ] Load few-shot examples dynamically from registry (top 3 by semantic similarity)
- [ ] Inject role-playing system prompt (Dr. Elena Voss persona)
- [ ] Add chain-of-thought instructions (4 reasoning steps)
- [ ] Add tone-specific guidance from tone profiles
- [ ] Format facts for prompt injection
- [ ] Add word count constraints
- [ ] Handle edge: no matching examples → use defaults

**Node 3: `call_llm`**

- [ ] Call primary LLM provider (Gemini) with rate limit handling
- [ ] Capture metadata: model_used, prompt_tokens, completion_tokens, latency_ms
- [ ] If rate limited → switch to fallback provider (Groq)
- [ ] If all providers fail → set error state
- [ ] Handle: cut-off output, safety filter refusal, wrong language, placeholder text

**Node 4: `post_process`**

- [ ] Strip `<thinking>...</thinking>` tags via regex
- [ ] Extract subject line (ensure it exists)
- [ ] Ensure proper greeting (Hi/Dear/Hello) and closing (Best/Regards/Sincerely)
- [ ] Validate word count (within ±20% of target)
- [ ] Add signature if missing from sender_name
- [ ] Normalize line endings and whitespace
- [ ] Handle: missing greeting, missing closing, placeholder text `[Name]`

**Node 5: `quality_check`**

- [ ] Quick heuristic checks:
  - Has greeting? (regex: `Hi|Dear|Hello|Hey`)
  - Has closing? (regex: `Best|Regards|Sincerely|Thanks|Cheers`)
  - Has subject line? (line starting with `Subject:`)
  - Word count within ±20% of target?
  - No placeholder text like `[Name]`, `[Company]`, `[Email]`?
- [ ] If fail AND `retry_count < 2` → route back to `construct_prompt` with stronger instructions
- [ ] If fail AND `retry_count >= 2` → set `quality_passed=False`, add warning flag, continue

**Checklist:**

- [ ] All 5 node functions implemented
- [ ] `validate_input` enriches state with intent category + complexity
- [ ] `construct_prompt` assembles complete prompt from templates
- [ ] `call_llm` handles rate limits + fallback
- [ ] `post_process` cleans output and fixes structure
- [ ] `quality_check` validates with auto-retry

---

### Task 2.3 — Graph Compilation & Conditional Edges

**Checklist:**

- [ ] Define `route_after_quality(state) -> str`:

```python
def route_after_quality(state: EmailGenerationState) -> str:
    if state.get("error"):
        return "error_handler"
    if state["retry_count"] < 2 and not state.get("quality_passed", True):
        return "construct_prompt"  # Retry with stronger prompt
    return "end"
```

- [ ] Build graph with `StateGraph(EmailGenerationState)`:

```python
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

- [ ] Add error handler node for unrecoverable failures
- [ ] Test graph with mock data (mock LLM responses)

---

### Task 2.4 — Main Orchestrator (`src/core/email_generator.py`)

**Checklist:**

- [ ] `async def generate_email(request: EmailRequest, preferred_model: str = "gemini") -> EmailResponse`
- [ ] Initialize `EmailGenerationState` from request
- [ ] Invoke compiled graph with `await graph.ainvoke(initial_state)`
- [ ] Extract response from final state
- [ ] Handle errors: if state has error → raise appropriate exception
- [ ] Return `EmailResponse` with cleaned email + metadata + quality_flags
- [ ] Add retry_count and quality_passed to metadata

---

## PHASE 3: PROMPT ENGINEERING LAYER ⏱️ 2–3 hrs

> **Goal:** Build the advanced prompt system with Jinja2 templates, few-shot examples, and tone profiles.
> **Can be parallelized with Phase 4 and Phase 5.**

### Task 3.1 — Prompt Templates (Jinja2)

**`src/prompts/templates/system_role.j2`:**

```
You are Dr. Elena Voss, a senior executive communication coach who has trained 
C-suite leaders at Fortune 500 companies for 18 years. You specialize in 
translating business intent into precisely-toned, fact-rich professional emails.

Your emails are known for:
- Seamless fact integration (never list-like, always narrative)
- Exact tone matching ({{ tone_profile.markers }})
- Strategic structure (hook → context → facts → call-to-action → close)
```

**`src/prompts/templates/cot_instructions.j2`:**

```
Before writing the email, think through:
1. What is the relationship between sender and recipient?
2. Which facts are most important and where should they appear?
3. What tone markers (word choice, sentence length) achieve the requested tone?
4. How should the email open and close for maximum impact?

Write your thinking in <thinking> tags, then the email.
```

**`src/prompts/templates/few_shot.j2`:**

```
FEW-SHOT EXAMPLES:

{% for example in examples %}
[Example {{ loop.index }} — {{ example.tone | capitalize }} / {{ example.category }}]
Intent: {{ example.intent }}
Facts: {{ example.facts | join(", ") }}
Tone: {{ example.tone }}

{{ example.output }}
---
{% endfor %}
```

**`src/prompts/templates/base.j2`:**

```
{{ system_prompt }}

{{ few_shot_examples }}

{{ cot_instructions }}

TASK:
Generate a professional email with the following specifications:

INTENT: {{ intent }}
KEY FACTS (MUST ALL BE INCLUDED):
{% for fact in facts %}
- {{ fact }}
{% endfor %}

TONE: {{ tone }}
TONE PROFILE:
{{ tone_specific_guidance }}

CONSTRAINTS:
- Word count: Between {{ min_words }} and {{ max_words }} words
- Must include all key facts naturally (not as a list)
- Must match the {{ tone }} tone precisely
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

**Checklist:**

- [ ] `system_role.j2` — Dr. Elena Voss persona with tone adaptation
- [ ] `cot_instructions.j2` — 4-step reasoning process
- [ ] `few_shot.j2` — Dynamic example injection with for loop
- [ ] `base.j2` — Complete prompt assembly template with all sections
- [ ] All templates render correctly with Jinja2

---

### Task 3.2 — Few-Shot Examples Registry (`src/prompts/registry.py`)

**Checklist:**

- [ ] `load_examples_by_intent(intent: str, top_k: int = 3) -> List[Dict]`
- [ ] Use sentence-transformers for semantic intent matching
- [ ] Cache embeddings for performance (LRU cache)
- [ ] Fallback: return default examples if no match found
- [ ] Load examples from JSON files in `src/prompts/examples/`

**Example JSON Format (`src/prompts/examples/formal_followup.json`):**

```json
{
    "id": "ex_001",
    "category": "follow-up",
    "intent": "Follow up after quarterly business review",
    "facts": ["Q3 revenue up 12%", "New hire Sarah starts Monday", "Budget approved"],
    "tone": "formal",
    "thinking": "1. Relationship: CEO to Board member...",
    "output": "Subject: Q3 Performance Update...\n\nDear Ms. Richardson,...",
    "word_count": 185
}
```

**Example Files to Create (8 total):**

- [ ] `formal_followup.json` — Formal follow-up after meeting
- [ ] `casual_request.json` — Casual team announcement
- [ ] `urgent_escalation.json` — Urgent security escalation
- [ ] `empathetic_apology.json` — Empathetic delay notification
- [ ] `persuasive_proposal.json` — Persuasive tool justification
- [ ] `formal_announcement.json` — Formal policy announcement
- [ ] `apologetic_payment.json` — Apologetic late payment
- [ ] `casual_congratulations.json` — Casual promotion congrats

---

### Task 3.3 — Tone Profiles (YAML)

**`src/prompts/tone_profiles/formal.yaml`:**

```yaml
name: formal
markers:
  contractions: false
  use_titles: true
  politeness: high
sentence_structure:
  avg_length: 20-30
  complexity: high
  imperative_ratio: 0.1
word_choice:
  - "would like"
  - "please advise"
  - "I am writing to"
  - "at your earliest convenience"
  - "please find attached"
avoid:
  - contractions
  - slang
  - exclamation marks
example_phrases:
  greeting: "Dear Mr./Ms. {Last Name},"
  closing: "Respectfully,"
  cta: "Please do not hesitate to contact me"
```

**Tone Profiles Summary:**

| Tone | Markers | Sentence Structure | Word Choice |
|------|---------|-------------------|-------------|
| Formal | No contractions, full titles, "would like", "please advise" | Complex, varied length | Precise, technical |
| Casual | Contractions OK, first names, "hey", "thanks" | Short to medium, conversational | Warm, accessible |
| Urgent | Direct imperatives, "immediately", "action required" | Short, punchy | Forceful, clear |
| Empathetic | "I understand", "I know this is difficult" | Gentle pacing, pauses | Supportive, validating |
| Persuasive | "Imagine", "opportunity", "benefit" | Build-up, rhetorical | Compelling, confident |
| Apologetic | "I apologize", "I take full responsibility" | Humble, deferential | Remorseful, accountable |

**Checklist:**

- [ ] `formal.yaml` — No contractions, full titles, structured
- [ ] `casual.yaml` — Contractions OK, first names, warm
- [ ] `urgent.yaml` — Direct imperatives, time-sensitive
- [ ] `empathetic.yaml` — Supportive, validating feelings
- [ ] `persuasive.yaml` — Benefit-focused, compelling
- [ ] `apologetic.yaml` — Remorseful, accountable
- [ ] All profiles have: markers, sentence_structure, word_choice, avoid, example_phrases
- [ ] `load_tone_profile(tone: str) -> Dict` function in `__init__.py`

---

## PHASE 4: LLM PROVIDER INTEGRATION ⏱️ 2–3 hrs

> **Goal:** Implement provider abstraction, Gemini + Groq integrations, rate limiting, and LLM-as-a-Judge.
> **Can be parallelized with Phase 3 and Phase 5.**

### Task 4.1 — Abstract Provider Interface (`src/llm/providers/base.py`)

**Checklist:**

- [ ] Define `LLMResponse` dataclass: content, model_used, provider, prompt_tokens, completion_tokens, latency_ms
- [ ] Define abstract `LLMProvider` class:
  - `async def generate(self, prompt: str, **kwargs) -> LLMResponse`
  - `async def check_health(self) -> dict`
  - `provider_name -> str`
  - `quota_remaining -> int`
  - `rpm_remaining -> int`
- [ ] Define `ProviderHealth` dataclass

---

### Task 4.2 — Gemini Provider (`src/llm/providers/gemini_provider.py`)

**Checklist:**

- [ ] Import `google-generativeai` SDK
- [ ] Configure with `GEMINI_API_KEY`
- [ ] Use `gemini-2.5-flash` model
- [ ] Async generation with `generate_content_async`
- [ ] Custom safety settings (least restrictive for business emails)
- [ ] Generation config: `temperature=0.0, top_p=0.95, max_output_tokens=2048`
- [ ] System instruction: "You are a professional email writer..."
- [ ] Track usage: daily request count, RPM counter
- [ ] Health check: send simple prompt, verify response
- [ ] Error handling: rate limit (429), timeout, invalid key, model overloaded
- [ ] Return `LLMResponse` with all metadata

---

### Task 4.3 — Groq Provider (`src/llm/providers/groq_provider.py`)

**Checklist:**

- [ ] Import OpenAI-compatible client (groq or openai with custom base_url)
- [ ] Configure with `GROQ_API_KEY`
- [ ] Use `llama-3.3-70b-versatile` model
- [ ] Async generation with streaming support
- [ ] Generation config: `temperature=0.0, max_tokens=2048`
- [ ] Track usage: daily request count, RPM counter
- [ ] Health check: send simple prompt, verify response
- [ ] Error handling: rate limit (429), timeout, invalid key
- [ ] Return `LLMResponse` with all metadata

---

### Task 4.4 — Provider Router with Failover (`src/llm/providers/router.py`)

**Checklist:**

- [ ] `async def generate_with_failover(prompt: str, preferred: str = "gemini") -> LLMResponse`:
  1. Try preferred provider first
  2. If rate limited → log warning, try next in chain
  3. If all providers fail → raise `LLMFailureException`
  4. Max 2 retries per provider (with exponential backoff)
- [ ] Implement exponential backoff with jitter: `sleep(min(2^attempt + random(0,1), 30))`
- [ ] Implement circuit breaker pattern:
  - >3 consecutive failures → circuit open (skip for 60s)
  - After 60s → half-open (try one request)
  - If success → circuit closed
- [ ] Fallback chain: `Gemini → Groq → Error`
- [ ] Pre-flight check: verify quota available for N requests before batch jobs

---

### Task 4.5 — Rate Limiter (`src/llm/rate_limiter.py`)

**Checklist:**

- [ ] Implement `TokenBucket` class:
  - `capacity`, `tokens`, `refill_rate`, `refill_period`
  - `async def acquire(tokens=1) -> bool`
  - Auto-refill based on elapsed time
- [ ] Implement `RateLimiter` class (per-provider buckets):
  - Gemini: capacity=10, refill_rate=10, refill_period=60 (10 RPM)
  - Groq: capacity=30, refill_rate=30, refill_period=60 (30 RPM)
  - Daily quota tracking (reset at midnight UTC)
- [ ] `async def wait_and_acquire(provider: str) -> bool` — blocking wait with timeout
- [ ] `get_quota_status(provider: str) -> Dict` — remaining daily, RPM remaining, reset time
- [ ] Persist quota state to disk (`reports/quota_state.json`) for resilience
- [ ] Integrate with provider router (check before calling)

---

### Task 4.6 — LLM-as-a-Judge (`src/llm/judge.py`)

**Checklist:**

- [ ] Create `LLMJudge` class wrapping Gemini 2.5 Flash (separate instance from generator)
- [ ] `async def judge_fact_inclusion(email: str, facts: List[str]) -> Dict`:
  - Judge prompt: "For each fact, determine if it's present in the email..."
  - Return per-fact verdict: present/paraphrased/missing
- [ ] `async def judge_tone_accuracy(email: str, target_tone: str) -> Dict`:
  - Judge prompt: "Rate how well this email matches {tone} tone on scale 1-5..."
  - Return score + reasoning
- [ ] `async def judge_hallucinations(email: str, facts: List[str]) -> List[str]`:
  - Judge prompt: "Identify any claims in the email not supported by these facts..."
  - Return list of hallucinated claims
- [ ] For consistency: run 3 times, take median score
- [ ] Structured judge prompts with clear scoring rubrics
- [ ] Avoid self-evaluation bias: always use separate model instance

---

## PHASE 5: TEST DATA & HUMAN REFERENCES ⏱️ 2–3 hrs

> **Goal:** Create 10 realistic test scenarios and write gold-standard human reference emails.
> **Can be parallelized with Phase 3 and Phase 4.**

### Task 5.1 — Test Scenarios JSON (`data/test_scenarios.json`)

**10 Scenarios Specification:**

| # | Intent | Key Facts | Tone | Complexity | Target Words |
|---|--------|-----------|------|------------|-------------|
| 1 | Follow up after product demo meeting | ["Demo held on June 15", "Client liked AI feature", "Pricing discussion pending", "Next steps: technical deep-dive"] | Formal | Medium | 180 |
| 2 | Request deadline extension for project deliverable | ["Original deadline: July 1", "Team member illness", "Need 2-week extension", "Quality will improve"] | Empathetic | Medium | 200 |
| 3 | Inform team about office relocation | ["Moving to Building C", "Effective August 1", "New address: 450 Market St", "Parking info attached"] | Casual | Low | 150 |
| 4 | Escalate unresolved security incident | ["Incident #4421", "48 hours no response", "Customer data potentially exposed", "Immediate action required"] | Urgent | High | 170 |
| 5 | Congratulate colleague on promotion | ["Promoted to VP Engineering", "Effective immediately", "10 years at company", "Team celebration Friday"] | Casual | Low | 140 |
| 6 | Request proposal details from vendor | ["RFP submitted May 20", "Need breakdown by module", "Budget cap: $150K", "Decision by June 30"] | Formal | Medium | 190 |
| 7 | Apologize for delayed payment to contractor | ["Invoice #8832", "Payment 15 days late", "Bank transfer processing", "Will include late fee"] | Apologetic | Medium | 160 |
| 8 | Announce new company policy on remote work | ["Effective Q3", "3 days office, 2 days remote", "Core collaboration days: Tue-Thu", "Hot-desking system"] | Formal | High | 220 |
| 9 | Persuade stakeholder to approve new tool purchase | ["Current tool has 99.5% uptime", "New tool reduces cost 30%", "Migration takes 2 weeks", "ROI: 6 months"] | Persuasive | High | 210 |
| 10 | Express condolences to team on loss of colleague | ["Sarah passed away June 10", "15 years of service", "Memorial service June 20", "Counseling available"] | Empathetic | High | 180 |

**Checklist:**

- [ ] All 10 scenarios defined with realistic business context
- [ ] All 6 tones covered at least once
- [ ] Complexity levels balanced: 3 Low, 4 Medium, 3 High
- [ ] Each scenario has 4 key facts (specific, 5+ chars each)
- [ ] JSON is valid and parseable
- [ ] Categories span: follow-up, request, announcement, escalation, congratulations, proposal, apology, policy, persuasion, condolences

---

### Task 5.2 — Human Reference Emails (`data/human_references/scenario_*.md`)

**Writing Guidelines:**

- All facts integrated naturally into narrative (no bullet lists)
- Tone matches precisely using linguistic markers from §3.3
- Word count within ±10% of target
- Include: subject line → greeting → body (hook, context, facts, CTA) → closing → signature
- Professional formatting in Markdown

**Reference Format:**

```markdown
# Scenario 1: Follow-up after Product Demo

**Intent:** Follow up after product demo meeting to discuss next steps
**Tone:** Formal
**Target Word Count:** 180

---

Subject: Follow-Up: Product Demo Discussion and Next Steps

Dear Mr. Chen,

I hope this message finds you well. I wanted to express my sincere appreciation for the time you and your team dedicated to the product demonstration on June 15. It was a productive session, and I was particularly pleased to see your positive response to our AI feature, which we believe aligns closely with your current technology roadmap.

Regarding the next phase of our discussion, I understand that pricing requires further consideration, and I want to assure you that we are prepared to work with you to find a mutually beneficial arrangement. To move forward, I suggest we schedule a technical deep-dive session with our engineering team to address any specific integration questions you may have.

Please let me know your availability for the coming week, and I will coordinate the necessary arrangements.

Warm regards,

Alice Johnson
Senior Account Manager
```

**Checklist:**

- [ ] `scenario_01.md` — Formal follow-up after demo (165-195 words)
- [ ] `scenario_02.md` — Empathetic deadline extension (180-220 words)
- [ ] `scenario_03.md` — Casual office relocation (135-165 words)
- [ ] `scenario_04.md` — Urgent security escalation (153-187 words)
- [ ] `scenario_05.md` — Casual promotion congratulations (126-154 words)
- [ ] `scenario_06.md` — Formal vendor proposal request (171-209 words)
- [ ] `scenario_07.md` — Apologetic delayed payment (144-176 words)
- [ ] `scenario_08.md` — Formal remote work policy (198-242 words)
- [ ] `scenario_09.md` — Persuasive tool purchase (189-231 words)
- [ ] `scenario_10.md` — Empathetic condolences (162-198 words)
- [ ] All references: grammar-checked, tone-verified, fact-complete

---

## PHASE 6: CUSTOM EVALUATION METRICS ⏱️ 6–8 hrs ⚠️ (MOST COMPLEX PHASE)

> **Goal:** Implement 3 custom metrics — FRS, TFI, SCS — with comprehensive scoring logic and corner-case handling.
> **Parallelizable:** Tasks 6.2, 6.3, and 6.4 can be developed concurrently after Task 6.1.

### Task 6.1 — Abstract Metric Interface (`src/evaluation/metrics/base.py`)

**Checklist:**

- [ ] Define abstract `BaseMetric` class:

```python
class BaseMetric(ABC):
    @abstractmethod
    async def score(self, email: str, **kwargs) -> Dict:
        """Returns {'score': float, 'breakdown': dict, ...}"""
        pass
    
    @property
    @abstractmethod
    def metric_name(self) -> str: ...
    
    @property
    @abstractmethod
    def weight(self) -> float: ...
```

- [ ] Standardized return format: `{"score": 0-100, "breakdown": {...}, "details": {...}}`

---

### Task 6.2 — Fact Recall Score (FRS) — `src/evaluation/metrics/fact_recall.py`

**Scoring Formula:**

```
FRS = max(0, avg_per_fact_score * 100 + integration_bonus - hallucination_penalty)

Per-fact scoring:
  - Similarity ≥ 0.85 → 1.0 (exact/near-exact)
  - Similarity 0.60-0.85 → 0.7 (paraphrased but accurate)
  - Similarity 0.40-0.60 → 0.4 (partially mentioned)
  - Similarity < 0.40 → 0.0 (missing)

Bonus: +5 if all facts integrated naturally (not listed)
Penalty: -10 per hallucinated fact
```

**Implementation Checklist:**

**Component A: Statement Extraction**

- [ ] Use spaCy NER to extract named entities (people, dates, orgs, money, percentages)
- [ ] Regex patterns for: dates (`June 15`, `2026-06-15`), numbers, percentages (`12%`), currency (`$150K`)
- [ ] Split compound sentences into discrete claims
- [ ] Extract subject line content separately

**Component B: Semantic Matching**

- [ ] Load `sentence-transformers/all-MiniLM-L6-v2` model
- [ ] Encode each input fact into embedding vector
- [ ] Encode each extracted statement into embedding vector
- [ ] Compute cosine similarity matrix
- [ ] For each fact, find best match score
- [ ] Convert similarity to score bucket

**Component C: Hallucination Check**

- [ ] LLM-as-Judge: are there claims in email not in input facts?
- [ ] Penalty: −10 per hallucination
- [ ] Edge: numerical unit conversion ("12%" → "twelve percent") → NOT a hallucination

**Component D: Natural Integration Check**

- [ ] Check if facts appear as a bullet list → no bonus
- [ ] If facts spread across narrative prose → +5 bonus
- [ ] Heuristic: ratio of newlines between fact-bearing sentences

**Corner Cases:**

- Date expressed as range vs exact → semantic match handles
- Fact split across sentences → aggregate embeddings
- Numerical fact with unit conversion → normalize before matching
- Fact implied but not stated → LLM judge decides conservatively (no credit)

**Checklist:**

- [ ] `FactRecallScorer` class with all 4 components
- [ ] Statement extraction (NER + regex)
- [ ] Semantic matching with sentence-transformers
- [ ] Hallucination detection via LLM judge
- [ ] Natural integration bonus
- [ ] Corner case handling (dates, numbers, units, split facts)
- [ ] Returns: `{score, per_fact_breakdown, hallucinations_found, natural_integration}`

---

### Task 6.3 — Tone Fidelity Index (TFI) — `src/evaluation/metrics/tone_fidelity.py`

**Scoring Formula:**

```
TFI = 0.3 * Lexical_Score + 0.25 * Syntactic_Score + 0.25 * Pragmatic_Score + 0.2 * Emotional_Score
```

**Implementation Checklist:**

**Lexical Score (30%) — Word Choice Analysis**

- [ ] Contraction ratio: count contractions / total words
- [ ] Modal verb frequency: formal→would/could, urgent→must/need
- [ ] Politeness markers: please advise, thank you, I would appreciate, cheers
- [ ] Jargon density: technical/business terms per sentence
- [ ] Compare each feature against target tone profile

**Syntactic Score (25%) — Sentence Structure**

- [ ] Average sentence length (words per sentence)
- [ ] Subordination complexity (clauses per sentence via spaCy)
- [ ] Imperative sentence ratio
- [ ] Sentence length variation coefficient (std/mean)

**Pragmatic Score (25%) — Communicative Function**

- [ ] Directness level: direct vs hedged statements
- [ ] Hedging frequency: might, perhaps, I think, possibly, maybe
- [ ] Question ratio: interrogative sentences / total sentences

**Emotional Score (20%) — Affective Markers**

- [ ] Positive/negative word ratio (NLTK sentiment or custom lexicon)
- [ ] Intensity modifiers: very, extremely, highly, slightly, somewhat
- [ ] Empathy markers: I understand, I know, I can imagine

**Corner Cases:**

- Mixed tone → segment analysis, worst segment dominates
- Tone drift in long emails → break into segments, score each, aggregate with penalty
- Sarcasm/irony → rare in business, LLM judge catches if present
- Cultural tone variations → standardize to US business English baseline

**Checklist:**

- [ ] `ToneFidelityScorer` class with 4 dimension scorers
- [ ] Lexical features: contraction ratio, modals, politeness, jargon
- [ ] Syntactic features: sentence length, subordination, imperatives
- [ ] Pragmatic features: directness, hedging, questions
- [ ] Emotional features: sentiment, intensity, empathy
- [ ] Weighted aggregation to final TFI score
- [ ] Tone drift detection (segment analysis)
- [ ] Returns: `{score, dimension_breakdown, detected_tone, confidence}`

---

### Task 6.4 — Structural Coherence Score (SCS) — `src/evaluation/metrics/structural_coherence.py`

**Scoring Formula:**

```
SCS = 0.25 * Grammar + 0.25 * Fluency + 0.20 * Conciseness + 0.15 * Structure + 0.15 * Professionalism
```

**Implementation Checklist:**

**Grammar Score (25%) — LanguageTool Integration**

- [ ] Integrate `language-tool-python`
- [ ] Base=100, −5 per spelling error, −10 per grammar error, −3 per punctuation error
- [ ] Floor at 0
- [ ] Return top 5 issues for feedback

**Fluency Score (25%) — Readability & Flow**

- [ ] Sentence transition smoothness (cosine similarity between adjacent sentences)
- [ ] Repetition penalty: unique bigram ratio
- [ ] Readability: Flesch Reading Ease (adapted for business context)
- [ ] Flow score: sentence length variation coefficient (ideal: 0.3-0.5)

**Conciseness Score (20%) — Word Efficiency**

- [ ] Word count vs target: exact match→100, ±10%→90, ±20%→70, ±30%→50, ±50%→20, >50%→0
- [ ] Redundancy detection: "in order to"→"to", "at this point in time"→"now"
- [ ] Information density: facts per sentence ratio

**Structure Score (15%) — Email Component Check**

- [ ] Clear subject line present → +10
- [ ] Appropriate greeting present → +10
- [ ] Opening hook quality (engages in first 2 sentences) → +20
- [ ] Body paragraph coherence (one idea per paragraph) → +20
- [ ] Clear call-to-action → +20
- [ ] Professional closing → +10
- [ ] Signature present → +10
- [ ] Total: 100

**Professionalism Score (15%) — Formatting & Etiquette**

- [ ] No inappropriate humor/slang (check against forbidden word list)
- [ ] Proper capitalization (first word of sentence, proper nouns)
- [ ] Consistent formatting (no mixed indentation, fonts)
- [ ] No ALL CAPS (except recognized acronyms)
- [ ] Appropriate bullet point usage (if any)

**Corner Cases:**

- Perfect grammar but terrible structure → SCS catches via Structure score
- Very short email (<50 words) → heavy conciseness penalty
- Very long email (>2x target) → heavy conciseness penalty
- Missing subject line → structure score hit
- Run-on sentences → fluency + grammar penalty
- language-tool-python offline → graceful degradation with regex-based fallback

**Checklist:**

- [ ] `StructuralCoherenceScorer` class with 5 components
- [ ] Grammar checking via LanguageTool API
- [ ] Fluency scoring (readability, transitions, repetition)
- [ ] Conciseness scoring (word count vs target, redundancy)
- [ ] Structure scoring (all 7 email components)
- [ ] Professionalism scoring (slang, caps, formatting)
- [ ] Weighted aggregation to final SCS score
- [ ] LanguageTool offline fallback
- [ ] Returns: `{score, grammar, fluency, conciseness, structure, professionalism, word_count, grammar_issues[:5]}`

---

### Task 6.5 — Evaluation Runner (`src/evaluation/runner.py`)

**Checklist:**

- [ ] `EvaluationRunner` class
- [ ] `async def evaluate_single(email: str, request: EmailRequest) -> Dict`:
  - Run all 3 metrics in parallel via `asyncio.gather()`
  - Calculate `overall_score = weighted_average(FRS, TFI, SCS)`
  - Return combined result with all breakdowns
- [ ] `async def batch_evaluate(scenarios: List[Dict], model: str = "gemini") -> BatchResult`:
  - Pre-check quota before starting (N scenarios × 1 generation each)
  - For each scenario: generate email → evaluate → collect results
  - Rate-limit-aware scheduling (stagger requests to respect RPM)
  - Calculate aggregate statistics (mean, median, std, min, max per metric)
  - Handle partial failures: if a scenario fails, log error, continue
- [ ] Cache scores to avoid re-evaluation of same email-metric pair

---

### Task 6.6 — Report Generator (`src/evaluation/reporter.py`)

**Checklist:**

- [ ] `EvaluationReporter` class
- [ ] `generate_json_report(results, output_path)` — full detailed JSON
- [ ] `generate_csv_report(results, output_path)` — row per scenario, columns: scenario_id, FRS, TFI, SCS, overall, model, latency
- [ ] `generate_markdown_summary(results, output_path)` — summary stats table + per-scenario breakdown
- [ ] All reports saved to `reports/` with timestamp: `evaluation_2026-06-19_143022.json`

---

## PHASE 7: MODEL COMPARISON FRAMEWORK ⏱️ 3–4 hrs

> **Goal:** Build A/B comparison pipeline and generate actionable comparative analysis.

### Task 7.1 — Model Comparison Logic (`src/evaluation/comparison.py`)

**Checklist:**

- [ ] `ModelComparison` class with `async def run_comparison(model_a, model_b, scenarios) -> CompareResponse`
- [ ] Pre-check quota for BOTH providers before starting
- [ ] For each scenario:
  1. Generate email with Model A → evaluate
  2. Generate email with Model B → evaluate
  3. Record per-scenario winner (higher overall score)
  4. Track: latency difference, hallucination rate, token usage
- [ ] Aggregate statistics per model
- [ ] Winner determination: which model wins more scenarios + average margin
- [ ] Failure pattern analysis:
  - "Model A fails on urgent tone" → check per-tone averages
  - "Model B hallucinates dates" → check FRS per-fact breakdowns
- [ ] Handle: partial failures (one model fails a scenario → note but continue)

**Analysis Questions to Answer:**

1. Which model performed better overall? (aggregate FRS, TFI, SCS)
2. Biggest failure mode of lower performer? (identify pattern in lowest scores)
3. Production recommendation? (justified with data)

---

### Task 7.2 — Comparison Report Generation

**Checklist:**

- [ ] Executive summary (1-2 paragraphs: winner, margin, key insight)
- [ ] Side-by-side score comparison table
- [ ] Per-scenario breakdown (table with all 10 scenarios)
- [ ] Failure mode analysis with examples
- [ ] Production recommendation with cost/benefit analysis
- [ ] Save to `reports/comparison_2026-06-19.json` and `docs/comparison_report.md`

---

## PHASE 8: SCRIPTS & CLI TOOLS ⏱️ 1–2 hrs

### Task 8.1 — Server Runner (`scripts/run_server.py`)

**Checklist:**

- [ ] CLI args via `argparse`: `--host`, `--port`, `--reload`, `--log-level`
- [ ] Import and run uvicorn with app from `src.main`
- [ ] Default: `python scripts/run_server.py` → serves on `localhost:8000`
- [ ] Show startup banner with API docs URL

---

### Task 8.2 — Evaluation Script (`scripts/run_evaluation.py`)

**Checklist:**

- [ ] CLI: `python scripts/run_evaluation.py --model gemini --output reports/eval_gemini.json`
- [ ] Arguments: `--model` (gemini/groq), `--output` (path), `--scenarios` (all or comma-separated IDs), `--verbose`
- [ ] Load test scenarios from `data/test_scenarios.json`
- [ ] Pre-check quota
- [ ] Run batch evaluation
- [ ] Generate JSON + CSV reports
- [ ] Print colored summary to console
- [ ] Exit code: 0 success, 1 partial failure, 2 fatal error

---

### Task 8.3 — Comparison Script (`scripts/run_comparison.py`)

**Checklist:**

- [ ] CLI: `python scripts/run_comparison.py --model-a gemini --model-b groq --output reports/comparison.json`
- [ ] Arguments: `--model-a`, `--model-b`, `--output`, `--scenarios`
- [ ] Load test scenarios
- [ ] Pre-check quota for both providers
- [ ] Run A/B comparison
- [ ] Generate comprehensive report
- [ ] Print winner + recommendation to console

---

## PHASE 9: TESTING & QUALITY ASSURANCE ⏱️ 3–4 hrs

> **Goal:** Achieve 80%+ code coverage with unit, integration, and end-to-end tests.

### Task 9.1 — Test Configuration (`tests/conftest.py`)

**Checklist:**

- [ ] Pytest fixtures:
  - `email_request_fixture` — valid EmailRequest for each tone
  - `mock_llm_provider` — returns canned responses
  - `test_scenarios` — load from JSON
  - `sample_generated_email` — realistic generated email
  - `human_reference_email` — load from Markdown file

---

### Task 9.2 — API Tests (`tests/test_api.py`)

**Checklist:**

- [ ] `test_health_endpoint` — GET /health returns 200 + correct structure
- [ ] `test_generate_valid_request` — POST /generate with valid data → 200 + email
- [ ] `test_generate_invalid_intent` — vague intent → 422
- [ ] `test_generate_invalid_tone` — bad tone → 422
- [ ] `test_generate_empty_facts` — empty facts → 422
- [ ] `test_evaluate_endpoint` — POST /evaluate → 200 + scores
- [ ] `test_batch_evaluate` — POST /batch-evaluate → 200 + results
- [ ] `test_compare` — POST /compare → 200 + winner
- [ ] `test_rate_limiting` — hit rate limit → 429
- [ ] `test_cors_headers` — OPTIONS preflight → correct CORS headers

---

### Task 9.3 — Metric Tests (`tests/test_metrics.py`)

**FRS Tests:**

- [ ] `test_frs_all_facts_present` — email with all facts → score 95-100
- [ ] `test_frs_missing_fact` — missing one fact → score deduction
- [ ] `test_frs_hallucination` — extra claims → penalty applied
- [ ] `test_frs_paraphrased_fact` — fact reworded → partial credit (0.6-0.8)
- [ ] `test_frs_natural_integration` — narrative → bonus applied
- [ ] `test_frs_bulleted_facts` — facts as list → no bonus
- [ ] `test_frs_empty_email` — empty email → score 0
- [ ] `test_frs_date_variations` — "June 15" vs "15 June" → full credit

**TFI Tests:**

- [ ] `test_tfi_formal_high_score` — formal email → TFI ≥ 80
- [ ] `test_tfi_casual_high_score` — casual email → TFI ≥ 80
- [ ] `test_tfi_tone_mismatch` — casual with formal writing → low TFI
- [ ] `test_tfi_mixed_tone` — starts formal, ends casual → penalty
- [ ] `test_tfi_urgent_markers` — urgent email → high directness score
- [ ] `test_tfi_empathetic_markers` — empathetic email → high emotional score

**SCS Tests:**

- [ ] `test_scs_perfect_email` — well-structured email → SCS ≥ 90
- [ ] `test_scs_missing_subject` — no subject line → structure penalty
- [ ] `test_scs_missing_greeting` — no greeting → structure penalty
- [ ] `test_scs_grammar_errors` — multiple errors → grammar penalty
- [ ] `test_scs_too_long` — 2x target word count → conciseness penalty
- [ ] `test_scs_too_short` — <50 words → heavy penalty
- [ ] `test_scs_no_signature` — missing closing → structure penalty

---

### Task 9.4 — LangGraph Flow Tests (`tests/test_graph.py`)

**Checklist:**

- [ ] `test_graph_happy_path` — full flow → successful email
- [ ] `test_graph_retry_on_quality_fail` — quality gate routes to retry
- [ ] `test_graph_max_retries` — stops after 2 failed retries
- [ ] `test_graph_fallback_provider` — switches provider on rate limit
- [ ] `test_graph_error_handler` — all providers fail → error path
- [ ] `test_state_persistence` — state preserved across all nodes

---

### Task 9.5 — Integration Tests (`tests/test_integration.py`)

**Checklist:**

- [ ] Mark all tests with `@pytest.mark.integration` (requires API keys)
- [ ] `test_end_to_end_gemini` — generate + evaluate with real Gemini
- [ ] `test_end_to_end_groq` — generate + evaluate with real Groq
- [ ] `test_batch_evaluation_small` — run 2 scenarios (save quota)
- [ ] `test_comparison_pipeline` — A/B comparison (2 scenarios)
- [ ] `test_rate_limit_handling` — rapid requests trigger backoff

---

### Task 9.6 — Run Test Suite & Fix Issues

**Checklist:**

- [ ] `pytest tests/ -v --cov=src` → all pass, coverage ≥ 80%
- [ ] `pytest tests/test_integration.py -v --run-integration` → integration pass
- [ ] Fix any failures
- [ ] `coverage html` → generate coverage report
- [ ] Verify no warnings or deprecation notices

---

## PHASE 10: DOCUMENTATION & FINAL DELIVERABLES ⏱️ 2–3 hrs

### Task 10.1 — Comprehensive README (`README.md`)

**Sections:**

- [ ] Project overview (2-3 sentences)
- [ ] Tech stack summary with version badges
- [ ] Architecture diagram (Mermaid)
- [ ] Quick start: setup, env, run
- [ ] API documentation with curl examples
- [ ] Evaluation commands
- [ ] Project structure map
- [ ] Key features list
- [ ] Troubleshooting FAQ

---

### Task 10.2 — Technical Documentation (`docs/`)

**Checklist:**

- [ ] `docs/architecture.md`:
  - High-level architecture diagram (Mermaid sequence diagram)
  - Data flow diagram (Mermaid)
  - Component responsibilities
  - Key design decisions (§15 from requirements)
- [ ] `docs/prompt_template.md`:
  - Complete prompt template with annotations
  - Few-shot example strategy explanation
  - Tone profile definitions table
  - Chain-of-thought rationale
- [ ] `docs/metrics_spec.md`:
  - FRS: formula, scoring buckets, implementation pipeline, corner cases
  - TFI: 4 dimensions, scoring formula, tone profiles table
  - SCS: 5 components, scoring rubrics, all corner cases
- [ ] `docs/api_reference.md`:
  - All 5 endpoints with request/response schemas
  - Error codes table
  - Rate limit headers
  - Example curl commands

---

### Task 10.3 — Final Comparison Report (`docs/comparison_report.md`)

**Checklist:**

- [ ] Executive summary (paragraph: winner, margin, recommendation)
- [ ] Aggregate scores comparison table
- [ ] Per-scenario breakdown (table with all 10)
- [ ] Per-metric analysis (FRS comparison, TFI comparison, SCS comparison)
- [ ] Failure mode analysis with examples
- [ ] Hallucination rate comparison
- [ ] Speed/latency comparison
- [ ] Production recommendation with justification
- [ ] Cost/benefit analysis (free tier usage)

---

### Task 10.4 — Final Validation & Handoff

**Checklist:**

- [ ] Full evaluation pipeline run successfully for both models
- [ ] Comparison report generated with clear winner
- [ ] All 5 API endpoints responding correctly
- [ ] Swagger docs at `/docs` working
- [ ] All unit tests pass (coverage ≥ 80%)
- [ ] Integration tests pass (with API keys)
- [ ] Code formatted with `black` and `isort`
- [ ] No API keys committed to git (verify `.env` is in `.gitignore`)
- [ ] No debug/test code remaining
- [ ] All functions have docstrings and type hints
- [ ] Deliverables checklist (§13 from requirements) fully verified

---

## QUICK REFERENCE — ALL COMMANDS

```bash
# ─── SETUP (with uv) ───
uv venv .venv
.venv\Scripts\activate                    # Windows
uv pip install -r requirements.txt        # or: uv sync (with pyproject.toml)
uv run python -m spacy download en_core_web_md
uv run python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"

# ─── RUN SERVER ───
python scripts/run_server.py --reload --port 8000
# API Docs: http://localhost:8000/docs

# ─── RUN EVALUATION ───
python scripts/run_evaluation.py --model gemini --output reports/eval_gemini.json
python scripts/run_evaluation.py --model groq --output reports/eval_groq.json

# ─── RUN COMPARISON ───
python scripts/run_comparison.py --model-a gemini --model-b groq --output reports/comparison.json

# ─── RUN TESTS ───
pytest tests/ -v --cov=src
pytest tests/test_integration.py -v --run-integration
pytest tests/test_metrics.py::test_frs_all_facts_present -v  # Single test
coverage html  # Open htmlcov/index.html

# ─── CODE QUALITY ───
black src/ tests/
isort src/ tests/
```

---

## DELIVERABLES FINAL CHECKLIST

### Code Repository

- [ ] Complete source code with modular structure (§9)
- [ ] `README.md` with setup instructions
- [ ] `requirements.txt` and `pyproject.toml`
- [ ] `.env.example` (no real keys)
- [ ] 10 test scenarios in `data/test_scenarios.json`
- [ ] 10 human reference emails in `data/human_references/`
- [ ] 8 few-shot examples in `src/prompts/examples/`
- [ ] 6 tone profiles in `src/prompts/tone_profiles/`
- [ ] Working FastAPI service with 5 endpoints
- [ ] LangGraph state machine (5 nodes + conditional edges)
- [ ] Advanced prompt template (Jinja2 + CoT + few-shot)
- [ ] 3 custom metrics fully implemented (FRS, TFI, SCS)
- [ ] Batch evaluation script
- [ ] Model comparison script
- [ ] Unit and integration tests (coverage ≥ 80%)
- [ ] Error handling for 20+ corner cases (§7)

### Documentation

- [ ] Architecture documentation (`docs/architecture.md`)
- [ ] Prompt template documentation (`docs/prompt_template.md`)
- [ ] Metrics specification (`docs/metrics_spec.md`)
- [ ] API reference (`docs/api_reference.md`)
- [ ] Final comparison report (`docs/comparison_report.md`)
- [ ] Raw evaluation data (JSON in `reports/`)

### Final Report Must Include (§13)

- [ ] Prompt Template (documented with rationale)
- [ ] 3 Custom Metric Definitions + Logic
- [ ] Raw Evaluation Data (CSV/JSON)
- [ ] Comparative Analysis:
  - [ ] Which model performed better
  - [ ] Biggest failure mode of lower performer
  - [ ] Production recommendation with justification

---

## RISK REGISTER

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Free tier quota exhausted mid-batch | High | High | Pre-check quota; stagger requests; queue for next day |
| API rate limit (429) during evaluation | High | Medium | Token bucket + exponential backoff + failover |
| `language-tool-python` API unavailable | Medium | Medium | Graceful degradation with regex fallback |
| Gemini safety filter blocks email | Low | Medium | Relax safety settings; retry with modified prompt |
| sentence-transformers model download fails | Low | Medium | Fallback to simpler TF-IDF similarity |
| Asyncio event loop blocking (sync calls) | Medium | High | All LLM/metric calls must be async |
| Groq changes free tier limits | Low | High | Monitor health endpoint; alert on quota changes |
| PII leakage in logs | Low | High | Sanitize all log outputs; don't persist emails |

---

> **Plan Version:** 1.0 | **Last Updated:** 2026-06-19  
> **Total:** ~100 sub-tasks across 11 phases | **Estimated:** 22–31 hours  
> **Status:** ✅ Ready for Execution — Start with Phase 0
