# Email Generation Assistant — Task Execution Plan

**Project:** Email Generation Assistant with Custom Evaluation Metrics & Model Comparison  
**Duration:** 22-31 hours (estimated)  
**Status:** Ready for Execution  
**Last Updated:** 2026-06-19

---

## Executive Overview

This document breaks down the entire project into actionable tasks organized by execution phase. Each task includes:
- Clear acceptance criteria
- Estimated effort (hours)
- Dependencies
- Implementation notes

---

## Phase 1: Project Setup & Infrastructure (2-3 hours)

### Task 1.1: Initialize Project Repository
**Status:** NOT STARTED  
**Effort:** 0.5 hours  
**Dependencies:** None

**Description:**
Create project directory structure and initialize version control.

**Steps:**
1. Create main project folder: `email-generation-assistant/`
2. Initialize git repository: `git init`
3. Create `.gitignore` (Python standard)
4. Create `LICENSE` file (MIT or Apache 2.0)

**Acceptance Criteria:**
- [ ] Git repository initialized
- [ ] `.gitignore` includes Python standard patterns
- [ ] Folder structure ready for code

---

### Task 1.2: Setup Python Environment & Dependencies
**Status:** NOT STARTED  
**Effort:** 1 hour  
**Dependencies:** Task 1.1

**Description:**
Create virtual environment and install all required Python packages.

**Steps:**
1. Create virtual environment: `python -m venv venv`
2. Activate venv
3. Create `requirements.txt` with all dependencies listed in project_requirement_analysis.md
4. Create `pyproject.toml` for modern Python packaging
5. Install dependencies: `pip install -r requirements.txt`
6. Download spaCy models: `python -m spacy download en_core_web_md`
7. Download NLTK datasets: `nltk.download('punkt')`, `nltk.download('stopwords')`
8. Verify all packages installed: `pip list`

**Acceptance Criteria:**
- [ ] Virtual environment created and activated
- [ ] All 25+ packages installed successfully
- [ ] spaCy and NLTK models downloaded
- [ ] `pip list` shows no conflicts or warnings
- [ ] `python -c "import fastapi, langchain, langgraph"` runs without errors

**Key Dependencies:** fastapi, uvicorn, pydantic, langchain, langgraph, sentence-transformers, google-generativeai, groq

---

### Task 1.3: Create Configuration System (.env & config.py)
**Status:** NOT STARTED  
**Effort:** 0.75 hours  
**Dependencies:** Task 1.2

**Description:**
Setup environment variable management and Pydantic Settings configuration.

**Steps:**
1. Create `.env.example` file with template variables:
   - `GEMINI_API_KEY=your_key_here`
   - `GROQ_API_KEY=your_key_here`
   - `ENVIRONMENT=development`
   - `LOG_LEVEL=INFO`
   - `PORT=8000`
2. Create `src/config.py` using Pydantic v2 Settings
3. Implement environment variable validation
4. Create local `.env` file from `.env.example` (add real API keys)

**Acceptance Criteria:**
- [ ] `.env.example` created with all required variables
- [ ] `.env` created locally (NOT committed to git)
- [ ] `config.py` loads all variables with validation
- [ ] Type hints for all config values
- [ ] Default values for optional configs

---

### Task 1.4: Create Project Structure & Core Directories
**Status:** NOT STARTED  
**Effort:** 0.5 hours  
**Dependencies:** Task 1.1

**Description:**
Create all required directory structure as specified in project_requirement_analysis.md section 9.

**Steps:**
1. Create all subdirectories:
   ```
   src/api/
   src/core/
   src/prompts/
   src/prompts/templates/
   src/prompts/examples/
   src/prompts/tone_profiles/
   src/llm/providers/
   src/evaluation/metrics/
   src/utils/
   data/test_scenarios/
   data/human_references/
   data/few_shot_examples/
   tests/
   reports/
   scripts/
   docs/
   ```
2. Create `__init__.py` files in all Python packages
3. Create `.gitkeep` in empty directories (reports, data/)

**Acceptance Criteria:**
- [ ] All directories exist
- [ ] All Python packages have `__init__.py`
- [ ] Directory tree matches specification
- [ ] Empty directories have `.gitkeep`

---

### Task 1.5: Setup Logging & Monitoring Infrastructure
**Status:** NOT STARTED  
**Effort:** 0.75 hours  
**Dependencies:** Task 1.3

**Description:**
Implement structured logging using structlog for production-ready logging.

**Steps:**
1. Create `src/utils/logging_config.py`
2. Setup structlog with JSON formatting
3. Configure log levels per environment
4. Create request ID tracing system
5. Setup file and console log handlers

**Acceptance Criteria:**
- [ ] Structured logging configured with JSON output
- [ ] Log level configurable via environment
- [ ] Request tracing ID generation implemented
- [ ] Test log output format is valid JSON

---

## Phase 2: Core Email Generation System (4-6 hours)

### Task 2.1: Create Pydantic Models for API
**Status:** NOT STARTED  
**Effort:** 1 hour  
**Dependencies:** Task 1.3

**Description:**
Define strict input/output validation models using Pydantic v2.

**Steps:**
1. Create `src/api/models.py` with:
   - `ToneEnum` (formal, casual, urgent, empathetic, persuasive, apologetic)
   - `EmailRequest` model with field validation
   - `EmailResponse` model with metadata
   - `ErrorResponse` model
   - Custom validators for intent specificity, fact validation
2. Implement field validators:
   - Intent minimum 10 chars, maximum 500
   - Facts list min 1, max 10 items
   - Each fact min 5 chars
   - Reject vague intents ("write", "email", "compose")
3. Add docstrings and examples for each model

**Acceptance Criteria:**
- [ ] All request/response models defined
- [ ] Pydantic v2 field validators implemented
- [ ] Corner case validation working (vague intents rejected, etc.)
- [ ] Models have proper docstrings and JSON schema examples

---

### Task 2.2: Implement LangGraph State & Base Logic
**Status:** NOT STARTED  
**Effort:** 1.5 hours  
**Dependencies:** Task 2.1, Task 1.3

**Description:**
Create LangGraph state machine for email generation workflow.

**Steps:**
1. Create `src/core/state.py`:
   - Define `EmailGenerationState` TypedDict
   - Include: request, prompt, raw_output, cleaned_email, metadata, retry_count, error, scores
2. Create `src/core/graph.py`:
   - Implement graph nodes (validate_input, construct_prompt, call_llm, post_process, quality_check)
   - Define conditional edges and routing logic
   - Compile graph with retry logic

**Acceptance Criteria:**
- [ ] State TypedDict defined with all required fields
- [ ] All 5 graph nodes implement correct logic
- [ ] Conditional edges route correctly
- [ ] Graph compiles without errors
- [ ] Retry logic limits to 2 retries

---

### Task 2.3: Create LLM Provider Abstraction
**Status:** NOT STARTED  
**Effort:** 2 hours  
**Dependencies:** Task 1.2, Task 1.3

**Description:**
Implement abstract provider interface and concrete implementations for Gemini and Groq.

**Steps:**
1. Create `src/llm/providers/base.py`:
   - Define abstract `LLMProvider` class
   - Methods: `generate()`, `get_quota()`, `check_health()`
2. Create `src/llm/providers/gemini_provider.py`:
   - Implement Google Gemini 2.5 Flash integration
   - Use `google-generativeai` SDK
   - Handle rate limiting (10 RPM, 1,500 req/day)
   - Async support
3. Create `src/llm/providers/groq_provider.py`:
   - Implement Groq Llama 3.3 70B integration
   - Use OpenAI-compatible client
   - Handle rate limiting (30 RPM, 1,000 req/day)
   - Async support
4. Create `src/llm/providers/router.py`:
   - Implement failover logic (Gemini → Groq → OpenRouter)
   - Select provider based on availability and quota

**Acceptance Criteria:**
- [ ] Base provider class with abstract methods
- [ ] Gemini provider fully implemented with API integration
- [ ] Groq provider fully implemented with API integration
- [ ] Failover router selects correct provider
- [ ] Both providers tested with real API calls
- [ ] Error handling for rate limits, timeouts, invalid keys

---

### Task 2.4: Implement Rate Limiting & Quota Management
**Status:** NOT STARTED  
**Effort:** 1 hour  
**Dependencies:** Task 2.3

**Description:**
Implement token bucket algorithm and quota tracking.

**Steps:**
1. Create `src/llm/rate_limiter.py`:
   - Implement token bucket per provider
   - Track quota remaining
   - Exponential backoff with jitter
   - Pre-flight quota check before batch jobs
2. Integrate rate limiter with provider router
3. Add quota status to health endpoint

**Acceptance Criteria:**
- [ ] Token bucket algorithm implemented
- [ ] Rate limiter prevents 429 errors
- [ ] Quota pre-check works before batch evaluation
- [ ] Exponential backoff implemented with jitter
- [ ] Rate limiter is async-safe

---

### Task 2.5: Create Prompt Engineering System
**Status:** NOT STARTED  
**Effort:** 2 hours  
**Dependencies:** Task 1.4, Task 2.1

**Description:**
Implement advanced prompt templates with few-shot examples and Chain-of-Thought.

**Steps:**
1. Create `src/prompts/templates/`:
   - `system_role.j2` — Expert executive coach persona
   - `few_shot.j2` — Few-shot example injection
   - `cot_instructions.j2` — Chain-of-Thought guidance
   - `base.j2` — Complete prompt assembly
2. Create `src/prompts/registry.py`:
   - Load and cache few-shot examples
   - Retrieve examples by intent similarity
   - Support for 8 example scenarios
3. Create `data/few_shot_examples/`:
   - Example 1: Formal / Follow-up
   - Example 2: Casual / Announcement
   - Example 3: Urgent / Escalation
   - Example 4: Empathetic / Request
   - Example 5: Apologetic / Notification
   - Example 6: Persuasive / Proposal
   - Example 7: Formal / Status Update
   - Example 8: Casual / Team Update
4. Create `src/prompts/tone_profiles/`:
   - `formal.yaml` — Tone markers and guidance
   - `casual.yaml`
   - `urgent.yaml`
   - `empathetic.yaml`
   - `persuasive.yaml`
   - `apologetic.yaml`

**Acceptance Criteria:**
- [ ] All Jinja2 templates created
- [ ] Few-shot examples loaded correctly
- [ ] Examples retrieved by similarity
- [ ] Tone profiles loaded from YAML
- [ ] Full prompt assembly produces valid output
- [ ] CoT instructions properly formatted
- [ ] System prompt establishes expertise context

---

### Task 2.6: Implement Email Generation Orchestration
**Status:** NOT STARTED  
**Effort:** 2 hours  
**Dependencies:** Task 2.2, Task 2.4, Task 2.5

**Description:**
Implement the main email generation logic that ties together LangGraph, prompts, and LLM providers.

**Steps:**
1. Create `src/core/email_generator.py`:
   - Implement graph node functions (validate_input, construct_prompt, call_llm, post_process, quality_check)
   - Handle input validation and enrichment
   - Assemble dynamic prompts with few-shot examples
   - Execute LLM calls with rate limit handling and fallback
   - Post-process output (remove thinking tags, validate formatting)
   - Implement quality gate with auto-retry logic
2. Add error handling for all failure modes
3. Add metadata tracking (model, tokens, latency)

**Acceptance Criteria:**
- [ ] All graph nodes implemented with correct logic
- [ ] Email generation end-to-end works
- [ ] Retry logic triggers on quality failures
- [ ] Metadata tracked and returned
- [ ] Error messages are descriptive
- [ ] Tests show successful generation with both models

---

## Phase 3: Evaluation Metrics System (6-8 hours)

### Task 3.1: Implement Fact Recall Score (FRS) Metric
**Status:** NOT STARTED  
**Effort:** 2.5 hours  
**Dependencies:** Task 1.2, Task 2.6

**Description:**
Implement the Fact Recall Score metric that measures fact completeness and accuracy.

**Steps:**
1. Create `src/evaluation/metrics/fact_recall.py`:
   - Implement `FactRecallScorer` class
   - Setup sentence-transformers embedder for semantic matching
   - Extract fact-like statements from email (NER + pattern matching)
   - Calculate semantic similarity for each input fact
   - Score each fact: exact match (1.0), paraphrased (0.7), partial (0.4), missing (0.0)
   - LLM-as-Judge for hallucination detection
   - Natural integration check (bonus/penalty)
   - Return detailed breakdown
2. Implement corner case handling:
   - Date format variations
   - Fact splitting across sentences
   - Subject line fact mentions
   - Numerical fact with unit conversions
   - Implied vs explicit facts

**Acceptance Criteria:**
- [ ] FRS metric calculates 0-100 score
- [ ] Semantic matching works correctly
- [ ] Hallucination detection implemented
- [ ] Natural integration scoring works
- [ ] Corner cases handled properly
- [ ] Returns detailed per-fact breakdown
- [ ] Tests show correct scoring on sample emails

---

### Task 3.2: Implement Tone Fidelity Index (TFI) Metric
**Status:** NOT STARTED  
**Effort:** 2.5 hours  
**Dependencies:** Task 1.2, Task 2.6

**Description:**
Implement the Tone Fidelity Index metric that measures tone matching accuracy.

**Steps:**
1. Create `src/evaluation/metrics/tone_fidelity.py`:
   - Implement `ToneFidelityScorer` class
   - Load tone profiles (YAML files from prompts/tone_profiles/)
   - Extract linguistic features:
     - Lexical: contraction ratio, modal verbs, politeness markers, jargon density
     - Syntactic: sentence length, subordination, imperatives
     - Pragmatic: directness, hedging, question ratio
     - Emotional: sentiment words, intensity modifiers
   - Calculate dimension scores (0-100)
   - Weighted average: 0.3 * Lexical + 0.25 * Syntactic + 0.25 * Pragmatic + 0.2 * Emotional
   - Detect detected tone vs target tone
   - Return dimension breakdown
2. Handle corner cases:
   - Mixed tone (penalize inconsistency)
   - Tone drift in long emails
   - Sarcasm or irony detection
   - Tone balance in negative contexts

**Acceptance Criteria:**
- [ ] TFI metric calculates 0-100 score
- [ ] Linguistic features extracted correctly
- [ ] Tone profiles loaded from YAML
- [ ] Weighted dimension averaging works
- [ ] Tone classification reasonably accurate
- [ ] Corner cases handled (tone drift, mixed tone)
- [ ] Tests show correct scoring on sample emails

---

### Task 3.3: Implement Structural Coherence Score (SCS) Metric
**Status:** NOT STARTED  
**Effort:** 2.5 hours  
**Dependencies:** Task 1.2, Task 2.6

**Description:**
Implement the Structural Coherence Score metric combining grammar, fluency, conciseness, structure, and professionalism.

**Steps:**
1. Create `src/evaluation/metrics/structural_coherence.py`:
   - Implement `StructuralCoherenceScorer` class
   - Grammar checking: language-tool or LanguageTool API
     - Spelling, subject-verb agreement, pronouns, punctuation
   - Fluency: textstat, transition smoothness, repetition, readability
   - Conciseness: word count vs target, redundancy detection, information density
   - Structure: subject line, greeting, opening hook, body coherence, CTA, closing, signature
   - Professionalism: slang check, capitalization, formatting, ALL CAPS
   - Weighted: 0.25 * Grammar + 0.25 * Fluency + 0.20 * Conciseness + 0.15 * Structure + 0.15 * Professionalism
2. Implement LanguageTool integration with error counting
3. Handle corner cases:
   - Missing subject line or signature
   - Run-on sentences
   - Very short or very long emails
   - Inconsistent formatting

**Acceptance Criteria:**
- [ ] SCS metric calculates 0-100 score
- [ ] Grammar checking integrated and working
- [ ] Fluency scoring implemented
- [ ] Conciseness scoring based on target word count
- [ ] Structure scoring verifies email components
- [ ] Professionalism checks working
- [ ] Weighted averaging correct
- [ ] Returns detailed breakdown of all dimensions
- [ ] Tests show correct scoring on sample emails

---

### Task 3.4: Create Metric Base Class & Interface
**Status:** NOT STARTED  
**Effort:** 0.75 hours  
**Dependencies:** Task 1.4

**Description:**
Create abstract base class for metrics to ensure consistency.

**Steps:**
1. Create `src/evaluation/metrics/base.py`:
   - Define abstract `Metric` class
   - Methods: `score()`, `validate()`, `get_details()`
   - Return format: `{"score": float, "breakdown": dict, "flags": list}`
2. Ensure all three metrics inherit from base class

**Acceptance Criteria:**
- [ ] Abstract Metric class defined
- [ ] All three metrics inherit correctly
- [ ] Score method signature consistent
- [ ] Return format standardized

---

### Task 3.5: Implement Batch Evaluation Runner
**Status:** NOT STARTED  
**Effort:** 1.5 hours  
**Dependencies:** Task 3.1, Task 3.2, Task 3.3, Task 2.6

**Description:**
Implement orchestration for running all three metrics on generated emails.

**Steps:**
1. Create `src/evaluation/runner.py`:
   - `EvaluationRunner` class
   - `evaluate_single()` — run all 3 metrics on one email
   - `evaluate_batch()` — run evaluation on all 10 test scenarios
   - Rate limit awareness (pre-check quota before starting batch)
   - Parallel metric execution where possible
   - Aggregate results and statistics
   - Error handling and partial failure recovery
2. Implement caching of scores to avoid re-evaluation

**Acceptance Criteria:**
- [ ] Single email evaluation works (all 3 metrics)
- [ ] Batch evaluation processes all 10 scenarios
- [ ] Quota pre-check prevents exhaustion
- [ ] Aggregate statistics calculated correctly
- [ ] Parallel execution improves performance
- [ ] Caching prevents duplicate evaluations

---

### Task 3.6: Implement Results Reporter
**Status:** NOT STARTED  
**Effort:** 1.5 hours  
**Dependencies:** Task 3.5

**Description:**
Create report generation for evaluation results in JSON, CSV, and Markdown formats.

**Steps:**
1. Create `src/evaluation/reporter.py`:
   - `EvaluationReporter` class
   - `generate_json_report()` — detailed JSON with all scores and breakdowns
   - `generate_csv_report()` — tabular format for spreadsheet analysis
   - `generate_markdown_report()` — human-readable summary with tables
   - Summary statistics: average, min, max, std dev per metric
   - Failure analysis: identify patterns in low scores
2. Save reports to `reports/` directory with timestamp

**Acceptance Criteria:**
- [ ] JSON report generated with all details
- [ ] CSV report with scenarios and scores
- [ ] Markdown report with formatted tables
- [ ] Summary statistics calculated
- [ ] Reports saved with timestamps
- [ ] Reports are readable and parseable

---

## Phase 4: Model Comparison Framework (3-4 hours)

### Task 4.1: Implement Model Comparison Logic
**Status:** NOT STARTED  
**Effort:** 2 hours  
**Dependencies:** Task 2.6, Task 3.5

**Description:**
Create A/B comparison framework for evaluating Model A (Gemini) vs Model B (Groq).

**Steps:**
1. Create `src/evaluation/comparison.py`:
   - `ModelComparison` class
   - Run email generation with both models for all test scenarios
   - Run evaluation on both sets
   - Calculate winner per scenario (overall score)
   - Calculate aggregate statistics
   - Identify failure patterns for each model
2. Comparison analysis:
   - Head-to-head per metric (FRS, TFI, SCS)
   - Win rate percentage
   - Biggest difference between models
   - Hallucination rate analysis
3. Error handling:
   - Continue if one model fails a scenario
   - Partial result handling
   - Quota management for both providers

**Acceptance Criteria:**
- [ ] Both models generate emails for all 10 scenarios
- [ ] Both sets evaluated on all 3 metrics
- [ ] Aggregate statistics calculated
- [ ] Winner determination logic working
- [ ] Failure pattern analysis implemented
- [ ] Handles partial results gracefully

---

### Task 4.2: Create Comparison Reporter
**Status:** NOT STARTED  
**Effort:** 1.5 hours  
**Dependencies:** Task 4.1, Task 3.6

**Description:**
Generate detailed comparison reports with actionable insights.

**Steps:**
1. Create comparison report generator:
   - Side-by-side score comparison
   - Per-metric performance tables
   - Failure mode analysis (which model failed on what)
   - Hallucination rate comparison
   - Speed/latency comparison
   - Recommendation with justification
2. Report sections:
   - Executive summary (1 paragraph)
   - Detailed metrics comparison
   - Per-scenario breakdown
   - Failure analysis
   - Production recommendation
   - Cost/benefit analysis

**Acceptance Criteria:**
- [ ] Comparison report generated
- [ ] All metrics compared side-by-side
- [ ] Failure patterns analyzed
- [ ] Recommendation includes justification
- [ ] Report format is professional and readable

---

## Phase 5: Test Data & Evaluation Preparation (2-3 hours)

### Task 5.1: Create 10 Test Scenarios
**Status:** NOT STARTED  
**Effort:** 1.5 hours  
**Dependencies:** Task 1.4

**Description:**
Create 10 diverse test scenarios covering all tones and complexity levels.

**Steps:**
1. Create `data/test_scenarios.json`:
   - Scenario 1: Follow-up after product demo (Formal, Medium)
   - Scenario 2: Request deadline extension (Empathetic, Medium)
   - Scenario 3: Office relocation announcement (Casual, Low)
   - Scenario 4: Security incident escalation (Urgent, High)
   - Scenario 5: Promotion congratulations (Casual, Low)
   - Scenario 6: Vendor proposal request (Formal, Medium)
   - Scenario 7: Apology for late payment (Apologetic, Medium)
   - Scenario 8: New remote work policy (Formal, High)
   - Scenario 9: Persuade tool purchase approval (Persuasive, High)
   - Scenario 10: Condolences on colleague loss (Empathetic, High)
2. Each scenario includes:
   - Scenario ID
   - Intent (clear, specific, 10-50 words)
   - Key facts (array of 4-6 facts)
   - Tone (one of 6 valid tones)
   - Complexity level (Low/Medium/High)
   - Target word count
   - Human reference length (for comparison)

**Acceptance Criteria:**
- [ ] All 10 scenarios defined in JSON format
- [ ] All 6 tones covered at least once
- [ ] Complexity levels balanced
- [ ] Facts are specific and substantial
- [ ] Scenarios are realistic business scenarios
- [ ] JSON valid and parseable

---

### Task 5.2: Create Human Reference Emails
**Status:** NOT STARTED  
**Effort:** 1.5 hours  
**Dependencies:** Task 5.1

**Description:**
Write gold-standard human reference emails for all 10 scenarios.

**Steps:**
1. For each scenario, write a professional email:
   - Integrate all facts naturally (no lists)
   - Match tone precisely with linguistic markers
   - Keep within target word count (±10%)
   - Include appropriate greeting and closing
   - Professional formatting with subject line
2. Save each reference as:
   - `data/human_references/scenario_01.md`
   - `data/human_references/scenario_02.md`
   - ... (10 total)
3. Format: Markdown with subject line in header

**Acceptance Criteria:**
- [ ] 10 reference emails written by skilled professional
- [ ] All facts integrated naturally per scenario
- [ ] Tone matching verified (formal has no contractions, casual is warm, etc.)
- [ ] Word counts within ±10% of target
- [ ] References saved as Markdown files
- [ ] Quality is consistent and professional

**Reference Writing Tips:**
- Use the prompt template as guide
- Include all facts in first 2-3 paragraphs
- Match tone markers precisely (formal: no contractions, casual: conversational)
- Professional closings appropriate to tone
- Proofread for grammar and clarity

---

## Phase 6: FastAPI Endpoints & API Layer (3-4 hours)

### Task 6.1: Create API Routes
**Status:** NOT STARTED  
**Effort:** 2 hours  
**Dependencies:** Task 2.1, Task 2.6, Task 3.5

**Description:**
Implement FastAPI routes for all endpoints.

**Steps:**
1. Create `src/api/routes.py`:
   - `POST /generate` — Generate single email
   - `POST /evaluate` — Evaluate single email
   - `POST /batch-evaluate` — Run full evaluation on 10 scenarios
   - `POST /compare` — A/B model comparison
   - `GET /health` — Service health + provider status
2. Each endpoint:
   - Request validation with Pydantic models
   - Proper HTTP status codes
   - Error responses with request ID for tracing
   - Rate limit info in response headers
3. Implement dependencies:
   - Authentication (optional bearer token)
   - Rate limiting per client
   - Request ID generation

**Acceptance Criteria:**
- [ ] All 5 endpoints implemented
- [ ] Request validation working
- [ ] Proper HTTP status codes (200, 422, 429, 500)
- [ ] Error responses include request ID
- [ ] Rate limit info in headers
- [ ] Endpoints tested with Swagger docs

---

### Task 6.2: Implement Middleware & Error Handling
**Status:** NOT STARTED  
**Effort:** 1 hour  
**Dependencies:** Task 6.1, Task 1.5

**Description:**
Create custom middleware for error handling, CORS, and request tracing.

**Steps:**
1. Create `src/api/middleware.py`:
   - Request ID middleware (generate UUID for each request)
   - Error handler for validation errors (422)
   - Error handler for rate limits (429)
   - Error handler for LLM failures (500, 503)
   - CORS middleware configuration
   - Logging middleware for all requests
2. Custom exception classes in `src/core/exceptions.py`:
   - RateLimitedException
   - LLMFailedException
   - ValidationException
   - etc.

**Acceptance Criteria:**
- [ ] Request ID generation and propagation working
- [ ] CORS configured for development
- [ ] All error types handled gracefully
- [ ] Errors logged with request ID
- [ ] Response format consistent

---

### Task 6.3: Create FastAPI Application
**Status:** NOT STARTED  
**Effort:** 0.75 hours  
**Dependencies:** Task 6.1, Task 6.2, Task 1.3

**Description:**
Create main FastAPI application and server initialization.

**Steps:**
1. Create `src/main.py`:
   - Initialize FastAPI app
   - Register routers from `src/api/routes.py`
   - Add middleware
   - Setup OpenAPI documentation
   - Add startup/shutdown events
2. Create `scripts/run_server.py`:
   - Run uvicorn with development settings
   - Support --reload, --host, --port arguments
3. Test server starts: `python scripts/run_server.py`

**Acceptance Criteria:**
- [ ] FastAPI app initializes without errors
- [ ] Swagger docs available at `/docs`
- [ ] Server starts and listens on port 8000
- [ ] Endpoints listed in Swagger
- [ ] Health endpoint returns correct structure

---

## Phase 7: Evaluation Execution Scripts (2-3 hours)

### Task 7.1: Create Single Model Evaluation Script
**Status:** NOT STARTED  
**Effort:** 1 hour  
**Dependencies:** Task 3.5, Task 3.6, Task 2.6

**Description:**
Create script to run full evaluation on a single model across all 10 scenarios.

**Steps:**
1. Create `scripts/run_evaluation.py`:
   - CLI arguments: --model (gemini/groq), --output (path)
   - Pre-check quota before starting
   - Run generation and evaluation for all 10 scenarios
   - Generate JSON and CSV reports
   - Print summary statistics to console
2. Error handling:
   - Graceful handling of scenario failures
   - Quota exhaustion detection
   - Partial result reporting
3. Usage: `python scripts/run_evaluation.py --model gemini --output reports/gemini_eval.json`

**Acceptance Criteria:**
- [ ] Script evaluates all 10 scenarios
- [ ] Reports generated in JSON and CSV
- [ ] Summary printed to console
- [ ] Exit code 0 on success, non-zero on failure
- [ ] Handles quota limits gracefully

---

### Task 7.2: Create Comparison Script
**Status:** NOT STARTED  
**Effort:** 1 hour  
**Dependencies:** Task 4.1, Task 4.2

**Description:**
Create script to run A/B comparison between Gemini and Groq.

**Steps:**
1. Create `scripts/run_comparison.py`:
   - CLI arguments: --model-a (provider), --model-b (provider), --output (path)
   - Pre-check quota for both providers
   - Run generation with both models on all scenarios
   - Run evaluation on both sets
   - Generate comparison report
   - Print recommendation to console
2. Output:
   - Comparison JSON with all details
   - Markdown summary report
   - CSV with head-to-head scores
3. Usage: `python scripts/run_comparison.py --model-a gemini --model-b groq --output reports/comparison.json`

**Acceptance Criteria:**
- [ ] Script runs full A/B comparison
- [ ] Comparison report generated
- [ ] Head-to-head metrics compared
- [ ] Failure analysis performed
- [ ] Production recommendation provided
- [ ] Exit code 0 on success

---

### Task 7.3: Create Batch Processing Utility
**Status:** NOT STARTED  
**Effort:** 0.5 hours  
**Dependencies:** Task 2.6

**Description:**
Create utility for reliable batch processing with quota management.

**Steps:**
1. Create `src/utils/batch_processor.py`:
   - Batch execution with quota pre-check
   - Request queuing
   - Error recovery
   - Progress reporting
   - Result aggregation

**Acceptance Criteria:**
- [ ] Batch processing handles 10+ items reliably
- [ ] Quota checked before starting
- [ ] Failed items don't block others
- [ ] Progress reported to stdout/logs

---

## Phase 8: Testing & Quality Assurance (3-4 hours)

### Task 8.1: Create Unit Tests for Models & Validation
**Status:** NOT STARTED  
**Effort:** 1 hour  
**Dependencies:** Task 2.1, Task 1.2

**Description:**
Write unit tests for Pydantic models and input validation.

**Steps:**
1. Create `tests/test_models.py`:
   - Test valid EmailRequest inputs
   - Test validation failures (invalid tone, too few facts, etc.)
   - Test field validators (intent specificity, fact length)
   - Test corner cases (empty strings, special characters)
2. Tests should cover:
   - Valid inputs accepted
   - Invalid inputs rejected with proper messages
   - Edge cases handled
   - All 6 tones recognized

**Acceptance Criteria:**
- [ ] All model tests pass
- [ ] Validation properly rejects invalid inputs
- [ ] Error messages are clear
- [ ] Edge cases covered

---

### Task 8.2: Create Integration Tests for LLM Providers
**Status:** NOT STARTED  
**Effort:** 1 hour  
**Dependencies:** Task 2.3, Task 2.4, Task 1.2

**Description:**
Write integration tests for LLM provider implementations.

**Steps:**
1. Create `tests/test_llm_providers.py`:
   - Test Gemini provider generates valid output
   - Test Groq provider generates valid output
   - Test rate limiting behavior
   - Test fallover logic (if primary fails)
   - Test quota tracking
2. Mark tests as integration tests (requires API keys)
3. Use pytest markers: `@pytest.mark.integration`
4. Tests should verify:
   - LLM returns non-empty response
   - Output is valid email format
   - Metadata tracked (tokens, latency)
   - Rate limit info available

**Acceptance Criteria:**
- [ ] All provider tests pass with real API calls
- [ ] Both models generate output successfully
- [ ] Rate limiting prevents exceeding limits
- [ ] Fallover works when primary fails
- [ ] Can skip integration tests if no API keys

---

### Task 8.3: Create Metric Unit Tests
**Status:** NOT STARTED  
**Effort:** 1.5 hours  
**Dependencies:** Task 3.1, Task 3.2, Task 3.3

**Description:**
Write comprehensive tests for all three custom metrics.

**Steps:**
1. Create `tests/test_metrics.py`:
   - Test Fact Recall Score (FRS)
     - Perfect recall scenario (all facts present)
     - Missing facts scenario
     - Hallucination detection
     - Paraphrased facts recognition
   - Test Tone Fidelity Index (TFI)
     - Formal tone recognition
     - Casual tone recognition
     - Tone mismatch detection
     - Mixed tone detection (penalty)
   - Test Structural Coherence Score (SCS)
     - Grammar error detection
     - Missing subject line penalty
     - Word count penalties (too long/short)
     - Structure completeness check
2. Test data:
   - Use sample emails (both good and bad)
   - Test corner cases
   - Verify score ranges (0-100)

**Acceptance Criteria:**
- [ ] All metric tests pass
- [ ] Metric calculations are correct (verified manually)
- [ ] Score ranges are 0-100
- [ ] Corner cases handled properly
- [ ] Breakdown details are accurate

---

### Task 8.4: Create End-to-End Integration Tests
**Status:** NOT STARTED  
**Effort:** 1 hour  
**Dependencies:** Task 6.3, Task 3.5, Task 2.6

**Description:**
Write end-to-end tests for complete workflows.

**Steps:**
1. Create `tests/test_e2e.py`:
   - Test full email generation workflow
   - Test API endpoints (using TestClient)
   - Test evaluation pipeline
   - Test comparison workflow
2. Test scenarios:
   - Generate email with simple request
   - Generate and evaluate
   - Run batch evaluation on all 10 scenarios
   - Run A/B comparison

**Acceptance Criteria:**
- [ ] Full workflows execute successfully
- [ ] All endpoints return expected responses
- [ ] Evaluation produces valid scores
- [ ] Error cases handled gracefully

---

### Task 8.5: Run Full Test Suite
**Status:** NOT STARTED  
**Effort:** 0.5 hours  
**Dependencies:** Task 8.1, Task 8.2, Task 8.3, Task 8.4

**Description:**
Run complete test suite and achieve minimum coverage.

**Steps:**
1. Run: `pytest tests/ -v --cov=src`
2. Target: 80%+ code coverage
3. Fix any failing tests
4. Generate coverage report: `coverage html`

**Acceptance Criteria:**
- [ ] All unit tests pass
- [ ] All integration tests pass (if API keys available)
- [ ] 80%+ code coverage
- [ ] No warnings in test output
- [ ] Coverage report generated

---

## Phase 9: Documentation & Final Deliverables (2-3 hours)

### Task 9.1: Create Comprehensive README
**Status:** NOT STARTED  
**Effort:** 1 hour  
**Dependencies:** All previous tasks

**Description:**
Create detailed README with setup, usage, and architecture overview.

**Steps:**
1. Create `README.md`:
   - Project overview (2-3 sentences)
   - Architecture diagram
   - Setup instructions:
     - Clone and virtual environment
     - Install dependencies
     - Configure API keys
     - Download NLP models
   - Running the service (development and production)
   - API usage examples (curl or Python)
   - Running evaluations
   - Running tests
   - Troubleshooting common issues
2. Include:
   - Tech stack summary
   - Key features
   - Cost estimates (free tier usage)
   - Project structure overview

**Acceptance Criteria:**
- [ ] README covers all major sections
- [ ] Setup instructions are clear and complete
- [ ] Code examples are working
- [ ] Architecture explained
- [ ] Troubleshooting section included

---

### Task 9.2: Create Technical Documentation
**Status:** NOT STARTED  
**Effort:** 0.75 hours  
**Dependencies:** All previous tasks

**Description:**
Create detailed technical documentation in `docs/` folder.

**Steps:**
1. Create `docs/architecture.md`:
   - System architecture overview
   - Component interactions
   - Data flow diagrams
   - LangGraph state machine visualization
2. Create `docs/metrics_spec.md`:
   - Detailed metric definitions
   - Scoring logic for FRS, TFI, SCS
   - Formula and examples
   - Corner cases addressed
3. Create `docs/prompt_template.md`:
   - Complete prompt template
   - Few-shot examples
   - CoT instructions
   - Tone profiles explanation
4. Create `docs/api_reference.md`:
   - Endpoint specifications
   - Request/response examples
   - Error codes
   - Rate limits

**Acceptance Criteria:**
- [ ] All documentation files created
- [ ] Technical details are accurate
- [ ] Examples are executable
- [ ] Architecture clearly explained

---

### Task 9.3: Generate Final Evaluation Report
**Status:** NOT STARTED  
**Effort:** 1 hour  
**Dependencies:** Task 4.1, Task 4.2, Task 7.2

**Description:**
Execute final comparison and generate comprehensive analysis report.

**Steps:**
1. Run full A/B comparison: `python scripts/run_comparison.py --model-a gemini --model-b groq`
2. Analyze results:
   - Which model performed better overall
   - Which model had better FRS (fact recall)
   - Which model had better TFI (tone fidelity)
   - Which model had better SCS (structural coherence)
   - Biggest failure mode of lower performer
   - Speed/latency comparison
3. Create `docs/final_analysis.md`:
   - Executive summary (1-2 paragraphs)
   - Detailed metrics comparison (tables)
   - Per-scenario results
   - Failure mode analysis
   - Production recommendation with cost/benefit
   - Conclusion

**Acceptance Criteria:**
- [ ] Both models evaluated on all 10 scenarios
- [ ] All metrics compared
- [ ] Failure patterns identified
- [ ] Recommendation justified with data
- [ ] Report is professional and actionable

---

### Task 9.4: Create Project Checklist & Handoff Document
**Status:** NOT STARTED  
**Effort:** 0.5 hours  
**Dependencies:** All previous tasks

**Description:**
Create final checklist and handoff documentation.

**Steps:**
1. Verify all deliverables:
   - [ ] Source code complete and tested
   - [ ] All tests passing (80%+ coverage)
   - [ ] Documentation complete
   - [ ] API running and responding
   - [ ] Evaluation working on test data
   - [ ] Comparison report generated
   - [ ] README includes setup and usage
   - [ ] All code is clean and documented
   - [ ] No API keys in git
   - [ ] Project structure follows spec
2. Create `DEPLOYMENT.md`:
   - Deployment steps
   - Environment setup
   - API key configuration
   - Running in production
   - Monitoring and health checks
   - Scaling considerations
3. Create `MAINTENANCE.md`:
   - How to update prompts
   - How to add new test scenarios
   - How to tune metric weights
   - Troubleshooting guide
   - Performance optimization tips

**Acceptance Criteria:**
- [ ] Checklist complete (all items checked)
- [ ] Deployment guide created
- [ ] Maintenance guide created
- [ ] Handoff documentation ready

---

## Phase 10: Final Testing & Validation (1-2 hours)

### Task 10.1: Smoke Test Complete System
**Status:** NOT STARTED  
**Effort:** 1 hour  
**Dependencies:** All previous tasks

**Description:**
Run smoke tests to verify entire system works end-to-end.

**Steps:**
1. Start API server: `python scripts/run_server.py`
2. Test each endpoint:
   - POST /generate with test request
   - POST /evaluate with test email
   - GET /health
   - Run small batch evaluation (2-3 scenarios instead of 10)
3. Verify:
   - Responses are valid JSON
   - No errors in logs
   - Metrics score emails correctly
   - Both LLM providers work
4. Test CLI scripts:
   - `python scripts/run_evaluation.py --model gemini` (1 scenario)
   - `python scripts/run_comparison.py` (1 scenario)

**Acceptance Criteria:**
- [ ] API starts without errors
- [ ] All endpoints respond correctly
- [ ] Email generation works
- [ ] Evaluation scores emails
- [ ] Both models tested
- [ ] Scripts execute successfully

---

### Task 10.2: Final Code Review & Cleanup
**Status:** NOT STARTED  
**Effort:** 0.5 hours  
**Dependencies:** All previous tasks

**Description:**
Code review and final cleanup.

**Steps:**
1. Code review checklist:
   - [ ] All code follows PEP 8 style guide
   - [ ] All functions have docstrings
   - [ ] Error handling is comprehensive
   - [ ] No hardcoded values (use config)
   - [ ] No debug prints (use logger)
   - [ ] Type hints on all functions
   - [ ] API keys not in any files
   - [ ] Comments explain complex logic
2. Cleanup:
   - Remove any test/debug code
   - Fix any linting warnings
   - Update any TODO comments
   - Format code with black: `black src/ tests/`
   - Check import order: `isort src/ tests/`

**Acceptance Criteria:**
- [ ] Code style consistent
- [ ] All functions documented
- [ ] No API keys exposed
- [ ] Linting passes
- [ ] Code is clean and production-ready

---

### Task 10.3: Create Git Repository & Commit
**Status:** NOT STARTED  
**Effort:** 0.5 hours  
**Dependencies:** All previous tasks

**Description:**
Finalize git repository and create clean commit history.

**Steps:**
1. Ensure `.gitignore` includes:
   - `venv/`, `__pycache__/`, `*.pyc`
   - `.env` (never commit secrets)
   - `.pytest_cache/`, `.coverage`, `htmlcov/`
   - `reports/` (optional, can commit example reports)
2. Review git status: `git status`
3. Commit all code:
   ```
   git add -A
   git commit -m "Complete email generation assistant implementation"
   ```
4. Verify remote (optional):
   - Create GitHub repository
   - Add remote: `git remote add origin <url>`
   - Push: `git push -u origin main`

**Acceptance Criteria:**
- [ ] All files committed except secrets
- [ ] `.gitignore` properly configured
- [ ] `.env` not committed
- [ ] Clean git history
- [ ] Ready for production deployment

---

## Execution Summary

### Total Time Estimate
- **Setup & Infrastructure:** 2-3 hours
- **Core Generation:** 4-6 hours
- **Evaluation Metrics:** 6-8 hours
- **Model Comparison:** 3-4 hours
- **Test Data:** 2-3 hours
- **API Layer:** 3-4 hours
- **Evaluation Scripts:** 2-3 hours
- **Testing & QA:** 3-4 hours
- **Documentation:** 2-3 hours
- **Final Validation:** 1-2 hours
- **TOTAL: 22-31 hours**

### Critical Path (Dependencies)
1. Phase 1 (Setup)
2. Phase 2 (Email Generation)
3. Phase 3 (Metrics)
4. Phase 4 (Comparison)
5. Phase 5 (Test Data)
6. Phase 6 (API)
7. Phases 7-10 (Parallel execution possible)

### Key Milestones
- ✅ Milestone 1 (Hour 4-5): First email generation working
- ✅ Milestone 2 (Hour 12-13): All 3 metrics implemented
- ✅ Milestone 3 (Hour 16-17): API endpoints functional
- ✅ Milestone 4 (Hour 22-23): Evaluation pipeline complete
- ✅ Milestone 5 (Hour 31): Full deployment ready

### Success Metrics
- [ ] All 10 scenarios successfully processed
- [ ] FRS, TFI, SCS metrics all produce 0-100 scores
- [ ] Both LLM models generate valid emails
- [ ] Evaluation produces consistent results
- [ ] Comparison identifies clear winner
- [ ] API fully functional and documented
- [ ] 80%+ test coverage
- [ ] Zero hardcoded secrets in code
- [ ] Complete documentation provided
- [ ] Production-ready code quality

---

**Document Version:** 1.0  
**Created:** 2026-06-19  
**Ready for:** Execution Phase  
**Status:** ✅ Complete & Actionable

