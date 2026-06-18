"""
LangGraph nodes and graph compilation for the email generation pipeline.

Implements the 5-node pipeline with conditional retry edges:

    validate_input → construct_prompt → call_llm → post_process → quality_check
                                                                     │
                                          ┌──────────────────────────┘
                                          ▼ (if quality failed & retries < 2)
                                     construct_prompt  (retry with stronger instructions)
                                          │
                                          ▼
                                     call_llm → post_process → quality_check …
                                                                     │
                                                                     ▼
                                                              (if passed or retries ≥ 2)
                                                              → end
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph

from src.api.models.enums import ToneEnum
from src.api.models.requests import EmailRequest
from src.core.state import EmailGenerationState, create_initial_state
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  NODE 1 — validate_input                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ── Intent category detection ───────────────────────────────────────────────

_INTENT_KEYWORDS: Dict[str, List[str]] = {
    "follow-up": ["follow up", "follow-up", "next steps", "update", "demo", "meeting"],
    "request": ["request", "need", "require", "deadline", "extension", "approval"],
    "announcement": ["announce", "inform", "notify", "relocat", "moving", "change"],
    "escalation": ["escalat", "urgent", "unresolved", "incident", "immediate", "security"],
    "congratulations": ["congratul", "promotion", "celebrat", "achievement", "well done"],
    "proposal": ["proposal", "vendor", "rfp", "budget", "propos", "purchase"],
    "apology": ["apolog", "delay", "late", "sorry", "overdue", "payment"],
    "policy": ["policy", "effective", "remote work", "office", "compliance"],
    "persuasion": ["persuade", "approve", "roi", "benefit", "opportunity", "stakeholder"],
    "condolences": ["condolence", "passed away", "loss", "memorial", "counseling"],
}


def _detect_intent_category(intent: str) -> str:
    """Classify the intent string into a category.

    Uses simple keyword matching.  Returns ``"general"`` if no specific
    category matches.
    """
    intent_lower = intent.lower()
    best_category = "general"
    best_score = 0

    for category, keywords in _INTENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in intent_lower)
        if score > best_score:
            best_score = score
            best_category = category

    return best_category


# ── Complexity estimation ───────────────────────────────────────────────────


def _estimate_complexity(request: EmailRequest) -> str:
    """Estimate email complexity based on facts, tone, and intent length.

    Returns ``"low"``, ``"medium"``, or ``"high"``.
    """
    score = 0

    # More facts = higher complexity
    num_facts = len(request.key_facts)
    if num_facts >= 4:
        score += 2
    elif num_facts >= 2:
        score += 1

    # Longer intent = more specific = higher complexity
    if len(request.intent) >= 150:
        score += 2
    elif len(request.intent) >= 80:
        score += 1

    # Certain tones add complexity
    complex_tones = {ToneEnum.PERSUASIVE, ToneEnum.EMPATHETIC, ToneEnum.APOLOGETIC}
    if request.tone in complex_tones:
        score += 1

    if score >= 4:
        return "high"
    if score >= 2:
        return "medium"
    return "low"


# ── PII detection (basic) ───────────────────────────────────────────────────

_PII_PATTERNS = [
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),  # email
    re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),  # phone
    re.compile(r"\b\d{3}[-]?\d{2}[-]?\d{4}\b"),  # SSN-like
]


def _detect_pii(text: str) -> List[str]:
    """Detect likely PII in a string and return redacted snippets."""
    found: List[str] = []
    for i, pattern in enumerate(_PII_PATTERNS):
        matches = pattern.findall(text)
        for m in matches:
            found.append(f"pattern_{i}:{m[:4]}****")
    return found


async def validate_input(state: EmailGenerationState) -> Dict[str, Any]:
    """Node 1: Validate and enrich the input state.

    * Detects intent category for few-shot selection.
    * Estimates complexity score.
    * Checks for PII in facts (logs warning, does not block).
    * Handles edge cases like ambiguous intent.
    """
    request = state["request"]
    logger.info(
        "Validating input",
        intent=request.intent[:60],
        tone=request.tone.value,
        facts_count=len(request.key_facts),
    )

    # Detect intent category
    intent_category = _detect_intent_category(request.intent)

    # Estimate complexity
    complexity = _estimate_complexity(request)

    # PII detection (non-blocking — just logs)
    for i, fact in enumerate(request.key_facts):
        pii_found = _detect_pii(fact)
        if pii_found:
            logger.warning(
                "Possible PII detected in fact",
                fact_index=i,
                patterns=pii_found,
            )

    logger.info(
        "Input validated",
        intent_category=intent_category,
        complexity=complexity,
    )

    return {
        "intent_category": intent_category,
        "complexity": complexity,
    }


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  NODE 2 — construct_prompt                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ── Inline tone profiles (used until Phase 3 tone profiles are loaded) ─────

_TONE_GUIDANCE: Dict[str, Dict[str, Any]] = {
    "formal": {
        "markers": "No contractions, full titles, structured phrasing",
        "word_choice": ["would like", "please advise", "at your earliest convenience"],
        "avoid": ["contractions", "slang", "exclamation marks"],
        "greeting": "Dear {name},",
        "closing": "Respectfully,",
    },
    "casual": {
        "markers": "Contractions OK, first names, warm and friendly",
        "word_choice": ["thanks", "hey", "great", "looking forward"],
        "avoid": ["overly formal language", "stiff phrasing"],
        "greeting": "Hi {name},",
        "closing": "Best,",
    },
    "urgent": {
        "markers": "Direct imperatives, time-sensitive, action-oriented",
        "word_choice": ["immediately", "action required", "urgent", "deadline"],
        "avoid": ["vague language", "long sentences", "softeners"],
        "greeting": "Dear {name},",
        "closing": "Regards,",
    },
    "empathetic": {
        "markers": "Supportive, validating feelings, gentle pacing",
        "word_choice": ["I understand", "I know this is difficult", "we are here to help"],
        "avoid": ["dismissive language", "rushing", "cold facts without context"],
        "greeting": "Dear {name},",
        "closing": "Warmly,",
    },
    "persuasive": {
        "markers": "Benefit-focused, compelling, confident",
        "word_choice": ["imagine", "opportunity", "benefit", "ROI", "value"],
        "avoid": ["aggressive pressure", "uncertainty", "hedging"],
        "greeting": "Dear {name},",
        "closing": "Best regards,",
    },
    "apologetic": {
        "markers": "Remorseful, accountable, deferential",
        "word_choice": ["I apologize", "I take full responsibility", "please accept"],
        "avoid": ["defensive language", "excuses", "blame shifting"],
        "greeting": "Dear {name},",
        "closing": "Sincerely,",
    },
}


def _build_system_prompt(tone: str, tone_profile: Dict[str, Any]) -> str:
    """Build the Dr. Elena Voss role-playing system prompt."""
    guidance = tone_profile
    return (
        "You are Dr. Elena Voss, a senior executive communication coach who has trained "
        "C-suite leaders at Fortune 500 companies for 18 years. You specialize in "
        "translating business intent into precisely-toned, fact-rich professional emails.\n\n"
        "Your emails are known for:\n"
        "- Seamless fact integration (never list-like, always narrative)\n"
        f"- Exact tone matching ({guidance.get('markers', 'professional')})\n"
        "- Strategic structure (hook → context → facts → call-to-action → close)"
    )


def _build_cot_instructions() -> str:
    """Build chain-of-thought reasoning instructions."""
    return (
        "Before writing the email, think through:\n"
        "1. What is the relationship between sender and recipient?\n"
        "2. Which facts are most important and where should they appear?\n"
        "3. What tone markers (word choice, sentence length) achieve the requested tone?\n"
        "4. How should the email open and close for maximum impact?\n\n"
        "Write your thinking in <thinking> tags, then the email."
    )


def _build_few_shot_section(examples: List[Dict[str, Any]]) -> str:
    """Build the few-shot examples section of the prompt."""
    if not examples:
        return ""

    lines = ["FEW-SHOT EXAMPLES:", ""]
    for i, ex in enumerate(examples, start=1):
        tone_label = ex.get("tone", "formal")
        category = ex.get("category", "general")
        lines.append(f"[Example {i} — {tone_label.capitalize()} / {category}]")
        lines.append(f"Intent: {ex.get('intent', '')}")
        lines.append(f"Facts: {', '.join(ex.get('facts', []))}")
        lines.append(f"Tone: {tone_label}")
        lines.append("")
        lines.append(ex.get("output", ""))
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def _build_tone_section(tone: str, tone_profile: Dict[str, Any]) -> str:
    """Build tone-specific guidance section."""
    guidance = tone_profile
    parts = [
        f"Tone: {tone}",
        f"Markers: {guidance.get('markers', 'professional')}",
    ]
    word_choice = guidance.get("word_choice", [])
    if word_choice:
        parts.append(f"Preferred words: {', '.join(word_choice[:5])}")
    avoid = guidance.get("avoid", [])
    if avoid:
        parts.append(f"Avoid: {', '.join(avoid[:4])}")
    return "\n".join(parts)


async def construct_prompt(state: EmailGenerationState) -> Dict[str, Any]:
    """Node 2: Assemble the complete prompt from components.

    Builds the prompt from:
    * System role prompt (Dr. Elena Voss)
    * Few-shot examples
    * Chain-of-thought instructions
    * Tone-specific guidance
    * Task specification with facts and constraints

    On retry cycles (when quality check failed), this node also
    increments ``retry_count`` and adds stronger instructions.
    """
    request = state["request"]
    tone = request.tone.value
    tone_profile = _TONE_GUIDANCE.get(tone, _TONE_GUIDANCE["formal"])
    examples = state.get("selected_examples", [])

    # Determine retry count — increment if we're in a retry loop
    current_retry = state.get("retry_count", 0)
    if not state.get("quality_passed", True) and current_retry < 2:
        current_retry += 1

    # If this is a retry, add stronger instructions
    retry_notice = ""
    if current_retry > 0:
        retry_notice = (
            f"\nNOTE: This is retry #{state['retry_count']}. The previous email "
            "did not pass quality checks. Please ensure:\n"
            "- A clear subject line starting with 'Subject:'\n"
            "- An appropriate greeting (Hi/Dear/Hello)\n"
            "- All facts integrated naturally (not as a bullet list)\n"
            f"- Word count between {max(50, request.max_words - 60)} and {request.max_words + 60} words\n"
            "- A professional closing\n"
            "- No placeholder text like [Name], [Company], [Email]\n"
        )

    min_words = max(50, request.max_words - 60)
    max_words = request.max_words + 60

    system_prompt = _build_system_prompt(tone, tone_profile)
    few_shot_section = _build_few_shot_section(examples)
    cot_instructions = _build_cot_instructions()
    tone_section = _build_tone_section(tone, tone_profile)

    recipient = request.recipient_name or "[Recipient Name]"
    greeting_line = tone_profile.get("greeting", "Dear {name},").format(name=recipient)
    closing_line = tone_profile.get("closing", "Best regards,")

    prompt = f"""{system_prompt}

{few_shot_section}

{cot_instructions}

{retry_notice}

TASK:
Generate a professional email with the following specifications:

INTENT: {request.intent}

KEY FACTS (MUST ALL BE INCLUDED):
{chr(10).join(f'- {fact}' for fact in request.key_facts)}

TONE: {tone}
TONE PROFILE:
{tone_section}

CONSTRAINTS:
- Word count: Between {min_words} and {max_words} words
- Must include all key facts naturally (not as a list)
- Must match the {tone} tone precisely
- Include appropriate greeting ({greeting_line}) and professional closing ({closing_line})
- Subject line should be clear and relevant
- Sender name: {request.sender_name or '[Your Name]'}

OUTPUT FORMAT:
<thinking>
[Your reasoning process here]
</thinking>

---
Subject: [Subject line]

[Email body]

[Closing]
[Name]"""

    logger.info(
        "Prompt constructed",
        tone=tone,
        examples_count=len(examples),
        retry=current_retry,
    )

    return {
        "prompt": prompt,
        "tone_profile": tone_profile,
        "retry_count": current_retry,
    }


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  NODE 3 — call_llm                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


async def call_llm(state: EmailGenerationState) -> Dict[str, Any]:
    """Node 3: Call the LLM provider to generate the email.

    Currently returns a placeholder message.  Phase 4 will implement
    the actual provider calls with rate-limit handling, fallback, etc.

    In production this will:
    1. Check rate-limiter for the requested provider.
    2. Call the primary provider (Gemini).
    3. If rate-limited → switch to fallback (Groq).
    4. If all fail → set error state.
    """
    request = state["request"]
    prompt = state["prompt"]

    logger.info(
        "Calling LLM",
        model=request.model.value,
        prompt_length=len(prompt),
        retry_count=state["retry_count"],
    )

    # ── Placeholder — Phase 4 replaces with real provider calls ──
    # Simulate latency for realistic metadata
    start_time = time.monotonic()
    await asyncio.sleep(0.05)  # Simulated 50ms call
    latency_ms = (time.monotonic() - start_time) * 1000

    # Generate a simple placeholder email
    facts_bullets = "\n".join(f"- {fact}" for fact in request.key_facts)
    raw_output = (
        f"<thinking>\n"
        f"1. The relationship is professional, with {request.recipient_name or 'the recipient'} "
        f"as the intended audience.\n"
        f"2. The most important facts revolve around: {request.intent[:80]}...\n"
        f"3. For {request.tone.value} tone, I will use appropriate markers.\n"
        f"4. The email will open with a clear subject line and greeting, "
        f"present facts in narrative form, and close professionally.\n"
        f"</thinking>\n\n"
        f"---\n"
        f"Subject: {request.intent[:80]}...\n\n"
        f"Dear {request.recipient_name or 'Team'},\n\n"
        f""
        f"I hope this message finds you well. I am writing to follow up regarding: "
        f"{request.intent}\n\n"
        f"Key points to consider:\n"
        f"{facts_bullets}\n\n"
        f"I look forward to your response and am happy to discuss further "
        f"at your convenience.\n\n"
        f"Best regards,\n"
        f"{request.sender_name or '[Your Name]'}"
    )

    metadata = {
        "model_used": request.model.value,
        "provider": request.model.value,
        "latency_ms": round(latency_ms, 1),
        "retry_count": state["retry_count"],
        "intent_category": state["intent_category"],
        "complexity": state["complexity"],
    }

    logger.info(
        "LLM response received",
        model=request.model.value,
        latency_ms=round(latency_ms, 1),
        output_length=len(raw_output),
    )

    return {
        "raw_output": raw_output,
        "metadata": metadata,
    }


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  NODE 4 — post_process                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ── Constants for post-processing ───────────────────────────────────────────

_THINKING_TAG_RE = re.compile(r"<thinking>.*?</thinking>", re.DOTALL)
_SUBJECT_RE = re.compile(r"^Subject:\s*(.+)$", re.MULTILINE)
_GREETING_RE = re.compile(r"^(Dear|Hi|Hello|Hey)\b", re.MULTILINE)
_CLOSING_RE = re.compile(
    r"^(Best|Regards|Sincerely|Thanks|Cheers|Warmly|Respectfully|Warm regards|"
    r"Best regards|Best wishes|Yours sincerely|Yours faithfully)",
    re.MULTILINE,
)
_PLACEHOLDER_RE = re.compile(r"\[(Name|Company|Email|Recipient Name|Your Name|Recipient)\]")


def _strip_thinking_tags(text: str) -> str:
    """Remove <thinking>...</thinking> blocks from LLM output."""
    return _THINKING_TAG_RE.sub("", text).strip()


def _extract_subject(text: str) -> str:
    """Extract the subject line from the email."""
    match = _SUBJECT_RE.search(text)
    return match.group(1).strip() if match else ""


def _ensure_greeting(text: str, recipient: Optional[str], tone: str) -> str:
    """Ensure the email has a proper greeting."""
    if _GREETING_RE.search(text):
        return text

    tone_profile = _TONE_GUIDANCE.get(tone, _TONE_GUIDANCE["formal"])
    name = recipient or "Team"
    greeting = tone_profile.get("greeting", "Dear {name},").format(name=name)
    # Insert greeting after the subject line
    lines = text.split("\n")
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("Subject:"):
            insert_idx = i + 1
            break
    # Skip any blank lines after subject
    while insert_idx < len(lines) and lines[insert_idx].strip() == "":
        insert_idx += 1
    lines.insert(insert_idx, f"\n{greeting}\n")
    return "\n".join(lines)


def _ensure_closing(text: str, sender: Optional[str], tone: str) -> str:
    """Ensure the email has a professional closing and signature."""
    if _CLOSING_RE.search(text):
        return text

    tone_profile = _TONE_GUIDANCE.get(tone, _TONE_GUIDANCE["formal"])
    closing = tone_profile.get("closing", "Best regards,")
    signature = sender or "[Your Name]"

    return f"{text}\n\n{closing}\n{signature}"


def _check_placeholder_text(text: str) -> List[str]:
    """Detect placeholder text like [Name], [Company], etc."""
    flags: List[str] = []
    for match in _PLACEHOLDER_RE.finditer(text):
        flags.append(f"Placeholder '{match.group()}' found — replace with actual value")
    return flags


def _ensure_signature(text: str, sender_name: Optional[str]) -> str:
    """Add a signature if the email doesn't have one."""
    if sender_name and sender_name not in text:
        text = f"{text.rstrip()}\n\n{_CLOSING_RE.search(text).group() if _CLOSING_RE.search(text) else 'Best regards'},\n{sender_name}"
    return text


async def post_process(state: EmailGenerationState) -> Dict[str, Any]:
    """Node 4: Clean and structure the raw LLM output.

    * Strips <thinking> tags.
    * Extracts subject line.
    * Ensures proper greeting and closing.
    * Validates word count.
    * Detects placeholder text.
    * Normalises whitespace.
    """
    raw = state["raw_output"]
    request = state["request"]
    tone = request.tone.value

    logger.info("Post-processing email", input_length=len(raw))

    # 1. Strip thinking tags
    cleaned = _strip_thinking_tags(raw)

    # 2. Ensure subject line
    subject = _extract_subject(cleaned)
    if not subject:
        cleaned = f"Subject: {request.intent[:80]}...\n\n{cleaned}"
        logger.warning("Missing subject line — prepended auto-subject")

    # 3. Ensure greeting
    cleaned = _ensure_greeting(cleaned, request.recipient_name, tone)

    # 4. Ensure closing and signature
    cleaned = _ensure_closing(cleaned, request.sender_name, tone)

    # 5. Check for placeholders
    placeholder_flags = _check_placeholder_text(cleaned)

    # 6. Normalise whitespace
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    # 7. Count words
    word_count = len(cleaned.split())

    logger.info(
        "Post-processing complete",
        word_count=word_count,
        has_greeting=bool(_GREETING_RE.search(cleaned)),
        has_closing=bool(_CLOSING_RE.search(cleaned)),
        has_subject=bool(_SUBJECT_RE.search(cleaned)),
        placeholders=len(placeholder_flags),
    )

    return {
        "cleaned_email": cleaned,
    }


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  NODE 5 — quality_check                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


_QUALITY_GREETING_RE = re.compile(r"^(Dear|Hi|Hello|Hey)\b", re.MULTILINE)
_QUALITY_CLOSING_RE = re.compile(
    r"^(Best|Regards|Sincerely|Thanks|Cheers|Warmly|Respectfully|"
    r"Warm regards|Best regards|Best wishes|Yours sincerely|Yours faithfully)",
    re.MULTILINE,
)
_QUALITY_SUBJECT_RE = re.compile(r"^Subject:\s*\S", re.MULTILINE)
_QUALITY_PLACEHOLDER_RE = re.compile(r"\[(Name|Company|Email|Your Name|Recipient)\]")


def _count_words(text: str) -> int:
    """Count the number of words in a text string."""
    return len(text.split())


async def quality_check(state: EmailGenerationState) -> Dict[str, Any]:
    """Node 5: Run heuristic quality checks on the cleaned email.

    Checks:
    * Has greeting? (Hi|Dear|Hello|Hey)
    * Has closing? (Best|Regards|Sincerely|Thanks|Cheers)
    * Has subject line? (line starting with ``Subject:``)
    * Word count within ±20% of target?
    * No placeholder text like ``[Name]``, ``[Company]``?

    If all pass → ``quality_passed=True``.
    If any fail AND ``retry_count < 2`` → ``quality_passed=False`` (retry triggered).
    If any fail AND ``retry_count >= 2`` → ``quality_passed=False`` (proceed with warnings).
    """
    email = state["cleaned_email"]
    request = state["request"]
    target_words = request.max_words
    actual_words = _count_words(email)

    flags: List[str] = []
    scores: Dict[str, float] = {}

    # 1. Greeting check
    has_greeting = bool(_QUALITY_GREETING_RE.search(email))
    if not has_greeting:
        flags.append("Missing greeting — expected Hi/Dear/Hello/Hey")
    scores["has_greeting"] = 1.0 if has_greeting else 0.0

    # 2. Closing check
    has_closing = bool(_QUALITY_CLOSING_RE.search(email))
    if not has_closing:
        flags.append("Missing closing — expected Best/Regards/Sincerely/Thanks")
    scores["has_closing"] = 1.0 if has_closing else 0.0

    # 3. Subject line check
    has_subject = bool(_QUALITY_SUBJECT_RE.search(email))
    if not has_subject:
        flags.append("Missing or empty subject line")
    scores["has_subject"] = 1.0 if has_subject else 0.0

    # 4. Word count check (±20%)
    min_ok = target_words * 0.8
    max_ok = target_words * 1.2
    word_count_ok = min_ok <= actual_words <= max_ok
    if not word_count_ok:
        if actual_words < min_ok:
            flags.append(
                f"Too short ({actual_words} words, target {target_words} ±20%)"
            )
        else:
            flags.append(
                f"Too long ({actual_words} words, target {target_words} ±20%)"
            )

    word_count_score = 1.0 if word_count_ok else 0.0
    if not word_count_ok:
        # Partial credit based on deviation
        deviation = abs(actual_words - target_words) / target_words
        word_count_score = max(0.0, 1.0 - deviation)
    scores["word_count"] = round(word_count_score, 2)

    # 5. Placeholder text check
    placeholders = _QUALITY_PLACEHOLDER_RE.findall(email)
    if placeholders:
        flags.append(
            f"Placeholder text detected: {', '.join(sorted(set(placeholders)))}"
        )
    scores["no_placeholders"] = 0.0 if placeholders else 1.0

    # Overall quality pass/fail
    all_essential = all(
        scores.get(k, 0) >= 0.5 for k in ("has_greeting", "has_closing", "has_subject")
    )
    quality_passed = all_essential and word_count_ok and not placeholders
    overall_score = round(
        (scores.get("has_greeting", 0) * 0.25
         + scores.get("has_closing", 0) * 0.25
         + scores.get("has_subject", 0) * 0.20
         + scores.get("word_count", 0) * 0.20
         + scores.get("no_placeholders", 0) * 0.10)
        * 100,
        1,
    )

    logger.info(
        "Quality check complete",
        passed=quality_passed,
        score=overall_score,
        flags=flags,
        retry_count=state["retry_count"],
    )

    return {
        "scores": {**scores, "overall": overall_score},
        "quality_passed": quality_passed,
        "quality_flags": flags,
    }


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CONDITIONAL EDGE — route_after_quality                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


def route_after_quality(state: EmailGenerationState) -> str:
    """Determine the next step after the quality check.

    Returns:
    * ``"construct_prompt"`` — if quality failed and retries remain.
    * ``"error_handler"`` — if there is an error.
    * ``"__end__"`` — if quality passed or retries exhausted.
    """
    if state.get("error"):
        return "error_handler"

    if not state.get("quality_passed", True) and state.get("retry_count", 0) < 2:
        logger.info(
            "Quality check failed — routing to retry",
            retry_count=state["retry_count"],
        )
        return "construct_prompt"

    if not state.get("quality_passed", True):
        logger.warning(
            "Quality check failed after max retries — proceeding with warnings",
            retry_count=state["retry_count"],
        )

    return "__end__"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  ERROR HANDLER NODE                                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


async def error_handler(state: EmailGenerationState) -> Dict[str, Any]:
    """Handle unrecoverable errors in the pipeline.

    Logs the error and returns the state with the error flag set so the
    orchestrator can raise the appropriate exception.
    """
    error_msg = state.get("error", "Unknown error in email generation pipeline")
    logger.error(
        "Email generation pipeline failed",
        error=error_msg,
        retry_count=state.get("retry_count", 0),
    )
    return {
        "quality_flags": state.get("quality_flags", []) + [f"Pipeline error: {error_msg}"],
    }


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  GRAPH COMPILATION                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


def build_email_graph() -> StateGraph:
    """Build and return the compiled email generation LangGraph.

    The graph has 5 main nodes plus an error handler:

    .. code-block::

        validate_input
            ↓
        construct_prompt
            ↓
        call_llm
            ↓
        post_process
            ↓
        quality_check
            ↓ (conditional)
        ┌─ construct_prompt  (if quality failed & retries < 2)
        │  ↓
        │  call_llm → post_process → quality_check …
        │                                            ↓
        └────────────────────────────────── (passed or retries ≥ 2)
                                                     ↓
                                                __end__
    """
    builder = StateGraph(EmailGenerationState)

    # Add nodes
    builder.add_node("validate_input", validate_input)
    builder.add_node("construct_prompt", construct_prompt)
    builder.add_node("call_llm", call_llm)
    builder.add_node("post_process", post_process)
    builder.add_node("quality_check", quality_check)
    builder.add_node("error_handler", error_handler)

    # Set entry point
    builder.set_entry_point("validate_input")

    # Sequential edges
    builder.add_edge("validate_input", "construct_prompt")
    builder.add_edge("construct_prompt", "call_llm")
    builder.add_edge("call_llm", "post_process")
    builder.add_edge("post_process", "quality_check")

    # Conditional retry edge from quality_check
    builder.add_conditional_edges(
        "quality_check",
        route_after_quality,
        {
            "construct_prompt": "construct_prompt",
            "error_handler": "error_handler",
            "__end__": END,
        },
    )

    # Error handler goes to END
    builder.add_edge("error_handler", END)

    logger.info("Email generation graph compiled")

    return builder.compile()
