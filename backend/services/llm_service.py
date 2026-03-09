"""
Gemini LLM Service (google-genai SDK).

Purpose: Wraps the Google Gemini API for intelligent reasoning in the HS
         classification pipeline. Generates clarifying questions and
         provides explanations based on candidate HS codes.

Inputs:  Candidate HS codes, user description, conversation context.
Outputs: Natural language questions, classification explanations.
Logging: Logs API calls and errors.
Failure: Returns graceful fallback responses on API errors.
Retry:   Retries up to 3 times with exponential backoff (respects 429 delays).
"""

import time

from google import genai
from google.genai import types

from backend.utils.logger import get_logger

logger = get_logger("llm_service")

# System prompt for the HS classification assistant
SYSTEM_PROMPT = """You are an expert customs classification assistant. Your job is to help users find the correct Harmonized System (HS) code for their products.

RULES:
1. You must ONLY use HS codes from the candidates provided. Never invent codes.
2. Ask short, simple, natural questions — like talking to a friend, not a bureaucrat.
3. Focus on distinguishing features: material, purpose, state (fresh/frozen/dried), processing.
4. One question at a time. Keep it under 2 sentences.
5. When you have enough info to decide, state the HS code confidently with a brief explanation.
6. Be friendly and approachable. Avoid trade jargon.
7. If the user's answer clearly matches one candidate, return the result immediately.

RESPONSE FORMAT:
- If you need more info: Ask ONE short question. Nothing else.
- If you can classify: Reply with EXACTLY this format:
  RESULT: [6-digit HS code]
  DESCRIPTION: [official description]
  EXPLANATION: [1-2 sentence explanation of why this code fits]
"""

MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 5.0  # seconds


class GeminiService:
    """Handles communication with the Google Gemini API via google-genai SDK."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash") -> None:
        self._api_key = api_key
        self._model_name = model_name
        self._client = None
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def initialize(self) -> None:
        """Initialize the Gemini client."""
        logger.info("Initializing Gemini model: %s", self._model_name)
        self._client = genai.Client(api_key=self._api_key)
        self._initialized = True
        logger.info("Gemini client initialized successfully")

    def generate_response(
        self,
        user_query: str,
        candidates_text: str,
        conversation_history: list[dict],
    ) -> str:
        """Generate a response using Gemini for classification reasoning.

        Args:
            user_query: The current user message.
            candidates_text: Formatted string of candidate HS codes.
            conversation_history: List of {"role": "user"/"model", "parts": [text]}.

        Returns:
            Generated text response from Gemini.
        """
        if not self._initialized or self._client is None:
            raise RuntimeError("Gemini not initialized. Call initialize() first.")

        # Build the content list for the API
        contents = []

        if not conversation_history:
            # New conversation — provide full context
            context_msg = (
                f"The user wants to classify this product: \"{user_query}\"\n\n"
                f"Here are the top candidate HS codes from semantic search:\n\n"
                f"{candidates_text}\n\n"
                f"Based on these candidates, either classify the product directly "
                f"if you're confident, or ask ONE short clarifying question to "
                f"narrow it down."
            )
            contents.append(
                types.Content(role="user", parts=[types.Part.from_text(text=context_msg)])
            )
        else:
            # Continue existing conversation
            for msg in conversation_history:
                role = msg.get("role", "user")
                text = msg.get("parts", [""])[0] if msg.get("parts") else ""
                contents.append(
                    types.Content(role=role, parts=[types.Part.from_text(text=text)])
                )
            # Add new user message with updated candidates
            follow_up = (
                f"User's answer: \"{user_query}\"\n\n"
                f"Remaining candidates:\n{candidates_text}\n\n"
                f"Based on this answer, either classify the product or ask "
                f"another short question."
            )
            contents.append(
                types.Content(role="user", parts=[types.Part.from_text(text=follow_up)])
            )

        # Call Gemini with retry logic and exponential backoff
        delay = INITIAL_RETRY_DELAY
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info("Gemini API call (attempt %d/%d)", attempt, MAX_RETRIES)

                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.3,
                        max_output_tokens=300,
                    ),
                )

                text = response.text.strip()
                logger.info("Gemini response (%d chars): %.100s", len(text), text)
                return text

            except Exception as e:
                error_str = str(e)
                logger.warning("Gemini API error (attempt %d): %s", attempt, error_str[:200])

                if "429" in error_str or "quota" in error_str.lower():
                    # Rate limited — use longer delay
                    wait_time = max(delay, 35.0)
                    logger.info("Rate limited, waiting %.0fs before retry...", wait_time)
                    time.sleep(wait_time)
                    delay *= 2
                elif attempt < MAX_RETRIES:
                    logger.info("Retrying in %.0fs...", delay)
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error("Gemini API failed after %d retries", MAX_RETRIES)
                    return self._fallback_response()

        return self._fallback_response()

    def _fallback_response(self) -> str:
        """Return a graceful fallback when the API fails."""
        return (
            "I'm having trouble connecting to my reasoning engine right now. "
            "Could you describe your product in more detail — "
            "what material is it made of and what is it used for?"
        )
