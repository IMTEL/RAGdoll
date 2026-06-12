"""Translation service abstraction for language-learning features."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import Lock
from time import monotonic

import requests

from src.config import Config


@dataclass(frozen=True)
class TranslationResult:
    text: str
    translated_text: str
    source: str
    target: str
    provider: str
    cached: bool = False

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "translatedText": self.translated_text,
            "source": self.source,
            "target": self.target,
            "provider": self.provider,
            "cached": self.cached,
        }


class TranslationService(ABC):
    @abstractmethod
    def translate(
        self, text: str, source: str = "auto", target: str = "en"
    ) -> TranslationResult:
        """Translate text from source language to target language."""

    def translate_many(
        self, texts: list[str], source: str = "auto", target: str = "en"
    ) -> list[TranslationResult]:
        """Translate multiple texts. Providers may override this for batching."""
        return [self.translate(text, source=source, target=target) for text in texts]

    def warmup(self, source: str = "es", target: str = "en") -> dict:
        """Warm the provider and loaded language models."""
        result = self.translate("Hola.", source=source, target=target)
        return {
            "ok": True,
            "provider": result.provider,
            "source": source,
            "target": target,
        }


class DisabledTranslationService(TranslationService):
    def translate(
        self, text: str, source: str = "auto", target: str = "en"
    ) -> TranslationResult:
        raise RuntimeError("Translation is disabled.")


class LibreTranslateService(TranslationService):
    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.base_url = self.config.TRANSLATION_BASE_URL.rstrip("/")
        self.api_key = self.config.TRANSLATION_API_KEY
        self.timeout = self.config.TRANSLATION_TIMEOUT_SECONDS
        self.cache_ttl_seconds = self.config.TRANSLATION_CACHE_TTL_SECONDS
        self.cache_max_entries = self.config.TRANSLATION_CACHE_MAX_ENTRIES
        self._cache: dict[tuple[str, str, str], tuple[float, TranslationResult]] = {}
        self._cache_lock = Lock()

    def translate(
        self, text: str, source: str = "auto", target: str = "en"
    ) -> TranslationResult:
        source_language = self._normalize_language_code(source)
        target_language = self._normalize_language_code(target)
        normalized_text = text.strip()
        if not normalized_text:
            raise RuntimeError("Text is required.")

        cache_key = self._cache_key(normalized_text, source_language, target_language)

        cached_result = self._get_cached(cache_key)
        if cached_result:
            return cached_result

        payload = {
            "q": normalized_text,
            "source": source_language,
            "target": target_language,
            "format": "text",
        }
        if self.api_key:
            payload["api_key"] = self.api_key

        response = requests.post(
            f"{self.base_url}/translate",
            json=payload,
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"LibreTranslate returned HTTP {response.status_code}: {response.text}"
            )

        data = response.json()
        translated_text = data.get("translatedText")
        if not translated_text:
            raise RuntimeError("LibreTranslate did not return translatedText.")

        detected_language = data.get("detectedLanguage")
        detected_source = source_language
        if isinstance(detected_language, dict):
            detected_source = detected_language.get("language") or source_language

        result = TranslationResult(
            text=normalized_text,
            translated_text=translated_text,
            source=detected_source or source_language,
            target=target_language,
            provider="libretranslate",
        )
        self._set_cached(cache_key, result)
        return result

    def translate_many(
        self, texts: list[str], source: str = "auto", target: str = "en"
    ) -> list[TranslationResult]:
        source_language = self._normalize_language_code(source)
        target_language = self._normalize_language_code(target)
        normalized_texts = [text.strip() for text in texts]
        results_by_index: dict[int, TranslationResult] = {}
        results_by_text: dict[str, TranslationResult] = {}
        missing: list[str] = []

        for index, text in enumerate(normalized_texts):
            if not text:
                continue
            cache_key = self._cache_key(text, source_language, target_language)
            cached_result = self._get_cached(cache_key)
            if cached_result:
                results_by_index[index] = cached_result
            elif text not in missing:
                missing.append(text)

        for text in missing:
            results_by_text[text] = self.translate(
                text, source=source_language, target=target_language
            )

        for index, text in enumerate(normalized_texts):
            if index not in results_by_index and text:
                results_by_index[index] = results_by_text[text]

        return [results_by_index[index] for index in sorted(results_by_index)]

    def warmup(self, source: str = "es", target: str = "en") -> dict:
        languages_response = requests.get(
            f"{self.base_url}/languages", timeout=self.timeout
        )
        if languages_response.status_code != 200:
            raise RuntimeError(
                f"LibreTranslate languages check returned HTTP "
                f"{languages_response.status_code}: {languages_response.text}"
            )
        result = self.translate("Hola.", source=source, target=target)
        return {
            "ok": True,
            "provider": result.provider,
            "source": result.source,
            "target": result.target,
            "cached": result.cached,
        }

    def _get_cached(
        self, cache_key: tuple[str, str, str]
    ) -> TranslationResult | None:
        if self.cache_ttl_seconds <= 0 or self.cache_max_entries <= 0:
            return None

        now = monotonic()
        with self._cache_lock:
            cached_entry = self._cache.get(cache_key)
            if not cached_entry:
                return None
            created_at, result = cached_entry
            if now - created_at > self.cache_ttl_seconds:
                self._cache.pop(cache_key, None)
                return None
            return TranslationResult(
                text=result.text,
                translated_text=result.translated_text,
                source=result.source,
                target=result.target,
                provider=result.provider,
                cached=True,
            )

    def _set_cached(
        self, cache_key: tuple[str, str, str], result: TranslationResult
    ) -> None:
        if self.cache_ttl_seconds <= 0 or self.cache_max_entries <= 0:
            return

        with self._cache_lock:
            if len(self._cache) >= self.cache_max_entries:
                oldest_key = min(self._cache, key=lambda key: self._cache[key][0])
                self._cache.pop(oldest_key, None)
            self._cache[cache_key] = (monotonic(), result)

    def _cache_key(self, text: str, source: str, target: str) -> tuple[str, str, str]:
        return (text, source, target)

    @staticmethod
    def _normalize_language_code(language: str) -> str:
        normalized = (language or "").strip().lower().replace("_", "-")
        if not normalized:
            return "auto"
        if normalized in {"no", "nor", "nb-no"}:
            return "nb"
        if normalized in {"nn-no"}:
            return "nn"
        return normalized


_translation_service: TranslationService | None = None
_translation_service_lock = Lock()


def get_translation_service() -> TranslationService:
    global _translation_service
    config = Config()
    with _translation_service_lock:
        if _translation_service is None:
            if config.TRANSLATION_PROVIDER == "libretranslate":
                _translation_service = LibreTranslateService(config)
            else:
                _translation_service = DisabledTranslationService()
        return _translation_service
