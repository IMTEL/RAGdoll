"""Local transliteration helpers for language-learning features."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransliterationResult:
    text: str
    romanized_text: str
    source: str
    provider: str

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "romanizedText": self.romanized_text,
            "source": self.source,
            "provider": self.provider,
        }


class TransliterationService:
    def transliterate(self, text: str, source: str) -> TransliterationResult:
        normalized_source = self._normalize_language(source)
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("Text is required.")

        if normalized_source not in {"zh", "zh-cn", "cmn"}:
            return TransliterationResult(
                text=normalized_text,
                romanized_text="",
                source=normalized_source,
                provider="none",
            )

        try:
            from pypinyin import Style, lazy_pinyin
        except ImportError as exc:
            raise RuntimeError(
                "Chinese transliteration requires the pypinyin package."
            ) from exc

        romanized = " ".join(
            lazy_pinyin(
                normalized_text,
                style=Style.TONE,
                neutral_tone_with_five=True,
                errors="default",
            )
        )
        return TransliterationResult(
            text=normalized_text,
            romanized_text=romanized,
            source="zh",
            provider="pypinyin",
        )

    @staticmethod
    def _normalize_language(language: str) -> str:
        value = (language or "").strip().lower().replace("_", "-")
        if value in {"zh-hans", "zh-cn", "cn", "mandarin", "chinese"}:
            return "zh"
        return value.split("-")[0] if value else ""


_service: TransliterationService | None = None


def get_transliteration_service() -> TransliterationService:
    global _service
    if _service is None:
        _service = TransliterationService()
    return _service
