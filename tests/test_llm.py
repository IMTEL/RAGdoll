import pytest
from src.LLM import create_llm

def test_llm_comparison():
    """
    Test that compares responses from both OpenAI and Gemini models.
    Demonstrates how to use both models and display their outputs.
    """
    # Test prompt
    base_prompt = "What are the key differences between Python and JavaScript?"
    
    # Create instances of both LLM types
    openai_llm = create_llm("openai")
    gemini_llm = create_llm("gemini")
    
    # Generate prompts with the same context
    context = {"audience": "beginners", "max_length": "brief"}
    openai_prompt = openai_llm.create_prompt(base_prompt, **context)
    gemini_prompt = gemini_llm.create_prompt(base_prompt, **context)
    
    # Generate responses
    print("\n========== GENERATING RESPONSES ==========")
    
    print("\n[OPENAI MODEL]:", openai_llm.model)
    openai_response = openai_llm.generate(openai_prompt)
    print(openai_response)
    
    print("\n[GEMINI MODEL]:", gemini_llm.model)
    gemini_response = gemini_llm.generate(gemini_prompt)
    print(gemini_response)
    
    print("\n========== END OF RESPONSES ==========\n")
    
    # Basic assertions to verify responses were generated
    assert len(openai_response) > 0, "OpenAI response should not be empty"
    assert len(gemini_response) > 0, "Gemini response should not be empty"
    assert openai_response != gemini_response, "Responses should differ between models"

if __name__ == "__main__":
    # This allows the test to be run directly with python tests/test_llm.py
    test_llm_comparison()