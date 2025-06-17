#!/usr/bin/env python3
"""
AISuite Client Class
A simple wrapper for AISuite with configurable prompts and response handling
"""

import aisuite as ai
from .constants import DEFAULT_MODEL, DEFAULT_SYSTEM_PROMPT, DEFAULT_TEMPERATURE, OLLAMA_TIMEOUT


class AISuiteClient:
    def __init__(self, model=DEFAULT_MODEL, system_prompt=DEFAULT_SYSTEM_PROMPT, temperature=DEFAULT_TEMPERATURE):
        """
        Initialize the AISuite client
        
        Args:
            model (str): Model to use in format "provider:model-name"
            system_prompt (str): System prompt for the AI
            temperature (float): Temperature for response generation
        """
        self.client = ai.Client()
        self.client.configure({
            "ollama": {
                "timeout": OLLAMA_TIMEOUT,
            }
        })
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
    
    def set_system_prompt(self, prompt):
        """Set the system prompt"""
        self.system_prompt = prompt
    
    def set_model(self, model):
        """Set the model to use"""
        self.model = model
    
    def respond(self, user_prompt, print_response=True):
        """
        Get response from the AI model
        
        Args:
            user_prompt (str): User's message/question
            print_response (bool): Whether to print the response
            
        Returns:
            str: The AI's response
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            )
            
            ai_response = response.choices[0].message.content
            
            if print_response:
                print("--------------------------------")
                print("AI Response:")
                print(ai_response)
                print("--------------------------------")
            
            return ai_response
            
        except Exception as e:
            error_msg = f"Error: {e}"
            if print_response:
                print(error_msg)
            return error_msg


def main():
    """Test the AISuiteClient class"""
    print("Testing AISuiteClient...")
    
    # Create client
    client = AISuiteClient()
    
    # Test basic response
    response = client.respond("Hello! Tell me a short joke.", print_response=True)
    print(f"Returned: {len(response)} characters")
    
    # Test with custom system prompt
    client.set_system_prompt("You are a helpful coding assistant. Be concise.")
    response = client.respond("What is Python?", print_response=False)
    print(f"Silent response: {response[:50]}...")


if __name__ == "__main__":
    main() 