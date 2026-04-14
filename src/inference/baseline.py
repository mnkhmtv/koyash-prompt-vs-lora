import ollama

class BaselineKoyashLLM:
    """Qwen3-4b baseline model"""
    def __init__(self,
                 system_prompt: str = "Ты - Koyash Ассистент", 
                 temperature: float = 0.7):
        self.name = f"BaselineKoyashLLM: qwen3:4b"
        self.system_prompt = system_prompt
        self.temperature = temperature

    def get_response(self, user_prompt: str, temperature: float = None, system_prompt: str = None) -> ollama.ChatResponse:
        response = ollama.chat(
            model="qwen3:4b",
            messages=[
                {"role": "system", "content": system_prompt if system_prompt is not None else self.system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            options={"temperature": temperature if temperature is not None else self.temperature},
            logprobs=True,
        )

        return response

    def get_answer(self, response : ollama.ChatResponse) -> str:
        return response['message']['content']
