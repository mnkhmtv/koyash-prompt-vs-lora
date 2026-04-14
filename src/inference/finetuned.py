import ollama


class FinetunedKoyashLLM:
    """finetuned on the train set of the Koyash dataset"""

    def __init__(self,
                 system_prompt: str = "Ты - Koyash Ассистент",
                 temperature: float = 0.7,
                 model_tag: str = "koyash-finetuned:latest"):
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.model_tag = model_tag
        self.name = f"FinetunedKoyashLLM: {model_tag}"

    def get_response(self, user_prompt: str, temperature: float = None, system_prompt: str = None) -> ollama.ChatResponse:
        return ollama.chat(
            model=self.model_tag,
            messages=[
                {"role": "system", "content": system_prompt if system_prompt is not None else self.system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            options={"temperature": temperature if temperature is not None else self.temperature},
            logprobs=True,
        )

    def get_answer(self, response: ollama.ChatResponse) -> str:
        return response['message']['content']
