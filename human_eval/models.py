from openai import OpenAI

# TODO: Delete before push!!!
SS_API_KEY = ''
SS_TOG_API_KEY = ''
client = OpenAI(api_key=SS_API_KEY)
togClient = OpenAI(api_key=SS_TOG_API_KEY, base_url='https://api.together.xyz')


"""
Object to deal with API calls to OAI models.
"""
class GPT:
    """
    Params
    - model | model to make api call to
    - system_prompt | system_prompt to use for model
    """
    def __init__(self,
                 model="gpt-3.5-turbo",
                 system_prompt="You are a Question Answering portal."):
        self.model = model
        self.system_prompt = system_prompt

    def answer(self, prompt: str):
        completion = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        )

        return completion

    def answer_txt(self, prompt: str) -> str:
        completion = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        return completion.choices[0].message.content


"""
Handles calls to tgoether ai models
"""
class TogModel:
    def __init__(self,
                 model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                 system_prompt="You are a Question Answering portal.",
                 max_tokens=1024):
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens

    def answer(self, prompt: str):
        completion = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        )

        return completion

    def answer_txt(self, prompt: str) -> str:
        completion = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        return completion.choices[0].message.content