import requests
from openai import OpenAI
class Render():
    def __init__(self,
                 config: dict,
                 ):
        self.render_provider = config.get("render_provider")
        self.api_key = config.get("api_key")
        self.api_endpoint = config.get("api_endpoint")
        self.width = config.get("width")
        self.height = config.get("height")
        self.model = config.get("model")
        self.guidance_scale = config.get("guidance_scale")
        self.openai = OpenAI(api_key=self.api_key)
        
    def generate(self, 
                 prompt: str,
                 ) -> str:
        if self.render_provider == "deepinfra":
            return self._deepinfra_render(prompt)
        elif self.render_provider == "openai":
            return self._openai_render(prompt)
        else:
            raise ValueError(f"Unsupported render provider: {self.render_provider}")

    def _deepinfra_render(self, prompt: str) -> str:
        headers = {
            "Authorization": f"bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {'input': 
                {'prompt': prompt, 
                 'width': self.width,
                 'height': self.height,
                 'guidance_scale': self.guidance_scale,
                 }
        }
        response = requests.post(self.api_endpoint, 
                                 json=data, 
                                 headers=headers
        )
        print(response.json())
        response.raise_for_status()
        return response.json().get('items')[0]

    def _openai_render(self, prompt: str) -> str:
        response = self.openai.images.generate(
            model=self.model,
            prompt=prompt,
            size=f"{self.width}x{self.height}",
            quality="standard",
            n=1,
            )

        image_url = response.data[0].url
        return image_url