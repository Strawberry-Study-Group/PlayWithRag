import requests
from openai import OpenAI
import replicate
import os
from urllib.parse import urlparse
import base64

class ImgRender():
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
        self.config = config
        if self.render_provider == "openai":
            self.openai = OpenAI(api_key=self.api_key)
        if self.render_provider == "replicate":
            os.environ["REPLICATE_API_TOKEN"] = self.api_key
        
    def generate(self, 
                 prompt: str,
                 ) -> str:
        if self.render_provider == "deepinfra":
            return self._deepinfra_render(prompt)
        elif self.render_provider == "openai":
            return self._openai_render(prompt)
        elif self.render_provider == "replicate":
            return self._replicate_render(prompt)
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
    
    def _replicate_render(self, prompt: str) -> str:
        output = replicate.run(
            self.model,
            input={
                "width": self.width,
                "height": self.height,
                "prompt": prompt,
                "scheduler": self.config.get("scheduler", "KarrasDPM"),
                "num_outputs": self.config.get("num_outputs",1),
                "guidance_scale": self.guidance_scale,
                "apply_watermark": self.config.get("apply_watermark", False),
                "negative_prompt": self.config.get("negative_prompt", "lowest quality, low quality"),
                "prompt_strength": self.config.get("prompt_strength", 0.7),
                "num_inference_steps": self.config.get("num_inference_steps", 20),
            }
        )

        # Assuming the output is a list of image URLs, return the first one
        if isinstance(output, list) and len(output) > 0:
            return output[0]
        else:
            raise ValueError("Unexpected output format from Replicate API")
        

class ImgRenderWithReferanceImg():
    def __init__(self,
                 config: dict,
                 ):
        self.render_provider = config.get("render_provider")
        self.api_key = config.get("api_key")
        if self.render_provider == "replicate":
            os.environ["REPLICATE_API_TOKEN"] = self.api_key
        else:
            raise ValueError(f"Unsupported render provider: {self.render_provider}")
        
        self.api_endpoint = config.get("api_endpoint")
        self.width = config.get("width")
        self.height = config.get("height")
        self.model = config.get("model")
        print(self.model)
        self.guidance_scale = config.get("guidance_scale")
        self.config = config

    def generate(self, 
                 prompt: str,
                 referance_img: str = "None",
                 ) -> str:
        if self.render_provider == "replicate" and "flux" not in self.model:
            if referance_img == "None" or referance_img is None:
                raise ValueError("Reference image is required for this API")
            return self._replicate_render(prompt, referance_img)
        if self.render_provider == "replicate" and "flux" in self.model:
            return self._replicate_flux_render(prompt, referance_img)
        else:
            raise ValueError(f"Unsupported render provider: {self.render_provider}")
        




    def _replicate_render(self, prompt: str, reference_img: str) -> str:
        def is_url(url):
            try:
                result = urlparse(url)
                return all([result.scheme, result.netloc])
            except ValueError:
                return False

        if is_url(reference_img):
            # If it's a URL, use it directly
            image_input = reference_img
        else:
            # If it's a local file, convert to base64 data URI
            if os.path.isfile(reference_img):
                with open(reference_img, "rb") as file:
                    file_content = file.read()
                    base64_encoded = base64.b64encode(file_content).decode('utf-8')
                    file_extension = os.path.splitext(reference_img)[1][1:]  # Get file extension without the dot
                    mime_type = f"image/{file_extension}"
                    image_input = f"data:{mime_type};base64,{base64_encoded}"
            else:
                raise ValueError(f"Invalid reference image path: {reference_img}")

        ref_img_key = ""
        if "kolors" in self.model:
            ref_img_key = "image"
        elif "photomaker" in self.model:
            ref_img_key = "input_image"

        output = replicate.run(
            self.model,
            input={
                "width": self.width,
                "height": self.height,
                "prompt": prompt,
                "cfg": self.config.get("cfg", 4),
                ref_img_key: image_input,
                "scheduler": self.config.get("scheduler", "karras"),
                "sampler": self.config.get("sampler", "dpmpp_2m_sde_gpu"),
                "output_format": self.config.get("output_format", "jpg"),
                "oupit_quality": self.config.get("output_quality", 80),
                "number_of_images": self.config.get("number_of_images", 1),
                "num_outputs": self.config.get("num_outputs", 1),
                "guidance_scale": self.config.get("guidance_scale", 5),
                "ip_adapter_weight": self.config.get("ip_adapter_weight", 1),
                "ip_adapter_weight_type": self.config.get("ip_adapter_weight_type", "style transfer precise"),
                "apply_watermark": self.config.get("apply_watermark", False),
                "negative_prompt": self.config.get("negative_prompt", "lowest quality, low quality"),
                "prompt_strength": self.config.get("prompt_strength", 0.7),
                "num_inference_steps": self.config.get("num_inference_steps", 20),
                "style_strength_ratio": self.config.get("style_strength_ratio", 20),
                "style_name": self.config.get("style_name","Photographic (Default)"),
                "num_steps": self.config.get("num_steps", 50),
            }
        )

        # Assuming the output is a list of image URLs, return the first one
        if isinstance(output, list) and len(output) > 0:
            return output[0]
        else:
            raise ValueError("Unexpected output format from Replicate API")


    def _replicate_flux_render(self, prompt: str, reference_img: str = "None") -> str:
        def is_url(url):
            try:
                result = urlparse(url)
                return all([result.scheme, result.netloc])
            except ValueError:
                return False

        if reference_img != "None":
            if is_url(reference_img):
                # If it's a URL, use it directly
                image_input = reference_img
            else:
                # If it's a local file, convert to base64 data URI
                if os.path.isfile(reference_img):
                    with open(reference_img, "rb") as file:
                        file_content = file.read()
                        base64_encoded = base64.b64encode(file_content).decode('utf-8')
                        file_extension = os.path.splitext(reference_img)[1][1:]  # Get file extension without the dot
                        mime_type = f"image/{file_extension}"
                        image_input = f"data:{mime_type};base64,{base64_encoded}"
                else:
                    raise ValueError(f"Invalid reference image path: {reference_img}")
        else:
            image_input = "None"
        
        input_data = {
                "prompt": prompt,
                "guidance": self.config.get("guidance", 3.5),
                "num_outputs": self.config.get("num_outputs", 1),
                "aspect_ratio": self.config.get("aspect_ratio", "1:1"),
                "output_format": self.config.get("output_format", "jpg"),
                "output_quality": self.config.get("output_quality", 80),
                "prompt_strength": self.config.get("prompt_strength", 0.8),
                "disable_safety_checker": self.config.get("disable_safety_checker", False),
            }
        
        if image_input != "None":
            input_data["image"] = image_input

        output = replicate.run(
            self.model,
            input=input_data
        )

        # Assuming the output is a list of image URLs, return the first one
        if isinstance(output, list) and len(output) > 0:
            return output[0]
        else:
            raise ValueError("Unexpected output format from Replicate API")
