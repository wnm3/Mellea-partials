import openai
from mellea.backends.openai import OpenAIBackend
from mellea.core import CBlock, Component
from mellea.stdlib.components import Document


class LMStudioBackend(OpenAIBackend):
    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:1234/v1",
        model_options: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            api_key="lm-studio",
            base_url=base_url,
            model_id=model_name,
            model_options=model_options,
            **kwargs,
        )

    @staticmethod
    def list_models(base_url: str = "http://localhost:1234/v1") -> list[str]:
        client = openai.OpenAI(api_key="lm-studio", base_url=base_url)
        response = client.models.list()
        return [model.id for model in response.data]


class FixedDocument(Document):
    """Fixed document"""

    def parts(self) -> list[Component | CBlock]:
        return []
