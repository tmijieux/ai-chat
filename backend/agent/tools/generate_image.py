import base64
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .base import BaseTool, tool_error, tool_rejected
from tool_result_types import GenerateImageResult, ToolResult

if TYPE_CHECKING:
    from agent.agent import AgentSession


class GenerateImageTool(BaseTool):
    name = "generate_image"
    description = (
        "Generate an image from a text prompt using the local Flux.1-schnell model. "
        "Slow (roughly 30-60 seconds): the local chat model is stopped to free GPU memory "
        "for image generation, then restarted afterward. Use only when the user actually "
        "wants a generated image, not for editing or analyzing existing images."
    )
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Text description of the image to generate.",
            },
        },
        "required": ["prompt"],
    }
    requires_confirmation = True
    measured_delta = 330

    def make_validation_text_for_user_confirmation(self, args: dict) -> str:
        return f"GENERATE IMAGE: {args.get('prompt', '')}"

    async def execute(self, args: dict, session: "AgentSession", working_directory: str | None) -> ToolResult:
        prompt = args.get("prompt", "")

        preview = self.make_validation_text_for_user_confirmation(args)
        approved, user_msg = await session.request_confirm(f"generate-image-{prompt}", self.name, args, preview)
        if not approved:
            return tool_rejected(self.name, user_msg)

        import imagegen_pipeline
        from database import AsyncSessionLocal
        import tables as db
        from llm import backend
        from llm.llama_server import LlamaServerBackend

        if isinstance(backend, LlamaServerBackend):
            await backend.stop()
        try:
            result = await imagegen_pipeline.generate(prompt)
        except Exception as e:
            return tool_error(self.name, f"Image generation failed: {e}")
        finally:
            if isinstance(backend, LlamaServerBackend):
                await backend.ensure_running()

        image_id = str(uuid.uuid4())
        async with AsyncSessionLocal() as sess:
            sess.add(db.Image(
                id=image_id,
                mime_type="image/png",
                data=base64.b64encode(result.png_bytes).decode("ascii"),
                width=result.width,
                height=result.height,
                created_at=datetime.now(timezone.utc).isoformat(),
            ))
            await sess.commit()

        return GenerateImageResult(
            tool=self.name,
            status="success",
            prompt=prompt,
            image_id=image_id,
        )
