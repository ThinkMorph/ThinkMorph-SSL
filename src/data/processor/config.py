from typing import Any, Dict, Optional

from pydantic import BaseModel


class ProcessorConfig(BaseModel):
    processor_name: str
    processor_type: str
    extra_kwargs: Optional[Dict[str, Any]] = None
