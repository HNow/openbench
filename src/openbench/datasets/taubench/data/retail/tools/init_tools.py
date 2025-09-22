from .calculate import calculate
from inspect_ai.tool import Tool
from typing import List, Type

TOOLS: List[Type[Tool]] = [calculate]