import aiohttp
from urllib.parse import urljoin
from typing import Dict, List, Any, Optional, Callable
import logging
import uuid
logger = logging.getLogger("DeepSeekMCPAgentDAGExecutor")

MCP_SERVER_BASE = "http://localhost:8080"  # MCP服务器地址
class MCPClient:
    """MCP (Model Context Protocol) 客户端"""

    def __init__(self, base_url: str = MCP_SERVER_BASE):
        self.base_url = base_url
        self.session = None
        self.headers = {
            "Content-Type": "application/json"
        }

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def list_tools(self) -> List[Dict[str, Any]]:
        """获取可用工具列表"""
        url = urljoin(self.base_url, "/tools")
        try:
            async with self.session.get(url, headers=self.headers, timeout=10) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"获取MCP工具列表失败: {e}")
            return []

    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """调用MCP工具"""
        url = urljoin(self.base_url, f"/tools/{tool_name}")
        payload = {
            "parameters": parameters,
            "request_id": str(uuid.uuid4())
        }

        try:
            async with self.session.post(url, headers=self.headers, json=payload, timeout=30) as response:
                response.raise_for_status()
                result = await response.json()
                return result
        except Exception as e:
            logger.error(f"MCP工具调用失败: {e}")
            return {"status": "error", "message": str(e)}

    async def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """获取工具模式"""
        url = urljoin(self.base_url, f"/tools/{tool_name}/schema")
        try:
            async with self.session.get(url, headers=self.headers, timeout=10) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"获取工具模式失败: {e}")
            return None