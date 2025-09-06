from typing import Dict, List, Any, Optional
import aiohttp
import logging

from dataclasses import dataclass, field

from src.utils.common import ResourceType,AgentType,TaskStatus

# DeepSeek API配置
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"  # 根据实际情况调整模型名称
DEEPSEEK_KEY = "sk-350c789908d542b6a7a45bef1c212a98" ##task_招聘_ak

logger = logging.getLogger("DeepSeekAgentDAGExecutor")

@dataclass
class ResourceRequirement:
    """资源需求"""
    resource_type: ResourceType
    min_amount: float
    max_amount: float
    priority: int = 1

@dataclass
class DeepSeekAgent:
    """DeepSeek Agent定义"""
    id: str
    name: str
    agent_type: AgentType
    description: str
    model: str = DEEPSEEK_MODEL  # 使用DeepSeek模型
    resource_requirements: List[ResourceRequirement] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)  # 可调用的工具列表
    input_schema: Dict[str, Any] = field(default_factory=dict)  # 输入数据格式
    output_schema: Dict[str, Any] = field(default_factory=dict)  # 输出数据格式
    system_prompt: str = ""  # 系统提示词


@dataclass
class DAGNode:
    """DAG节点"""
    id: str
    agent: DeepSeekAgent
    inputs: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # 依赖的节点ID列表
    timeout: int = 300  # 超时时间（秒）
    retry_count: int = 3  # 重试次数


@dataclass
class ExecutionResult:
    """执行结果"""
    node_id: str
    status: TaskStatus
    output: Optional[Any] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None  # 执行时间（秒）
    resource_usage: Optional[Dict[ResourceType, float]] = None  # 资源使用情况
    api_calls: int = 0  # API调用次数
    tool_calls: int = 0  # 工具调用次数

class DeepSeekClient:
    """DeepSeek API客户端"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = None
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def chat_completion(self, messages: List[Dict[str, str]], model: str = DEEPSEEK_MODEL,
                              temperature: float = 0.7, max_tokens: int = 2000) -> Dict[str, Any]:
        """调用DeepSeek聊天补全API"""
        if not self.session:
            self.session = aiohttp.ClientSession()

        url = f"{DEEPSEEK_API_BASE}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            async with self.session.post(url, headers=self.headers, json=payload, timeout=30) as response:
                response.raise_for_status()
                result = await response.json()
                return result
        except Exception as e:
            logger.error(f"DeepSeek API调用失败: {e}")
            raise

    async def generate_text(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        """生成文本"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self.chat_completion(messages, **kwargs)
        return response["choices"][0]["message"]["content"]