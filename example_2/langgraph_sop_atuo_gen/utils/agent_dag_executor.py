from src.utils.agent_dag_generator import DeepSeekAgentDAGGenerator,DeepSeekMCPAgentDAGGenerator
from src.utils.common import ResourceType
from src.utils.util import (DeepSeekClient,DeepSeekAgent,
                  DAGNode,ExecutionResult,TaskStatus,ResourceRequirement)
from src.utils.mcp_tools import MCPClient
import json
import networkx as nx
from typing import Dict, List, Any
import logging
import time
import random
import asyncio
import threading

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DeepSeekAgentDAGExecutor")

class DeepSeekAgentDAGExecutor:
    """DeepSeek Agent DAG执行器"""

    def __init__(self, dag_generator: DeepSeekAgentDAGGenerator, api_key: str):
        self.dag_generator = dag_generator
        self.api_key = api_key
        self.deepseek_client = DeepSeekClient(api_key)
        self.results = {}
        self.resource_pool = {
            ResourceType.LLM: 10.0,  # 总LLM资源
            ResourceType.CPU: 16.0,  # 总CPU核心
            ResourceType.MEMORY: 32.0,  # 总内存(GB)
            ResourceType.STORAGE: 100.0,  # 总存储(GB)
            ResourceType.NETWORK: 1000.0  # 总网络带宽(Mbps)
        }
        self.available_resources = self.resource_pool.copy()
        self.lock = threading.Lock()
        self.total_api_calls = 0

    async def execute_node(self, node: DAGNode, inputs: Dict[str, Any]) -> ExecutionResult:
        """执行单个节点"""
        start_time = time.time()
        node_id = node.id
        api_calls = 0

        try:
            # 检查资源是否足够
            if not self.check_resources(node.agent.resource_requirements):
                logger.warning(f"资源不足，等待执行节点 {node_id}")
                await asyncio.sleep(2)  # 等待资源释放
                if not self.check_resources(node.agent.resource_requirements):
                    return ExecutionResult(
                        node_id=node_id,
                        status=TaskStatus.FAILED,
                        error="资源不足，无法执行"
                    )

            # 分配资源
            self.allocate_resources(node.agent.resource_requirements)

            logger.info(f"开始执行节点: {node_id}")

            # 模拟工具调用
            tool_results = {}
            for tool in node.agent.tools:
                tool_result = self.dag_generator.mock_tool_call(tool, **inputs)
                tool_results[tool] = tool_result

            # 准备DeepSeek API调用
            prompt = self._build_prompt_for_agent(node.agent, inputs, tool_results)

            # 调用DeepSeek API
            async with self.deepseek_client as client:
                try:
                    response = await client.generate_text(
                        prompt=prompt,
                        system_prompt=node.agent.system_prompt,
                        temperature=0.7,
                        max_tokens=2000
                    )
                    api_calls += 1
                    self.total_api_calls += 1

                    # 解析响应
                    output = self._parse_agent_response(response, node.agent.output_schema)
                    output["tool_results"] = tool_results
                    output["processed_inputs"] = inputs

                except Exception as e:
                    logger.error(f"DeepSeek API调用失败: {e}")
                    # 失败时使用模拟输出作为备选
                    output = {
                        "processed_inputs": inputs,
                        "tool_results": tool_results,
                        "decision": random.choice(["approve", "reject", "need_more_info"]),
                        "confidence": random.uniform(0.7, 0.95),
                        "recommendations": [
                            f"建议{random.randint(1, 3)}: {random.choice(['加强沟通', '增加预算', '调整时间', '优化流程'])}"
                        ],
                        "fallback": True  # 标记为备选输出
                    }

            execution_time = time.time() - start_time

            # 释放资源
            self.release_resources(node.agent.resource_requirements)

            return ExecutionResult(
                node_id=node_id,
                status=TaskStatus.COMPLETED,
                output=output,
                execution_time=execution_time,
                resource_usage=self.calculate_resource_usage(node.agent.resource_requirements, execution_time),
                api_calls=api_calls
            )

        except Exception as e:
            execution_time = time.time() - start_time
            # 释放资源
            self.release_resources(node.agent.resource_requirements)

            return ExecutionResult(
                node_id=node_id,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time=execution_time,
                api_calls=api_calls
            )

    def _build_prompt_for_agent(self, agent: DeepSeekAgent, inputs: Dict[str, Any],
                                tool_results: Dict[str, Any]) -> str:
        """为Agent构建提示词"""
        prompt = f"""
        你是一个{agent.name}，负责{agent.description}。

        ## 输入信息:
        {json.dumps(inputs, ensure_ascii=False, indent=2)}

        ## 工具调用结果:
        {json.dumps(tool_results, ensure_ascii=False, indent=2)}

        ## 输出要求:
        请根据以上信息，生成符合以下格式的输出:
        {json.dumps(agent.output_schema, ensure_ascii=False, indent=2)}

        请直接输出JSON格式的数据，不要输出其他任何内容。
        """

        return prompt

    def _parse_agent_response(self, response: str, output_schema: Dict[str, Any]) -> Dict[str, Any]:
        """解析Agent响应"""
        try:
            # 尝试从响应中提取JSON
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                # 如果没有找到JSON，返回原始响应
                return {"raw_response": response, "parse_error": "No JSON found in response"}
        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败: {e}")
            return {"raw_response": response, "parse_error": str(e)}

    def check_resources(self, requirements: List[ResourceRequirement]) -> bool:
        """检查资源是否足够"""
        with self.lock:
            for req in requirements:
                if self.available_resources[req.resource_type] < req.min_amount:
                    return False
            return True

    def allocate_resources(self, requirements: List[ResourceRequirement]):
        """分配资源"""
        with self.lock:
            for req in requirements:
                # 分配平均资源量
                amount = (req.min_amount + req.max_amount) / 2
                self.available_resources[req.resource_type] -= amount

    def release_resources(self, requirements: List[ResourceRequirement]):
        """释放资源"""
        with self.lock:
            for req in requirements:
                # 释放平均资源量
                amount = (req.min_amount + req.max_amount) / 2
                self.available_resources[req.resource_type] += amount

    def calculate_resource_usage(self, requirements: List[ResourceRequirement], execution_time: float) -> Dict[
        ResourceType, float]:
        """计算资源使用量"""
        usage = {}
        for req in requirements:
            # 计算资源使用量 (资源量 * 时间)
            amount = (req.min_amount + req.max_amount) / 2
            usage[req.resource_type] = amount * execution_time
        return usage

    async def execute_dag(self, nodes: Dict[str, DAGNode]) -> Dict[str, ExecutionResult]:
        """执行整个DAG"""
        # 构建执行顺序
        G = nx.DiGraph()
        for node_id, node in nodes.items():
            G.add_node(node_id)
            for dep in node.dependencies:
                G.add_edge(dep, node_id)

        execution_order = list(nx.topological_sort(G))
        results = {}

        # 按顺序执行节点
        for node_id in execution_order:
            node = nodes[node_id]

            # 收集输入数据
            inputs = {}
            for dep_id in node.dependencies:
                if dep_id in results and results[dep_id].status == TaskStatus.COMPLETED:
                    inputs.update(results[dep_id].output.get('processed_inputs', {}))
                    inputs.update(results[dep_id].output.get('tool_results', {}))

            # 执行节点
            result = await self.execute_node(node, inputs)
            results[node_id] = result

            if result.status == TaskStatus.COMPLETED:
                logger.info(f"节点 {node_id} 执行成功，耗时 {result.execution_time:.2f}秒，API调用: {result.api_calls}次")
            else:
                logger.error(f"节点 {node_id} 执行失败: {result.error}")

        return results


class DeepSeekMCPAgentDAGExecutor:
    """DeepSeek Agent DAG执行器，支持MCP工具调用"""

    def __init__(self, dag_generator: DeepSeekMCPAgentDAGGenerator, api_key: str):
        self.dag_generator = dag_generator
        self.api_key = api_key
        self.deepseek_client = DeepSeekClient(api_key)
        self.mcp_client = MCPClient()
        self.results = {}
        self.resource_pool = {
            ResourceType.LLM: 10.0,  # 总LLM资源
            ResourceType.CPU: 16.0,  # 总CPU核心
            ResourceType.MEMORY: 32.0,  # 总内存(GB)
            ResourceType.STORAGE: 100.0,  # 总存储(GB)
            ResourceType.NETWORK: 1000.0  # 总网络带宽(Mbps)
        }
        self.available_resources = self.resource_pool.copy()
        self.lock = threading.Lock()
        self.total_api_calls = 0
        self.total_tool_calls = 0

    async def execute_node(self, node: DAGNode, inputs: Dict[str, Any]) -> ExecutionResult:
        """执行单个节点"""
        start_time = time.time()
        node_id = node.id
        api_calls = 0
        tool_calls = 0

        try:
            # 检查资源是否足够
            if not self.check_resources(node.agent.resource_requirements):
                logger.warning(f"资源不足，等待执行节点 {node_id}")
                await asyncio.sleep(2)  # 等待资源释放
                if not self.check_resources(node.agent.resource_requirements):
                    return ExecutionResult(
                        node_id=node_id,
                        status=TaskStatus.FAILED,
                        error="资源不足，无法执行"
                    )

            # 分配资源
            self.allocate_resources(node.agent.resource_requirements)

            logger.info(f"开始执行节点: {node_id}")

            # 调用MCP工具
            tool_results = {}
            async with self.mcp_client as mcp_client:
                for tool in node.agent.tools:
                    try:
                        tool_result = await mcp_client.call_tool(tool, inputs)
                        tool_results[tool] = tool_result
                        tool_calls += 1
                        self.total_tool_calls += 1
                        logger.info(f"成功调用MCP工具: {tool}")
                    except Exception as e:
                        logger.error(f"MCP工具调用失败: {tool}, 错误: {e}")
                        tool_results[tool] = {"status": "error", "message": str(e)}

            # 准备DeepSeek API调用
            prompt = self._build_prompt_for_agent(node.agent, inputs, tool_results)

            # 调用DeepSeek API
            async with self.deepseek_client as client:
                try:
                    response = await client.generate_text(
                        prompt=prompt,
                        system_prompt=node.agent.system_prompt,
                        temperature=0.7,
                        max_tokens=2000
                    )
                    api_calls += 1
                    self.total_api_calls += 1

                    # 解析响应
                    output = self._parse_agent_response(response, node.agent.output_schema)
                    output["tool_results"] = tool_results
                    output["processed_inputs"] = inputs

                except Exception as e:
                    logger.error(f"DeepSeek API调用失败: {e}")
                    # 失败时使用模拟输出作为备选
                    output = {
                        "processed_inputs": inputs,
                        "tool_results": tool_results,
                        "decision": random.choice(["approve", "reject", "need_more_info"]),
                        "confidence": random.uniform(0.7, 0.95),
                        "recommendations": [
                            f"建议{random.randint(1, 3)}: {random.choice(['加强沟通', '增加预算', '调整时间', '优化流程'])}"
                        ],
                        "fallback": True  # 标记为备选输出
                    }

            execution_time = time.time() - start_time

            # 释放资源
            self.release_resources(node.agent.resource_requirements)

            return ExecutionResult(
                node_id=node_id,
                status=TaskStatus.COMPLETED,
                output=output,
                execution_time=execution_time,
                resource_usage=self.calculate_resource_usage(node.agent.resource_requirements, execution_time),
                api_calls=api_calls,
                tool_calls=tool_calls
            )

        except Exception as e:
            execution_time = time.time() - start_time
            # 释放资源
            self.release_resources(node.agent.resource_requirements)

            return ExecutionResult(
                node_id=node_id,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time=execution_time,
                api_calls=api_calls,
                tool_calls=tool_calls
            )

    def _build_prompt_for_agent(self, agent: DeepSeekAgent, inputs: Dict[str, Any],
                                tool_results: Dict[str, Any]) -> str:
        """为Agent构建提示词"""
        prompt = f"""
        你是一个{agent.name}，负责{agent.description}。

        ## 输入信息:
        {json.dumps(inputs, ensure_ascii=False, indent=2)}

        ## 工具调用结果:
        {json.dumps(tool_results, ensure_ascii=False, indent=2)}

        ## 输出要求:
        请根据以上信息，生成符合以下格式的输出:
        {json.dumps(agent.output_schema, ensure_ascii=False, indent=2)}

        请直接输出JSON格式的数据，不要输出其他任何内容。
        """

        return prompt

    def _parse_agent_response(self, response: str, output_schema: Dict[str, Any]) -> Dict[str, Any]:
        """解析Agent响应"""
        try:
            # 尝试从响应中提取JSON
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                # 如果没有找到JSON，返回原始响应
                return {"raw_response": response, "parse_error": "No JSON found in response"}
        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败: {e}")
            return {"raw_response": response, "parse_error": str(e)}

    def check_resources(self, requirements: List[ResourceRequirement]) -> bool:
        """检查资源是否足够"""
        with self.lock:
            for req in requirements:
                if self.available_resources[req.resource_type] < req.min_amount:
                    return False
            return True

    def allocate_resources(self, requirements: List[ResourceRequirement]):
        """分配资源"""
        with self.lock:
            for req in requirements:
                # 分配平均资源量
                amount = (req.min_amount + req.max_amount) / 2
                self.available_resources[req.resource_type] -= amount

    def release_resources(self, requirements: List[ResourceRequirement]):
        """释放资源"""
        with self.lock:
            for req in requirements:
                # 释放平均资源量
                amount = (req.min_amount + req.max_amount) / 2
                self.available_resources[req.resource_type] += amount

    def calculate_resource_usage(self, requirements: List[ResourceRequirement], execution_time: float) -> Dict[
        ResourceType, float]:
        """计算资源使用量"""
        usage = {}
        for req in requirements:
            # 计算资源使用量 (资源量 * 时间)
            amount = (req.min_amount + req.max_amount) / 2
            usage[req.resource_type] = amount * execution_time
        return usage

    async def execute_dag(self, nodes: Dict[str, DAGNode]) -> Dict[str, ExecutionResult]:
        """执行整个DAG"""
        # 构建执行顺序
        G = nx.DiGraph()
        for node_id, node in nodes.items():
            G.add_node(node_id)
            for dep in node.dependencies:
                G.add_edge(dep, node_id)

        execution_order = list(nx.topological_sort(G))
        results = {}

        # 按顺序执行节点
        for node_id in execution_order:
            node = nodes[node_id]

            # 收集输入数据
            inputs = {}
            for dep_id in node.dependencies:
                if dep_id in results and results[dep_id].status == TaskStatus.COMPLETED:
                    inputs.update(results[dep_id].output.get('processed_inputs', {}))
                    inputs.update(results[dep_id].output.get('tool_results', {}))

            # 执行节点
            result = await self.execute_node(node, inputs)
            results[node_id] = result

            if result.status == TaskStatus.COMPLETED:
                logger.info(
                    f"节点 {node_id} 执行成功，耗时 {result.execution_time:.2f}秒，API调用: {result.api_calls}次，工具调用: {result.tool_calls}次")
            else:
                logger.error(f"节点 {node_id} 执行失败: {result.error}")

        return results