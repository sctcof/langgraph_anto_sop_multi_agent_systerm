import yaml
import json
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import logging
from datetime import datetime, timedelta
import time
import random
import asyncio
import threading
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LLMAgentDAGExecutor")


class AgentType(Enum):
    """大模型Agent类型"""
    PLANNER = "planner"  # 规划型Agent
    ANALYZER = "analyzer"  # 分析型Agent
    EXECUTOR = "executor"  # 执行型Agent
    VALIDATOR = "validator"  # 验证型Agent
    DECISION = "decision"  # 决策型Agent


class ResourceType(Enum):
    """资源类型"""
    LLM = "llm"  # 大模型资源
    CPU = "cpu"  # CPU资源
    MEMORY = "memory"  # 内存资源
    STORAGE = "storage"  # 存储资源
    NETWORK = "network"  # 网络资源


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ResourceRequirement:
    """资源需求"""
    resource_type: ResourceType
    min_amount: float
    max_amount: float
    priority: int = 1


@dataclass
class LLMAgent:
    """大模型Agent定义"""
    id: str
    name: str
    agent_type: AgentType
    description: str
    model: str  # 使用的大模型类型
    resource_requirements: List[ResourceRequirement]
    tools: List[str]  # 可调用的工具列表
    input_schema: Dict[str, Any]  # 输入数据格式
    output_schema: Dict[str, Any]  # 输出数据格式


@dataclass
class DAGNode:
    """DAG节点"""
    id: str
    agent: LLMAgent
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


class LLMAgentDAGGenerator:
    """从YAML生成大模型Agent DAG"""

    def __init__(self):
        # 预定义的Agent模板
        self.agent_templates = {
            "position_planning": LLMAgent(
                id="position_planner",
                name="职位规划Agent",
                agent_type=AgentType.PLANNER,
                description="负责职位需求和规划的Agent",
                model="gpt-4",
                resource_requirements=[
                    ResourceRequirement(ResourceType.LLM, 1, 5, 2),
                    ResourceRequirement(ResourceType.CPU, 2, 4, 1)
                ],
                tools=["market_analysis", "budget_calculator", "role_definition"],
                input_schema={"department": "str", "business_needs": "str"},
                output_schema={"position_title": "str", "requirements": "list", "budget_range": "dict"}
            ),
            "jd_creation": LLMAgent(
                id="jd_creator",
                name="职位描述创建Agent",
                agent_type=AgentType.EXECUTOR,
                description="负责创建职位描述的Agent",
                model="claude-2",
                resource_requirements=[
                    ResourceRequirement(ResourceType.LLM, 1, 3, 2),
                    ResourceRequirement(ResourceType.CPU, 1, 2, 1)
                ],
                tools=["jd_template", "skill_extractor", "industry_benchmark"],
                input_schema={"position_title": "str", "requirements": "list"},
                output_schema={"jd_content": "str", "key_skills": "list", "experience_level": "str"}
            ),
            "candidate_screening": LLMAgent(
                id="candidate_screener",
                name="候选人筛选Agent",
                agent_type=AgentType.ANALYZER,
                description="负责筛选候选人的Agent",
                model="llama-2",
                resource_requirements=[
                    ResourceRequirement(ResourceType.LLM, 2, 4, 2),
                    ResourceRequirement(ResourceType.CPU, 2, 4, 1),
                    ResourceRequirement(ResourceType.MEMORY, 4, 8, 1)
                ],
                tools=["resume_parser", "skill_matcher", "experience_evaluator"],
                input_schema={"jd_content": "str", "candidate_profiles": "list"},
                output_schema={"qualified_candidates": "list", "rejection_reasons": "dict", "ranking": "list"}
            ),
            "interview_planning": LLMAgent(
                id="interview_planner",
                name="面试规划Agent",
                agent_type=AgentType.PLANNER,
                description="负责规划面试流程的Agent",
                model="gpt-4",
                resource_requirements=[
                    ResourceRequirement(ResourceType.LLM, 1, 3, 2),
                    ResourceRequirement(ResourceType.CPU, 1, 2, 1)
                ],
                tools=["calendar_integration", "interview_question_generator", "evaluation_criteria"],
                input_schema={"candidates": "list", "interviewers": "list"},
                output_schema={"schedule": "dict", "questions": "list", "evaluation_form": "dict"}
            ),
            "offer_decision": LLMAgent(
                id="offer_decider",
                name="录用决策Agent",
                agent_type=AgentType.DECISION,
                description="负责做出录用决策的Agent",
                model="gpt-4",
                resource_requirements=[
                    ResourceRequirement(ResourceType.LLM, 2, 4, 3),
                    ResourceRequirement(ResourceType.CPU, 2, 4, 2)
                ],
                tools=["compensation_analyzer", "market_comparison", "risk_assessment"],
                input_schema={"candidate_evaluations": "dict", "budget_constraints": "dict"},
                output_schema={"offer_decisions": "dict", "compensation_packages": "dict", "start_dates": "dict"}
            )
        }

        # 工具模拟函数映射
        self.tool_functions = {
            "market_analysis": self.mock_market_analysis,
            "budget_calculator": self.mock_budget_calculator,
            "role_definition": self.mock_role_definition,
            "jd_template": self.mock_jd_template,
            "skill_extractor": self.mock_skill_extractor,
            "industry_benchmark": self.mock_industry_benchmark,
            "resume_parser": self.mock_resume_parser,
            "skill_matcher": self.mock_skill_matcher,
            "experience_evaluator": self.mock_experience_evaluator,
            "calendar_integration": self.mock_calendar_integration,
            "interview_question_generator": self.mock_interview_question_generator,
            "evaluation_criteria": self.mock_evaluation_criteria,
            "compensation_analyzer": self.mock_compensation_analyzer,
            "market_comparison": self.mock_market_comparison,
            "risk_assessment": self.mock_risk_assessment
        }

    def parse_yaml(self, yaml_content: str) -> Dict[str, Any]:
        """解析YAML内容"""
        return yaml.safe_load(yaml_content)

    def generate_agents_from_yaml(self, yaml_data: Dict[str, Any]) -> Dict[str, LLMAgent]:
        """从YAML数据生成Agent"""
        agents = {}

        for task in yaml_data.get('tasks', []):
            task_name = task.get('name', '')
            task_id = task.get('id', '')

            # 根据任务名称选择合适的Agent模板
            agent_key = None
            if '审批' in task_name:
                agent_key = "position_planning"
            elif '描述' in task_name:
                agent_key = "jd_creation"
            elif '筛选' in task_name:
                agent_key = "candidate_screening"
            elif '安排' in task_name or '面试' in task_name:
                agent_key = "interview_planning"
            elif '录用' in task_name or '决策' in task_name:
                agent_key = "offer_decision"
            else:
                # 默认使用规划Agent
                agent_key = "position_planning"

            if agent_key in self.agent_templates:
                # 创建新的Agent实例
                template = self.agent_templates[agent_key]
                agent = LLMAgent(
                    id=f"{agent_key}_{task_id}",
                    name=f"{task_name} Agent",
                    agent_type=template.agent_type,
                    description=template.description,
                    model=template.model,
                    resource_requirements=template.resource_requirements.copy(),
                    tools=template.tools.copy(),
                    input_schema=template.input_schema.copy(),
                    output_schema=template.output_schema.copy()
                )
                agents[task_id] = agent

        return agents

    def build_dag_from_yaml(self, yaml_data: Dict[str, Any], agents: Dict[str, LLMAgent]) -> Dict[str, DAGNode]:
        """从YAML数据构建DAG"""
        nodes = {}
        task_id_to_node_id = {}

        # 第一遍：创建所有节点
        for task in yaml_data.get('tasks', []):
            task_id = task.get('id', '')
            if task_id in agents:
                node_id = f"node_{task_id}"
                task_id_to_node_id[task_id] = node_id

                # 确定依赖关系
                dependencies = []
                prerequisites = task.get('prerequisites', [])
                for prereq in prerequisites:
                    # 查找前置任务ID
                    for t in yaml_data.get('tasks', []):
                        if any(output in prereq for output in t.get('outputs', [])):
                            if t.get('id') in task_id_to_node_id:
                                dependencies.append(task_id_to_node_id[t.get('id')])

                # 创建DAG节点
                nodes[node_id] = DAGNode(
                    id=node_id,
                    agent=agents[task_id],
                    inputs={},  # 输入将在执行时动态填充
                    dependencies=dependencies
                )

        return nodes

    def generate_dag_visualization(self, nodes: Dict[str, DAGNode], output_path: str = "dag_visualization.png"):
        """生成DAG可视化图表"""
        G = nx.DiGraph()

        # 添加节点
        for node_id, node in nodes.items():
            G.add_node(node_id, label=f"{node.agent.name}\n({node.agent.agent_type.value})")

        # 添加边
        for node_id, node in nodes.items():
            for dep in node.dependencies:
                G.add_edge(dep, node_id)

        # 绘制图形
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=2, iterations=50)

        # 根据Agent类型设置节点颜色
        node_colors = []
        for node_id in G.nodes():
            agent_type = nodes[node_id].agent.agent_type
            if agent_type == AgentType.PLANNER:
                node_colors.append('lightblue')
            elif agent_type == AgentType.ANALYZER:
                node_colors.append('lightgreen')
            elif agent_type == AgentType.EXECUTOR:
                node_colors.append('lightyellow')
            elif agent_type == AgentType.VALIDATOR:
                node_colors.append('lightcoral')
            elif agent_type == AgentType.DECISION:
                node_colors.append('plum')
            else:
                node_colors.append('lightgray')

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000)
        nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20)
        nx.draw_networkx_labels(G, pos, nx.get_node_attributes(G, 'label'), font_size=8)

        plt.title("LLM Agent DAG Execution Flow")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"DAG可视化图表已保存到: {output_path}")

        # 生成DAG的文本描述
        dag_description = self.generate_dag_description(G, nodes)
        return dag_description

    def generate_dag_description(self, G: nx.DiGraph, nodes: Dict[str, DAGNode]) -> str:
        """生成DAG的文本描述"""
        description = "LLM Agent DAG 执行流程:\n\n"

        # 按层级组织节点
        levels = {}
        for node_id in nx.topological_sort(G):
            level = 0
            for pred in G.predecessors(node_id):
                if pred in levels:
                    level = max(level, levels[pred] + 1)
            levels[node_id] = level

        # 按层级输出节点信息
        max_level = max(levels.values()) if levels else 0
        for level in range(max_level + 1):
            description += f"=== 层级 {level} ===\n"
            for node_id, lvl in levels.items():
                if lvl == level:
                    node = nodes[node_id]
                    description += f"- {node.agent.name} ({node.agent.agent_type.value})\n"
                    description += f"  模型: {node.agent.model}\n"
                    description += f"  工具: {', '.join(node.agent.tools)}\n"
                    if node.dependencies:
                        description += f"  依赖: {', '.join(node.dependencies)}\n"
                    description += "\n"

        return description

    def mock_market_analysis(self, **kwargs):
        """模拟市场分析工具"""
        time.sleep(random.uniform(0.5, 2.0))
        return {
            "market_demand": random.randint(70, 95),
            "salary_benchmark": random.randint(8000, 20000),
            "competition_level": random.choice(["低", "中", "高"])
        }

    def mock_budget_calculator(self, **kwargs):
        """模拟预算计算工具"""
        time.sleep(random.uniform(0.3, 1.5))
        base_salary = random.randint(10000, 25000)
        return {
            "base_salary": base_salary,
            "bonus": round(base_salary * random.uniform(0.1, 0.3)),
            "benefits": round(base_salary * random.uniform(0.15, 0.25)),
            "total_compensation": round(base_salary * random.uniform(1.25, 1.55))
        }

    def mock_role_definition(self, **kwargs):
        """模拟角色定义工具"""
        time.sleep(random.uniform(0.5, 1.5))
        return {
            "key_responsibilities": [
                "负责系统架构设计和核心模块开发",
                "指导团队成员进行技术开发",
                "参与技术决策和代码审查"
            ],
            "required_skills": ["Python", "分布式系统", "数据库设计", "云平台"],
            "experience_level": random.choice(["中级", "高级", "专家"])
        }

    def mock_jd_template(self, **kwargs):
        """模拟JD模板工具"""
        time.sleep(random.uniform(0.5, 1.5))
        position_title = kwargs.get('position_title', '软件工程师')
        return {
            "jd_content": f"""
            # {position_title} 职位描述

            ## 职位职责
            1. 负责系统架构设计和核心模块开发
            2. 指导团队成员进行技术开发
            3. 参与技术决策和代码审查

            ## 职位要求
            1. 本科及以上学历，计算机相关专业
            2. 3年以上相关工作经验
            3. 熟练掌握Python/Java等编程语言
            4. 有分布式系统开发经验者优先
            """,
            "status": "completed"
        }

    def mock_skill_extractor(self, **kwargs):
        """模拟技能提取工具"""
        time.sleep(random.uniform(0.5, 1.5))
        return {
            "key_skills": ["Python", "Java", "分布式系统", "数据库", "云平台"],
            "skill_levels": {
                "Python": random.randint(3, 5),
                "Java": random.randint(2, 4),
                "分布式系统": random.randint(3, 5),
                "数据库": random.randint(3, 4),
                "云平台": random.randint(2, 4)
            }
        }

    def mock_industry_benchmark(self, **kwargs):
        """模拟行业基准工具"""
        time.sleep(random.uniform(0.5, 1.5))
        return {
            "industry_average_salary": random.randint(15000, 25000),
            "top_companies": ["阿里巴巴", "腾讯", "字节跳动", "华为", "百度"],
            "skill_demand": {
                "Python": random.randint(70, 90),
                "Java": random.randint(60, 80),
                "分布式系统": random.randint(75, 85),
                "数据库": random.randint(65, 75),
                "云平台": random.randint(70, 80)
            }
        }

    def mock_resume_parser(self, **kwargs):
        """模拟简历解析工具"""
        time.sleep(random.uniform(0.5, 1.5))
        return {
            "parsed_resumes": [
                {
                    "name": f"候选人{random.randint(1, 100)}",
                    "experience": random.randint(2, 10),
                    "skills": random.sample(["Python", "Java", "C++", "JavaScript", "Go", "Rust"], 3),
                    "education": random.choice(["本科", "硕士", "博士"]),
                    "score": random.randint(60, 95)
                }
                for _ in range(random.randint(3, 8))
            ],
            "status": "completed"
        }

    def mock_skill_matcher(self, **kwargs):
        """模拟技能匹配工具"""
        time.sleep(random.uniform(0.5, 1.5))
        jd_content = kwargs.get('jd_content', '')
        candidate_profiles = kwargs.get('candidate_profiles', [])

        matched_candidates = []
        for candidate in candidate_profiles[:random.randint(2, 5)]:  # 随机选择部分候选人
            match_score = random.randint(70, 95)
            matched_candidates.append({
                **candidate,
                "match_score": match_score,
                "status": "推荐" if match_score > 80 else "待定"
            })

        return {
            "matched_candidates": matched_candidates,
            "matching_criteria": ["技能匹配度", "经验匹配度", "教育背景匹配度"],
            "status": "completed"
        }

    def mock_experience_evaluator(self, **kwargs):
        """模拟经验评估工具"""
        time.sleep(random.uniform(0.5, 1.5))
        return {
            "evaluation_results": [
                {
                    "candidate_id": random.randint(1, 100),
                    "experience_score": random.randint(70, 95),
                    "key_strengths": random.sample(["技术深度", "项目经验", "团队协作", "问题解决"], 2),
                    "areas_for_improvement": random.sample(["沟通技巧", "行业知识", "技术广度"], 1)
                }
                for _ in range(random.randint(2, 5))
            ],
            "status": "completed"
        }

    def mock_calendar_integration(self, **kwargs):
        """模拟日历集成工具"""
        time.sleep(random.uniform(0.5, 1.5))
        candidates = kwargs.get('candidates', [])
        interviewers = kwargs.get('interviewers', [])

        # 生成面试时间表
        schedule = {}
        base_date = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
        for i, candidate in enumerate(candidates[:min(5, len(candidates))]):  # 最多安排5个候选人
            interview_time = base_date + timedelta(days=i // 3, hours=(i % 3) * 2)
            interviewer = interviewers[i % len(interviewers)] if interviewers else "面试官1"

            schedule[f"candidate_{candidate.get('id', i + 1)}"] = {
                "time": interview_time.strftime("%Y-%m-%d %H:%M"),
                "interviewer": interviewer,
                "duration": "60分钟",
                "format": random.choice(["现场面试", "视频面试"])
            }

        return {
            "schedule": schedule,
            "status": "completed"
        }

    def mock_interview_question_generator(self, **kwargs):
        """模拟面试问题生成工具"""
        time.sleep(random.uniform(0.5, 1.5))
        jd_content = kwargs.get('jd_content', '')

        # 生成面试问题
        technical_questions = [
            "请描述一个你解决过的复杂技术问题",
            "你如何保证代码质量和可维护性?",
            "请解释一下你熟悉的某个设计模式及其应用场景"
        ]

        behavioral_questions = [
            "请描述一个你在团队中解决冲突的经历",
            "你如何应对项目中的紧迫期限?",
            "请分享一个你从失败中学到宝贵经验的例子"
        ]

        return {
            "technical_questions": technical_questions,
            "behavioral_questions": behavioral_questions,
            "evaluation_criteria": ["技术能力", "问题解决", "沟通能力", "团队合作"],
            "status": "completed"
        }

    def mock_evaluation_criteria(self, **kwargs):
        """模拟评估标准工具"""
        time.sleep(random.uniform(0.5, 1.5))
        return {
            "evaluation_form": {
                "technical_skills": {
                    "weight": 0.4,
                    "criteria": ["编程能力", "系统设计", "问题解决", "技术知识"]
                },
                "soft_skills": {
                    "weight": 0.3,
                    "criteria": ["沟通能力", "团队合作", "学习能力", "主动性"]
                },
                "cultural_fit": {
                    "weight": 0.3,
                    "criteria": ["价值观匹配", "工作风格", "职业目标", "公司认同"]
                }
            },
            "rating_scale": "1-5分 (1: 需要改进, 3: 符合期望, 5: 超出期望)",
            "status": "completed"
        }

    def mock_compensation_analyzer(self, **kwargs):
        """模拟薪酬分析工具"""
        time.sleep(random.uniform(0.5, 1.5))
        candidate_evaluations = kwargs.get('candidate_evaluations', {})

        # 生成薪酬建议
        compensation_packages = {}
        for candidate_id, evaluation in list(candidate_evaluations.items())[:5]:  # 最多处理5个候选人
            base_salary = random.randint(15000, 30000)
            compensation_packages[candidate_id] = {
                "base_salary": base_salary,
                "bonus": round(base_salary * random.uniform(0.1, 0.3)),
                "stock_options": random.randint(0, 10000),
                "benefits": ["五险一金", "补充医疗保险", "年度体检", "带薪年假"]
            }

        return {
            "compensation_packages": compensation_packages,
            "market_comparison": f"薪酬水平位于市场{random.randint(60, 90)}分位",
            "status": "completed"
        }

    def mock_market_comparison(self, **kwargs):
        """模拟市场比较工具"""
        time.sleep(random.uniform(0.5, 1.5))
        return {
            "market_data": {
                "similar_positions": random.randint(50, 200),
                "average_salary": random.randint(15000, 25000),
                "salary_range": f"{random.randint(12000, 18000)} - {random.randint(25000, 40000)}",
                "in_demand_skills": random.sample(["Python", "机器学习", "云计算", "大数据", "DevOps"], 3)
            },
            "status": "completed"
        }

    def mock_risk_assessment(self, **kwargs):
        """模拟风险评估工具"""
        time.sleep(random.uniform(0.5, 1.5))
        return {
            "risk_factors": [
                {
                    "factor": "市场竞争力",
                    "risk_level": random.choice(["低", "中", "高"]),
                    "mitigation": "提供有竞争力的薪酬 package"
                },
                {
                    "factor": "候选人接受率",
                    "risk_level": random.choice(["低", "中", "高"]),
                    "mitigation": "准备备选候选人"
                },
                {
                    "factor": "入职时间",
                    "risk_level": random.choice(["低", "中", "高"]),
                    "mitigation": "灵活协商入职日期"
                }
            ],
            "overall_risk": random.choice(["低", "中", "高"]),
            "status": "completed"
        }

    def mock_tool_call(self, tool_name: str, **kwargs) -> Any:
        """模拟工具调用"""
        if tool_name in self.tool_functions:
            logger.info(f"调用工具: {tool_name}")
            return self.tool_functions[tool_name](**kwargs)
        else:
            logger.warning(f"未知工具: {tool_name}")
            return {"status": "error", "message": f"未知工具: {tool_name}"}


class LLMAgentDAGExecutor:
    """LLM Agent DAG执行器"""

    def __init__(self, dag_generator: LLMAgentDAGGenerator):
        self.dag_generator = dag_generator
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

    async def execute_node(self, node: DAGNode, inputs: Dict[str, Any]) -> ExecutionResult:
        """执行单个节点"""
        start_time = time.time()
        node_id = node.id

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

            # 模拟大模型处理
            processing_time = random.uniform(3.0, 10.0)
            await asyncio.sleep(processing_time)

            # 模拟工具调用
            tool_results = {}
            for tool in node.agent.tools:
                tool_result = self.dag_generator.mock_tool_call(tool, **inputs)
                tool_results[tool] = tool_result

            # 模拟生成输出
            output = {
                "processed_inputs": inputs,
                "tool_results": tool_results,
                "decision": random.choice(["approve", "reject", "need_more_info"]),
                "confidence": random.uniform(0.7, 0.95),
                "recommendations": [
                    f"建议{random.randint(1, 3)}: {random.choice(['加强沟通', '增加预算', '调整时间', '优化流程'])}"
                ]
            }

            execution_time = time.time() - start_time

            # 释放资源
            self.release_resources(node.agent.resource_requirements)

            return ExecutionResult(
                node_id=node_id,
                status=TaskStatus.COMPLETED,
                output=output,
                execution_time=execution_time,
                resource_usage=self.calculate_resource_usage(node.agent.resource_requirements, execution_time)
            )

        except Exception as e:
            execution_time = time.time() - start_time
            # 释放资源
            self.release_resources(node.agent.resource_requirements)

            return ExecutionResult(
                node_id=node_id,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time=execution_time
            )

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
                logger.info(f"节点 {node_id} 执行成功，耗时 {result.execution_time:.2f}秒")
            else:
                logger.error(f"节点 {node_id} 执行失败: {result.error}")

        return results


# 示例使用
def main():
    # 示例YAML内容（替代从文件读取）
    yaml_file = "recruitment_sop.yaml"
    with open(yaml_file, 'r', encoding='utf-8') as f:
        yaml_content = f.read()

    # 创建DAG生成器
    dag_generator = LLMAgentDAGGenerator()

    # 解析YAML
    yaml_data = dag_generator.parse_yaml(yaml_content)

    # 生成Agent
    agents = dag_generator.generate_agents_from_yaml(yaml_data)

    # 构建DAG
    nodes = dag_generator.build_dag_from_yaml(yaml_data, agents)

    # 生成可视化图表
    dag_description = dag_generator.generate_dag_visualization(nodes, "recruitment_dag.png")

    print(dag_description)

    # 保存DAG描述到文件
    with open("dag_description.txt", "w", encoding="utf-8") as f:
        f.write(dag_description)

    # 执行DAG
    executor = LLMAgentDAGExecutor(dag_generator)

    # 异步执行
    async def run_execution():
        results = await executor.execute_dag(nodes)

        # 打印执行结果
        print("\n=== 执行结果 ===")
        for node_id, result in results.items():
            status_icon = "✅" if result.status == TaskStatus.COMPLETED else "❌"
            print(f"{status_icon} {node_id}: {result.status.value}")
            if result.execution_time:
                print(f"   耗时: {result.execution_time:.2f}秒")
            if result.resource_usage:
                print(f"   资源使用: {result.resource_usage}")

        # 保存详细结果
        with open("execution_results.json", "w", encoding="utf-8") as f:
            result_dict = {}
            for node_id, result in results.items():
                result_dict[node_id] = {
                    "status": result.status.value,
                    "execution_time": result.execution_time,
                    "resource_usage": {k.value: v for k, v in
                                       result.resource_usage.items()} if result.resource_usage else None,
                    "error": result.error
                }
            json.dump(result_dict, f, ensure_ascii=False, indent=2)

    # 运行异步执行
    asyncio.run(run_execution())


if __name__ == "__main__":
    main()