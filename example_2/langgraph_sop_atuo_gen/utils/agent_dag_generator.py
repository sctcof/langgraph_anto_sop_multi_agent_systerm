from src.utils.common import AgentType,ResourceType
from src.utils.util import DeepSeekAgent,ResourceRequirement,DAGNode
from src.utils.mcp_tools import MCPClient
import yaml
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Any,List
import logging
from datetime import datetime, timedelta
import time
import random


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DeepSeekAgentDAGGenerator")

class DeepSeekAgentDAGGenerator:
    """从YAML生成DeepSeek Agent DAG"""

    def __init__(self):
        # 预定义的Agent模板
        self.agent_templates = {
            "position_planning": DeepSeekAgent(
                id="position_planner",
                name="职位规划Agent",
                agent_type=AgentType.PLANNER,
                description="负责职位需求和规划的Agent",
                resource_requirements=[
                    ResourceRequirement(ResourceType.LLM, 1, 5, 2),
                    ResourceRequirement(ResourceType.CPU, 2, 4, 1)
                ],
                tools=["market_analysis", "budget_calculator", "role_definition"],
                input_schema={"department": "str", "business_needs": "str"},
                output_schema={"position_title": "str", "requirements": "list", "budget_range": "dict"},
                system_prompt="你是一个专业的招聘规划专家，擅长分析业务需求并制定合适的职位规划。"
            ),
            "jd_creation": DeepSeekAgent(
                id="jd_creator",
                name="职位描述创建Agent",
                agent_type=AgentType.EXECUTOR,
                description="负责创建职位描述的Agent",
                resource_requirements=[
                    ResourceRequirement(ResourceType.LLM, 1, 3, 2),
                    ResourceRequirement(ResourceType.CPU, 1, 2, 1)
                ],
                tools=["jd_template", "skill_extractor", "industry_benchmark"],
                input_schema={"position_title": "str", "requirements": "list"},
                output_schema={"jd_content": "str", "key_skills": "list", "experience_level": "str"},
                system_prompt="你是一个专业的招聘专员，擅长编写清晰、吸引人的职位描述。"
            ),
            "candidate_screening": DeepSeekAgent(
                id="candidate_screener",
                name="候选人筛选Agent",
                agent_type=AgentType.ANALYZER,
                description="负责筛选候选人的Agent",
                resource_requirements=[
                    ResourceRequirement(ResourceType.LLM, 2, 4, 2),
                    ResourceRequirement(ResourceType.CPU, 2, 4, 1),
                    ResourceRequirement(ResourceType.MEMORY, 4, 8, 1)
                ],
                tools=["resume_parser", "skill_matcher", "experience_evaluator"],
                input_schema={"jd_content": "str", "candidate_profiles": "list"},
                output_schema={"qualified_candidates": "list", "rejection_reasons": "dict", "ranking": "list"},
                system_prompt="你是一个专业的招聘分析师，擅长评估候选人简历并与职位要求进行匹配。"
            ),
            "interview_planning": DeepSeekAgent(
                id="interview_planner",
                name="面试规划Agent",
                agent_type=AgentType.PLANNER,
                description="负责规划面试流程的Agent",
                resource_requirements=[
                    ResourceRequirement(ResourceType.LLM, 1, 3, 2),
                    ResourceRequirement(ResourceType.CPU, 1, 2, 1)
                ],
                tools=["calendar_integration", "interview_question_generator", "evaluation_criteria"],
                input_schema={"candidates": "list", "interviewers": "list"},
                output_schema={"schedule": "dict", "questions": "list", "evaluation_form": "dict"},
                system_prompt="你是一个专业的面试协调员，擅长安排面试流程和设计面试问题。"
            ),
            "offer_decision": DeepSeekAgent(
                id="offer_decider",
                name="录用决策Agent",
                agent_type=AgentType.DECISION,
                description="负责做出录用决策的Agent",
                resource_requirements=[
                    ResourceRequirement(ResourceType.LLM, 2, 4, 3),
                    ResourceRequirement(ResourceType.CPU, 2, 4, 2)
                ],
                tools=["compensation_analyzer", "market_comparison", "risk_assessment"],
                input_schema={"candidate_evaluations": "dict", "budget_constraints": "dict"},
                output_schema={"offer_decisions": "dict", "compensation_packages": "dict", "start_dates": "dict"},
                system_prompt="你是一个专业的招聘决策者，擅长评估候选人并做出合理的录用决策。"
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

    def generate_agents_from_yaml(self, yaml_data: Dict[str, Any]) -> Dict[str, DeepSeekAgent]:
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
                agent = DeepSeekAgent(
                    id=f"{agent_key}_{task_id}",
                    name=f"{task_name} Agent",
                    agent_type=template.agent_type,
                    description=template.description,
                    model=template.model,
                    resource_requirements=template.resource_requirements.copy(),
                    tools=template.tools.copy(),
                    input_schema=template.input_schema.copy(),
                    output_schema=template.output_schema.copy(),
                    system_prompt=template.system_prompt
                )
                agents[task_id] = agent

        return agents

    def build_dag_from_yaml(self, yaml_data: Dict[str, Any], agents: Dict[str, DeepSeekAgent]) -> Dict[str, DAGNode]:
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

        plt.title("DeepSeek Agent DAG Execution Flow")
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
        description = "DeepSeek Agent DAG 执行流程:\n\n"

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

    # 各种工具模拟函数（与之前相同）
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


class DeepSeekMCPAgentDAGGenerator:
    """从YAML生成DeepSeek Agent DAG，支持MCP工具调用"""

    def __init__(self, mcp_client: MCPClient):
        self.mcp_client = mcp_client
        self.available_tools = []

        # 预定义的Agent模板
        self.agent_templates = {
            "position_planning": DeepSeekAgent(
                id="position_planner",
                name="职位规划Agent",
                agent_type=AgentType.PLANNER,
                description="负责职位需求和规划的Agent",
                resource_requirements=[
                    ResourceRequirement(ResourceType.LLM, 1, 5, 2),
                    ResourceRequirement(ResourceType.CPU, 2, 4, 1)
                ],
                tools=["market_analysis", "budget_calculator", "role_definition"],
                input_schema={"department": "str", "business_needs": "str"},
                output_schema={"position_title": "str", "requirements": "list", "budget_range": "dict"},
                system_prompt="你是一个专业的招聘规划专家，擅长分析业务需求并制定合适的职位规划。"
            ),
            "jd_creation": DeepSeekAgent(
                id="jd_creator",
                name="职位描述创建Agent",
                agent_type=AgentType.EXECUTOR,
                description="负责创建职位描述的Agent",
                resource_requirements=[
                    ResourceRequirement(ResourceType.LLM, 1, 3, 2),
                    ResourceRequirement(ResourceType.CPU, 1, 2, 1)
                ],
                tools=["jd_template", "skill_extractor", "industry_benchmark"],
                input_schema={"position_title": "str", "requirements": "list"},
                output_schema={"jd_content": "str", "key_skills": "list", "experience_level": "str"},
                system_prompt="你是一个专业的招聘专员，擅长编写清晰、吸引人的职位描述。"
            ),
            "candidate_screening": DeepSeekAgent(
                id="candidate_screener",
                name="候选人筛选Agent",
                agent_type=AgentType.ANALYZER,
                description="负责筛选候选人的Agent",
                resource_requirements=[
                    ResourceRequirement(ResourceType.LLM, 2, 4, 2),
                    ResourceRequirement(ResourceType.CPU, 2, 4, 1),
                    ResourceRequirement(ResourceType.MEMORY, 4, 8, 1)
                ],
                tools=["resume_parser", "skill_matcher", "experience_evaluator"],
                input_schema={"jd_content": "str", "candidate_profiles": "list"},
                output_schema={"qualified_candidates": "list", "rejection_reasons": "dict", "ranking": "list"},
                system_prompt="你是一个专业的招聘分析师，擅长评估候选人简历并与职位要求进行匹配。"
            ),
            "interview_planning": DeepSeekAgent(
                id="interview_planner",
                name="面试规划Agent",
                agent_type=AgentType.PLANNER,
                description="负责规划面试流程的Agent",
                resource_requirements=[
                    ResourceRequirement(ResourceType.LLM, 1, 3, 2),
                    ResourceRequirement(ResourceType.CPU, 1, 2, 1)
                ],
                tools=["calendar_integration", "interview_question_generator", "evaluation_criteria"],
                input_schema={"candidates": "list", "interviewers": "list"},
                output_schema={"schedule": "dict", "questions": "list", "evaluation_form": "dict"},
                system_prompt="你是一个专业的面试协调员，擅长安排面试流程和设计面试问题。"
            ),
            "offer_decision": DeepSeekAgent(
                id="offer_decider",
                name="录用决策Agent",
                agent_type=AgentType.DECISION,
                description="负责做出录用决策的Agent",
                resource_requirements=[
                    ResourceRequirement(ResourceType.LLM, 2, 4, 3),
                    ResourceRequirement(ResourceType.CPU, 2, 4, 2)
                ],
                tools=["compensation_analyzer", "market_comparison", "risk_assessment"],
                input_schema={"candidate_evaluations": "dict", "budget_constraints": "dict"},
                output_schema={"offer_decisions": "dict", "compensation_packages": "dict", "start_dates": "dict"},
                system_prompt="你是一个专业的招聘决策者，擅长评估候选人并做出合理的录用决策。"
            )
        }

    async def initialize(self):
        """初始化，获取可用的MCP工具"""
        try:
            self.available_tools = await self.mcp_client.list_tools()
            logger.info(f"获取到 {len(self.available_tools)} 个MCP工具")
        except Exception as e:
            logger.error(f"初始化MCP工具失败: {e}")
            self.available_tools = []

    def parse_yaml(self, yaml_content: str) -> Dict[str, Any]:
        """解析YAML内容"""
        return yaml.safe_load(yaml_content)

    def generate_agents_from_yaml(self, yaml_data: Dict[str, Any]) -> Dict[str, DeepSeekAgent]:
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
                agent = DeepSeekAgent(
                    id=f"{agent_key}_{task_id}",
                    name=f"{task_name} Agent",
                    agent_type=template.agent_type,
                    description=template.description,
                    model=template.model,
                    resource_requirements=template.resource_requirements.copy(),
                    tools=self._filter_available_tools(template.tools),
                    input_schema=template.input_schema.copy(),
                    output_schema=template.output_schema.copy(),
                    system_prompt=template.system_prompt
                )
                agents[task_id] = agent

        return agents

    def _filter_available_tools(self, requested_tools: List[str]) -> List[str]:
        """过滤出可用的工具"""
        available_tool_names = [tool.get("name") for tool in self.available_tools]
        return [tool for tool in requested_tools if tool in available_tool_names]

    def build_dag_from_yaml(self, yaml_data: Dict[str, Any], agents: Dict[str, DeepSeekAgent]) -> Dict[str, DAGNode]:
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

        plt.title("DeepSeek Agent DAG Execution Flow with MCP Tools")
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
        description = "DeepSeek Agent DAG 执行流程 (使用MCP工具):\n\n"

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