import asyncio
import random
import json
import re
import os
from typing import Dict, List, Optional, TypedDict, Annotated, Any
from datetime import datetime
from enum import Enum

from jedi.inference.recursion import recursion_limit
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_deepseek import ChatDeepSeek
from langchain.agents import Tool, AgentExecutor
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
import functools

from networkx.classes import is_empty


# 模拟Word文档操作
class WordDocument:
    """模拟Word文档操作类"""

    def __init__(self, title):
        self.title = title
        self.content = []
        self.sections = {}

    def add_heading(self, text, level=1):
        self.content.append(f"{'#' * level} {text}")
        return self

    def add_paragraph(self, text):
        self.content.append(text)
        return self

    def add_table(self, data, headers=None):
        if headers:
            table_str = "| " + " | ".join(headers) + " |\n"
            table_str += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        else:
            headers = list(data[0].keys()) if data else []
            table_str = "| " + " | ".join(headers) + " |\n"
            table_str += "| " + " | ".join(["---"] * len(headers)) + " |\n"

        for row in data:
            if isinstance(row, dict):
                table_str += "| " + " | ".join(str(row.get(h, "")) for h in headers) + " |\n"
            else:
                table_str += "| " + " | ".join(str(cell) for cell in row) + " |\n"

        self.content.append(table_str)
        return self

    def save(self, filename):
        """模拟保存文档"""
        os.makedirs("docs", exist_ok=True)
        with open(f"docs/{filename}.md", "w", encoding="utf-8") as f:
            f.write("\n\n".join(self.content))
        return f"docs/{filename}.md"

    def get_content(self):
        return "\n\n".join(self.content)


# 设置OpenAI API密钥
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"


# 定义任务类型枚举
class TaskType(Enum):
    REQUIREMENT_ANALYSIS = "requirement_analysis"
    RESUME_SCREENING = "resume_screening"
    TECHNICAL_ASSESSMENT = "technical_assessment"
    CULTURAL_FIT = "cultural_fit"
    REFERENCE_CHECK = "reference_check"
    SALARY_NEGOTIATION = "salary_negotiation"
    FINAL_EVALUATION = "final_evaluation"


# 定义状态结构
class RecruitmentState(TypedDict):
    # 输入信息
    job_description: str
    candidate_profiles: List[Dict]
    sop_definition: str  # 标准操作流程定义

    # 任务拆解
    tasks: List[Dict]
    task_dependencies: Dict[str, List[str]]
    agent_configs: Dict[str, Dict]

    # 工作状态
    assigned_tasks: List[Dict]
    completed_tasks: List[Dict]
    worker_status: Dict[str, str]

    # 文档存储
    generated_documents: Dict[str, str]

    # 结果
    evaluations: List[Dict]
    final_recommendations: List[Dict]

    # 系统消息
    messages: Annotated[List, add_messages]

# DeepSeek API配置
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"  # 根据实际情况调整模型名称
DEEPSEEK_KEY = "sk-f26c5faf4bce41eabc0fca675f36c6c0" ##task_招聘_ak

# 初始化LLM模型
# llm = ChatOpenAI(model="gpt-4", temperature=0.7)

llm = ChatDeepSeek(
    model="deepseek-chat",  # 或者具体的 DeepSeek 模型名称
    api_base=DEEPSEEK_API_BASE,  # DeepSeek 的 API 地址
    api_key=DEEPSEEK_KEY,  # 你的 DeepSeek API Key
)


# SOP解析器
class SOPParser:
    @staticmethod
    def parse(sop_text: str) -> Dict:
        """解析SOP文本，提取任务和依赖关系"""
        tasks = []
        dependencies = {}

        # 使用正则表达式提取任务和依赖
        task_pattern = r"任务\s*(\d+):\s*(.*?)(?=任务\s*\d+|$)"
        dep_pattern = r"依赖:\s*([\d,\s]+)"
        rule_pattern = r"规则:\s*(.*?)(?=依赖:|$)"

        matches = re.finditer(task_pattern, sop_text, re.DOTALL)
        for match in matches:
            task_num = match.group(1)
            task_desc = match.group(2).strip()

            # 提取依赖
            deps = []
            dep_match = re.search(dep_pattern, task_desc)
            if dep_match:
                deps = [d.strip() for d in dep_match.group(1).split(",")]
                # 移除依赖描述部分
                task_desc = re.sub(dep_pattern, "", task_desc).strip()

            # 提取规则
            rules = []
            rule_match = re.search(rule_pattern, task_desc)
            if rule_match:
                rules = [r.strip() for r in rule_match.group(1).split(";")]
                # 移除规则描述部分
                task_desc = re.sub(rule_pattern, "", task_desc).strip()

            # 确定任务类型
            task_type = SOPParser._determine_task_type(task_desc)

            tasks.append({
                "task_id": f"task_{task_num}",
                "description": task_desc,
                "type": task_type.value if task_type else TaskType.REQUIREMENT_ANALYSIS.value,
                "rules": rules
            })

            if deps:
                dependencies[f"task_{task_num}"] = [f"task_{d}" for d in deps]

        return {
            "tasks": tasks,
            "dependencies": dependencies
        }

    @staticmethod
    def _determine_task_type(description: str) -> Optional[TaskType]:
        """根据描述确定任务类型"""
        description_lower = description.lower()

        if any(word in description_lower for word in ["需求", "分析", "职位描述"]):
            return TaskType.REQUIREMENT_ANALYSIS
        elif any(word in description_lower for word in ["简历", "筛选", "初筛"]):
            return TaskType.RESUME_SCREENING
        elif any(word in description_lower for word in ["技术", "编程", "代码", "测试"]):
            return TaskType.TECHNICAL_ASSESSMENT
        elif any(word in description_lower for word in ["文化", "价值观", "团队", "沟通"]):
            return TaskType.CULTURAL_FIT
        elif any(word in description_lower for word in ["参考", "背景", "核实"]):
            return TaskType.REFERENCE_CHECK
        elif any(word in description_lower for word in ["薪资", "薪酬", "谈判"]):
            return TaskType.SALARY_NEGOTIATION
        elif any(word in description_lower for word in ["最终", "汇总", "评估", "决定"]):
            return TaskType.FINAL_EVALUATION

        return None


# 规则引擎
class RuleEngine:
    """规则引擎，用于评估任务结果是否符合要求"""

    @staticmethod
    def evaluate_rule(rule: str, task_result: Dict, previous_results: Dict) -> bool:
        """评估单个规则"""
        # 简单的规则评估逻辑，实际应用中可以使用更复杂的规则引擎
        try:
            # 替换规则中的变量
            evaluated_rule = rule
            for key, value in task_result.items():
                if isinstance(value, (str, int, float)):
                    evaluated_rule = evaluated_rule.replace(f"{{{key}}}", str(value))

            # 添加对前置任务结果的引用
            for task_id, result in previous_results.items():
                if isinstance(result, dict):
                    for k, v in result.items():
                        if isinstance(v, (str, int, float)):
                            evaluated_rule = evaluated_rule.replace(f"{{{task_id}.{k}}}", str(v))

            # 简单的规则评估
            if ">=" in evaluated_rule:
                parts = evaluated_rule.split(">=")
                if len(parts) == 2:
                    left, right = parts
                    return float(left.strip()) >= float(right.strip())
            elif "<=" in evaluated_rule:
                parts = evaluated_rule.split("<=")
                if len(parts) == 2:
                    left, right = parts
                    return float(left.strip()) <= float(right.strip())
            elif ">" in evaluated_rule:
                parts = evaluated_rule.split(">")
                if len(parts) == 2:
                    left, right = parts
                    return float(left.strip()) > float(right.strip())
            elif "<" in evaluated_rule:
                parts = evaluated_rule.split("<")
                if len(parts) == 2:
                    left, right = parts
                    return float(left.strip()) < float(right.strip())
            elif "==" in evaluated_rule:
                parts = evaluated_rule.split("==")
                if len(parts) == 2:
                    left, right = parts
                    return left.strip() == right.strip()
            elif "!=" in evaluated_rule:
                parts = evaluated_rule.split("!=")
                if len(parts) == 2:
                    left, right = parts
                    return left.strip() != right.strip()
            elif "包含" in evaluated_rule:
                parts = evaluated_rule.split("包含")
                if len(parts) == 2:
                    container, target = parts
                    return target.strip() in container.strip()

            # 默认返回True，避免规则解析失败导致任务失败
            return True
        except:
            return True

    @staticmethod
    def evaluate_all_rules(rules: List[str], task_result: Dict, previous_results: Dict) -> Dict:
        """评估所有规则"""
        evaluation = {
            "passed": True,
            "failed_rules": [],
            "feedback": ""
        }

        for rule in rules:
            if not RuleEngine.evaluate_rule(rule, task_result, previous_results):
                evaluation["passed"] = False
                evaluation["failed_rules"].append(rule)

        # 生成反馈信息
        if not evaluation["passed"]:
            evaluation["feedback"] = f"任务未通过规则检查，失败的规则: {', '.join(evaluation['failed_rules'])}"

        return evaluation

# 通用工具函数 - 在没有其他工具时使用的默认函数，保证流程正确
def general_tool(general_input: str) -> str:
    profiles={}
    return json.dumps(profiles.get(general_input,{"info": "输入信息正常，执行后续动作"}), ensure_ascii=False)

# 模拟工具函数 - 简历筛选相关
def search_linkedin_profile(candidate_name: str) -> str:
    """模拟搜索LinkedIn个人资料"""
    profiles = {
        "张三": {
            "姓名": "张三",
            "当前职位": "高级Python工程师 at 公司A",
            "经验": "6年",
            "技能": ["Python", "Django", "Flask", "机器学习", "深度学习"],
            "教育背景": "清华大学计算机科学硕士",
            "认证": ["AWS认证解决方案架构师", "Python高级开发认证"],
            "推荐信数量": 3,
            "关注者": 450
        },
        "李四": {
            "姓名": "李四",
            "当前职位": "全栈开发工程师 at 公司C",
            "经验": "4年Java, 2年Python",
            "技能": ["Java", "Python", "Spring", "微服务", "Docker"],
            "教育背景": "北京大学软件工程学士",
            "认证": ["Java认证开发专家", "Spring专业认证"],
            "推荐信数量": 2,
            "关注者": 280
        },
        "candidate_1": {
            "姓名": "张三",
            "当前职位": "高级Python工程师 at 公司A",
            "经验": "6年",
            "技能": ["Python", "Django", "Flask", "机器学习", "深度学习"],
            "教育背景": "清华大学计算机科学硕士",
            "认证": ["AWS认证解决方案架构师", "Python高级开发认证"],
            "推荐信数量": 3,
            "关注者": 450
        },
        "candidate_2": {
            "姓名": "李四",
            "当前职位": "全栈开发工程师 at 公司C",
            "经验": "4年Java, 2年Python",
            "技能": ["Java", "Python", "Spring", "微服务", "Docker"],
            "教育背景": "北京大学软件工程学士",
            "认证": ["Java认证开发专家", "Spring专业认证"],
            "推荐信数量": 2,
            "关注者": 280
        }
    }
    return json.dumps(profiles.get(candidate_name, {"error": "未找到该候选人的LinkedIn资料"}), ensure_ascii=False)


def check_github_activity(candidate_name: str) -> str:
    """模拟检查GitHub活动"""
    activities = {
        "张三": {
            "用户名": "zhangsan-dev",
            "贡献度": "高",
            "主要项目": ["ml-library", "web-framework"],
            "星星数": 128,
            "fork数": 45,
            "最近活动": "上周有15次提交",
            "代码质量评分": 4.8
        },
        "李四": {
            "用户名": "lisi-coder",
            "贡献度": "中等",
            "主要项目": ["microservice-auth", "api-gateway"],
            "星星数": 42,
            "fork数": 18,
            "最近活动": "上月有8次提交",
            "代码质量评分": 4.2
        },
        "candidate_1": {
            "用户名": "zhangsan-dev",
            "中文名":"张三",
            "贡献度": "高",
            "主要项目": ["ml-library", "web-framework"],
            "星星数": 128,
            "fork数": 45,
            "最近活动": "上周有15次提交",
            "代码质量评分": 4.8
        },
        "candidate_2": {
            "用户名": "lisi-coder",
            "中文名": "李四",
            "贡献度": "中等",
            "主要项目": ["microservice-auth", "api-gateway"],
            "星星数": 42,
            "fork数": 18,
            "最近活动": "上月有8次提交",
            "代码质量评分": 4.2
        }
    }
    return json.dumps(activities.get(candidate_name, {"error": "未找到该候选人的GitHub活动"}), ensure_ascii=False)


def analyze_resume_compatibility(candidate_name: str) -> str:
    """模拟分析简历与职位描述的匹配度"""
    compatibilities = {
        "张三": {
            "技能匹配度": "92%",
            "经验匹配度": "95%",
            "教育背景匹配度": "100%",
            "总体匹配度": "94%",
            "关键匹配点": ["Python高级开发", "机器学习经验", "云计算经验"],
            "潜在差距": ["需要更多团队管理经验"]
        },
        "李四": {
            "技能匹配度": "78%",
            "经验匹配度": "82%",
            "教育背景匹配度": "90%",
            "总体匹配度": "83%",
            "关键匹配点": ["全栈开发", "微服务架构"],
            "潜在差距": ["Python经验较少", "机器学习经验有限"]
        },
        "candidate_1": {
            "技能匹配度": "92%",
            "经验匹配度": "95%",
            "教育背景匹配度": "100%",
            "总体匹配度": "94%",
            "关键匹配点": ["Python高级开发", "机器学习经验", "云计算经验"],
            "潜在差距": ["需要更多团队管理经验"]
        },
        "candidate_2": {
            "技能匹配度": "78%",
            "经验匹配度": "82%",
            "教育背景匹配度": "90%",
            "总体匹配度": "83%",
            "关键匹配点": ["全栈开发", "微服务架构"],
            "潜在差距": ["Python经验较少", "机器学习经验有限"]
        }
    }
    return json.dumps(compatibilities.get(candidate_name, {"error": "未找到该候选人的匹配度分析"}), ensure_ascii=False)


# 模拟工具函数 - 技术评估相关
def run_technical_test(candidate_name: str, test_type: str = "python") -> str:
    """模拟运行技术测试"""
    results = {
        "张三": {
            "测试类型": test_type,
            "得分": 92,
            "完成时间": "45分钟",
            "正确率": "92%",
            "代码质量": "优秀",
            "算法效率": "高效",
            "代码可读性": "优秀",
            "反馈": "算法实现高效，代码结构清晰，有良好的注释习惯"
        },
        "李四": {
            "测试类型": test_type,
            "得分": 78,
            "完成时间": "55分钟",
            "正确率": "78%",
            "代码质量": "良好",
            "算法效率": "中等",
            "代码可读性": "良好",
            "反馈": "基本功能实现正确，但在边界情况处理上有所欠缺"
        },
        "candidate_1": {
            "测试类型": test_type,
            "得分": 92,
            "完成时间": "45分钟",
            "正确率": "92%",
            "代码质量": "优秀",
            "算法效率": "高效",
            "代码可读性": "优秀",
            "反馈": "算法实现高效，代码结构清晰，有良好的注释习惯"
        },
        "candidate_2": {
            "测试类型": test_type,
            "得分": 78,
            "完成时间": "55分钟",
            "正确率": "78%",
            "代码质量": "良好",
            "算法效率": "中等",
            "代码可读性": "良好",
            "反馈": "基本功能实现正确，但在边界情况处理上有所欠缺"
        }
    }
    return json.dumps(results.get(candidate_name, {"error": "未找到该候选人的测试结果"}), ensure_ascii=False)


def assess_problem_solving_skills(candidate_name: str) -> str:
    """模拟评估问题解决能力"""
    assessments = {
        "张三": {
            "问题分析能力": "优秀",
            "解决方案创新性": "高",
            "实施效率": "优秀",
            "调试能力": "优秀",
            "学习能力": "优秀",
            "总体评分": 4.8
        },
        "李四": {
            "问题分析能力": "良好",
            "解决方案创新性": "中等",
            "实施效率": "良好",
            "调试能力": "良好",
            "学习能力": "良好",
            "总体评分": 4.0
        },
        "candidate_1": {
            "问题分析能力": "优秀",
            "解决方案创新性": "高",
            "实施效率": "优秀",
            "调试能力": "优秀",
            "学习能力": "优秀",
            "总体评分": 4.8
        },
        "candidate_2": {
            "问题分析能力": "良好",
            "解决方案创新性": "中等",
            "实施效率": "良好",
            "调试能力": "良好",
            "学习能力": "良好",
            "总体评分": 4.0
        }
    }
    return json.dumps(assessments.get(candidate_name, {"error": "未找到该候选人的问题解决能力评估"}),
                      ensure_ascii=False)


# 模拟工具函数 - 文化匹配度评估相关
def check_cultural_fit_indicators(candidate_name: str) -> str:
    """模拟检查文化匹配度指标"""
    indicators = {
        "张三": {
            "团队合作": "优秀",
            "沟通能力": "良好",
            "适应性": "优秀",
            "价值观匹配": "高",
            "工作风格": "注重细节，追求卓越",
            "潜在关注点": "可能过于专注于技术细节",
            "文化匹配度评分": 4.5
        },
        "李四": {
            "团队合作": "良好",
            "沟通能力": "优秀",
            "适应性": "良好",
            "价值观匹配": "中等",
            "工作风格": "务实，注重效率",
            "潜在关注点": "可能需要更多时间适应敏捷开发流程",
            "文化匹配度评分": 3.8
        },
        "candidate_1": {
            "团队合作": "优秀",
            "沟通能力": "良好",
            "适应性": "优秀",
            "价值观匹配": "高",
            "工作风格": "注重细节，追求卓越",
            "潜在关注点": "可能过于专注于技术细节",
            "文化匹配度评分": 4.5
        },
        "candidate_2": {
            "团队合作": "良好",
            "沟通能力": "优秀",
            "适应性": "良好",
            "价值观匹配": "中等",
            "工作风格": "务实，注重效率",
            "潜在关注点": "可能需要更多时间适应敏捷开发流程",
            "文化匹配度评分": 3.8
        }
    }
    return json.dumps(indicators.get(candidate_name, {"error": "未找到该候选人的文化匹配指标"}), ensure_ascii=False)


def search_professional_references(candidate_name: str) -> str:
    """模拟搜索专业推荐信"""
    references = {
        "张三": [
            {
                "推荐人": "王五, 技术总监 at 公司A",
                "关系": "前直接上司",
                "合作时长": "3年",
                "评价": "张三是我们团队中最出色的工程师之一，技术能力强，学习速度快，能够快速适应新技术",
                "推荐强度": "强烈推荐",
                "可信度": "高"
            },
            {
                "推荐人": "赵六, 高级工程师 at 公司A",
                "关系": "前同事",
                "合作时长": "2年",
                "评价": "张三有很强的技术领导能力，能够有效指导 junior 工程师",
                "推荐强度": "推荐",
                "可信度": "中"
            }
        ],
        "李四": [
            {
                "推荐人": "钱七, 项目经理 at 公司C",
                "关系": "前直接上司",
                "合作时长": "2年",
                "评价": "李四是一个可靠的全栈开发人员，沟通能力出色，能够很好地理解业务需求",
                "推荐强度": "推荐",
                "可信度": "高"
            }
        ],
        "candidate_1": [
            {
                "推荐人": "王五, 技术总监 at 公司A",
                "关系": "前直接上司",
                "合作时长": "3年",
                "评价": "张三是我们团队中最出色的工程师之一，技术能力强，学习速度快，能够快速适应新技术",
                "推荐强度": "强烈推荐",
                "可信度": "高"
            },
            {
                "推荐人": "赵六, 高级工程师 at 公司A",
                "关系": "前同事",
                "合作时长": "2年",
                "评价": "张三有很强的技术领导能力，能够有效指导 junior 工程师",
                "推荐强度": "推荐",
                "可信度": "中"
            }
        ],
        "candidate_2": [
            {
                "推荐人": "钱七, 项目经理 at 公司C",
                "关系": "前直接上司",
                "合作时长": "2年",
                "评价": "李四是一个可靠的全栈开发人员，沟通能力出色，能够很好地理解业务需求",
                "推荐强度": "推荐",
                "可信度": "高"
            }
        ]
    }
    return json.dumps(references.get(candidate_name, {"error": "未找到该候选人的推荐信"}), ensure_ascii=False)


# 模拟工具函数 - 薪资谈判相关
def analyze_market_salary(role: str, experience: int, location: str = "北京") -> str:
    """模拟分析市场薪资水平"""
    salaries = {
        "高级Python工程师": {
            "经验要求": "5-8年",
            "薪资范围": "¥35,000 - ¥55,000",
            "中位数": "¥45,000",
            "市场热度": "高",
            "趋势": "上升"
        },
        "全栈开发工程师": {
            "经验要求": "3-6年",
            "薪资范围": "¥25,000 - ¥40,000",
            "中位数": "¥32,000",
            "市场热度": "中等",
            "趋势": "稳定"
        }
    }
    return json.dumps(salaries.get(role, {"error": "未找到该职位的薪资数据"}), ensure_ascii=False)


def assess_candidate_expectations(candidate_name: str) -> str:
    """模拟评估候选人的薪资期望"""
    expectations = {
        "张三": {
            "当前薪资": "¥42,000",
            "期望薪资": "¥55,000 - ¥65,000",
            "期望涨幅": "30%",
            "灵活性": "中等",
            "优先考虑因素": ["技术挑战", "成长空间", "团队文化"],
            "谈判空间": "有一定灵活性"
        },
        "李四": {
            "当前薪资": "¥32,000",
            "期望薪资": "¥40,000 - ¥45,000",
            "期望涨幅": "25%",
            "灵活性": "高",
            "优先考虑因素": ["工作生活平衡", "稳定性", "技术栈"],
            "谈判空间": "较大灵活性"
        },
        "candidate_1": {
            "当前薪资": "¥42,000",
            "期望薪资": "¥55,000 - ¥65,000",
            "期望涨幅": "30%",
            "灵活性": "中等",
            "优先考虑因素": ["技术挑战", "成长空间", "团队文化"],
            "谈判空间": "有一定灵活性"
        },
        "candidate_2": {
            "当前薪资": "¥32,000",
            "期望薪资": "¥40,000 - ¥45,000",
            "期望涨幅": "25%",
            "灵活性": "高",
            "优先考虑因素": ["工作生活平衡", "稳定性", "技术栈"],
            "谈判空间": "较大灵活性"
        }
    }
    return json.dumps(expectations.get(candidate_name, {"error": "未找到该候选人的薪资期望"}), ensure_ascii=False)


# 工具定义
tools = [
    Tool(
        name="general_tool",
        func=general_tool,
        description="通用兜底工具，不做任何处理"
    ),
    Tool(
        name="search_linkedin_profile",
        func=search_linkedin_profile,
        description="搜索候选人的LinkedIn个人资料，获取职业经历、技能和教育背景"
    ),
    Tool(
        name="check_github_activity",
        func=check_github_activity,
        description="检查候选人的GitHub活动，了解其开源贡献和代码质量"
    ),
    Tool(
        name="analyze_resume_compatibility",
        func=analyze_resume_compatibility,
        description="分析候选人简历与职位描述的匹配度，包括技能、经验和教育背景"
    ),
    Tool(
        name="run_technical_test",
        func=run_technical_test,
        description="运行技术测试，评估候选人的编程能力和问题解决技巧"
    ),
    Tool(
        name="assess_problem_solving_skills",
        func=assess_problem_solving_skills,
        description="评估候选人的问题解决能力和学习方法"
    ),
    Tool(
        name="check_cultural_fit_indicators",
        func=check_cultural_fit_indicators,
        description="检查文化匹配度指标，评估候选人与公司文化的适配程度"
    ),
    Tool(
        name="search_professional_references",
        func=search_professional_references,
        description="搜索专业推荐信，了解候选人过往工作表现"
    ),
    Tool(
        name="analyze_market_salary",
        func=analyze_market_salary,
        description="分析市场薪资水平，了解特定职位和经验的薪资范围"
    ),
    Tool(
        name="assess_candidate_expectations",
        func=assess_candidate_expectations,
        description="评估候选人的薪资期望和谈判空间"
    )
]

# 智能体提示词模板
AGENT_PROMPTS = {
    TaskType.REQUIREMENT_ANALYSIS.value: """
    你是一名招聘需求分析专家，负责分析职位描述并生成详细的候选人需求文档。
    你需要深入理解职位要求，明确所需的技能、经验和特质。

    分析要点：
    1. 核心技能要求：职位需要的核心技术栈和工具
    2. 经验要求：所需的工作经验和项目经验
    3. 教育背景：所需的学历和专业背景
    4. 软技能：沟通、团队合作等软技能要求
    5. 文化匹配：公司文化和工作环境匹配度要求

    请生成一个结构化的Word文档，包含以上要点和详细说明。
    """,

    TaskType.RESUME_SCREENING.value: """
    你是一名专业的简历筛选专家，负责评估候选人是否符合职位的基本要求。
    你需要使用工具获取候选人的详细信息，然后给出全面评估。

    评估要点：
    1. 技能匹配度：候选人的技能是否符合职位要求
    2. 经验匹配度：候选人的工作经验是否符合职位要求
    3. 教育背景：候选人的教育背景是否符合职位要求
    4. 总体匹配度：综合评估候选人是否适合进入下一轮

    请生成一个结构化的Word评估报告，包含以上要点和详细理由。
    """,

    TaskType.TECHNICAL_ASSESSMENT.value: """
    你是一名技术面试官，负责深入评估候选人的技术能力。
    你需要使用工具运行技术测试并分析候选人的技术背景。

    评估要点：
    1. 技术技能：候选人的编程语言和框架掌握程度
    2. 问题解决能力：候选人分析和解决问题的能力
    3. 代码质量：候选人的代码编写规范和可维护性
    4. 学习能力：候选人学习新技术的速度和适应性

    请生成一个结构化的Word技术评估报告，包含以上要点和详细理由。
    """,

    TaskType.CULTURAL_FIT.value: """
    你是一名企业文化专家，负责评估候选人与公司文化的匹配度。
    你需要使用工具分析候选人的工作风格和价值观。

    评估要点：
    1. 团队合作：候选人的团队协作能力和沟通风格
    2. 价值观匹配：候选人的价值观是否与公司文化一致
    3. 工作风格：候选人的工作方式和偏好
    4. 适应性：候选人适应新环境和文化的能力

    请生成一个结构化的Word文化匹配度评估报告，包含以上要点和详细理由。
    """,

    TaskType.REFERENCE_CHECK.value: """
    你是一名背景调查专家，负责核实候选人的工作经历和表现。
    你需要使用工具获取候选人的推荐信和背景信息。

    评估要点：
    1. 工作经历真实性：核实候选人提供的工作经历是否真实
    2. 工作表现：了解候选人在前公司的表现和贡献
    3. 人际关系：了解候选人的团队合作和沟通能力
    4. 发展潜力：评估候选人的成长空间和潜力

    请生成一个结构化的Word背景调查报告，包含以上要点和详细理由。
    """,

    TaskType.SALARY_NEGOTIATION.value: """
    你是一名薪酬专家，负责分析市场薪资和候选人的期望。
    你需要使用工具获取市场数据和候选人的薪资期望。

    评估要点：
    1. 市场水平：分析当前市场类似职位的薪资范围
    2. 候选人期望：了解候选人的薪资期望和底线
    3. 谈判策略：制定合适的薪资谈判策略
    4. 建议薪资：给出基于市场和候选人能力的薪资建议

    请生成一个结构化的Word薪资分析报告，包含以上要点和详细理由。
    """,

    TaskType.FINAL_EVALUATION.value: """
    你是一名高级招聘专家，负责综合所有评估结果生成最终推荐。
    你需要综合分析所有评估数据，给出最终录用建议。

    评估要点：
    1. 综合评分：基于所有评估维度的综合评分
    2. 优势分析：候选人的主要优势和亮点
    3. 风险分析：候选人的潜在风险和不足
    4. 最终建议：是否录用及理由

    请生成一个结构化的Word最终评估报告，包含以上要点和详细理由。
    """
}

# 智能体工具映射
AGENT_TOOLS = {
    TaskType.REQUIREMENT_ANALYSIS.value: ["general_tool"],
    TaskType.RESUME_SCREENING.value: [
        "search_linkedin_profile",
        "check_github_activity",
        "analyze_resume_compatibility"
    ],
    TaskType.TECHNICAL_ASSESSMENT.value: [
        "run_technical_test",
        "assess_problem_solving_skills",
        "check_github_activity"
    ],
    TaskType.CULTURAL_FIT.value: [
        "check_cultural_fit_indicators",
        "search_professional_references"
    ],
    TaskType.REFERENCE_CHECK.value: [
        "search_professional_references",
        "search_linkedin_profile"
    ],
    TaskType.SALARY_NEGOTIATION.value: [
        "analyze_market_salary",
        "assess_candidate_expectations"
    ],
    TaskType.FINAL_EVALUATION.value: ["general_tool"]  # 最终评估不需要额外工具，使用已有评估结果
}


# 文档生成器
class DocumentGenerator:
    """文档生成器，用于生成各类Word文档"""

    @staticmethod
    def generate_requirement_document(job_description: str, analysis_result: Dict) -> WordDocument:
        """生成招聘需求文档"""
        doc = WordDocument("招聘需求分析文档")
        doc.add_heading("招聘需求分析", 1)
        doc.add_paragraph(f"职位描述: {job_description}")
        doc.add_heading("核心技能要求", 2)
        doc.add_paragraph(analysis_result.get("core_skills", ""))
        doc.add_heading("经验要求", 2)
        doc.add_paragraph(analysis_result.get("experience_requirements", ""))
        doc.add_heading("教育背景", 2)
        doc.add_paragraph(analysis_result.get("education_requirements", ""))
        doc.add_heading("软技能要求", 2)
        doc.add_paragraph(analysis_result.get("soft_skills", ""))
        doc.add_heading("文化匹配要求", 2)
        doc.add_paragraph(analysis_result.get("cultural_fit", ""))

        return doc

    @staticmethod
    def generate_evaluation_document(task_type: str, candidate: Dict, evaluation: Dict) -> WordDocument:
        """生成评估文档"""
        doc_type_map = {
            TaskType.RESUME_SCREENING.value: "简历筛选评估报告",
            TaskType.TECHNICAL_ASSESSMENT.value: "技术能力评估报告",
            TaskType.CULTURAL_FIT.value: "文化匹配度评估报告",
            TaskType.REFERENCE_CHECK.value: "背景调查报告",
            TaskType.SALARY_NEGOTIATION.value: "薪资分析报告",
            TaskType.FINAL_EVALUATION.value: "最终评估报告"
        }

        doc = WordDocument(f"{candidate['name']} - {doc_type_map.get(task_type, '评估报告')}")
        doc.add_heading(f"候选人信息", 2)
        doc.add_table([candidate])

        doc.add_heading("评估结果", 2)
        for key, value in evaluation.items():
            if key not in ["candidate_id", "task_id", "agent_name", "task_type"]:
                if isinstance(value, (str, int, float)):
                    doc.add_paragraph(f"{key}: {value}")
                elif isinstance(value, list):
                    doc.add_paragraph(f"{key}: {', '.join(value)}")
                elif isinstance(value, dict):
                    doc.add_heading(key, 3)
                    for k, v in value.items():
                        doc.add_paragraph(f"{k}: {v}")

        return doc


# Master智能体 - 负责任务拆解和调度
class MasterAgent:
    def __init__(self):
        self.llm = llm
        self.sop_parser = SOPParser()
        self.rule_engine = RuleEngine()

    async def decompose_tasks(self, state: RecruitmentState):
        """基于SOP拆解任务，并生成智能体配置"""
        sop_text = state["sop_definition"]
        job_desc = state["job_description"]

        # 解析SOP
        parsed_sop = self.sop_parser.parse(sop_text)
        tasks = parsed_sop["tasks"]
        dependencies = parsed_sop["dependencies"]

        # 为每个任务分配候选人
        candidate_ids = [candidate["id"] for candidate in state["candidate_profiles"]]
        for task in tasks:
            task["candidate_ids"] = candidate_ids

        # 生成智能体配置
        agent_configs = {}
        for task in tasks:
            task_type = task["type"]
            agent_configs[task["task_id"]] = {
                "name": f"{task_type}_agent",
                "role": f"{task_type}专家",
                "tools": AGENT_TOOLS.get(task_type, []),
                "instructions": AGENT_PROMPTS.get(task_type, "执行评估任务"),
                "task_type": task_type
            }

        return {
            "tasks": tasks,
            "task_dependencies": dependencies,
            "agent_configs": agent_configs,
            "messages": [HumanMessage(content=f"基于SOP拆解了{len(tasks)}个任务")]
        }

    async def assign_tasks(self, state: RecruitmentState):
        """动态分配任务给Worker，考虑任务依赖关系"""
        # 获取所有未分配的任务

        all_tasks = state["tasks"]
        assigned_tasks = state.get("assigned_tasks", [])
        # print("assigned_tasks_input:",state.get("assigned_tasks", []))
        completed_tasks = state.get("completed_tasks", [])
        worker_status = state.get("worker_status", {})

        # 找出已完成的任务ID
        completed_task_ids = [t["task_id"] for t in completed_tasks]

        # 找出可分配的任务（无依赖或依赖已满足）
        available_tasks = []
        for task in all_tasks:
            task_id = task["task_id"]

            # 检查任务是否已完成或已分配
            if task_id in completed_task_ids or task_id=='task_1' or any(t["task_id"] == task_id for t in assigned_tasks):
                continue

            # 检查任务依赖是否满足
            dependencies = state["task_dependencies"].get(task_id, [])
            if all(dep in completed_task_ids for dep in dependencies):
                available_tasks.append(task)

        if len(available_tasks)<1 and "task_1" not in completed_task_ids:
            available_tasks.append(all_tasks[0])
        # 分配可用任务
        for task in available_tasks:
            worker_name = f"{task['type']}_agent"
            if worker_status.get(worker_name, "idle") == "idle":
                task["assigned_to"] = worker_name
                task["assigned_at"] = datetime.now().isoformat()
                assigned_tasks.append(task)
                worker_status[worker_name] = "busy"
        # print("assigned_tasks:",assigned_tasks)
        return {
            "assigned_tasks": assigned_tasks,
            "worker_status": worker_status,
            "messages": [HumanMessage(content=f"分配了 {len(available_tasks)} 个可用任务")]
        }

    async def monitor_progress(self, state: RecruitmentState):
        """监控任务进度和Worker状态"""
        completed_count = len(state.get("completed_tasks", []))
        total_count = len(state.get("tasks", []))

        # 检查是否有任务超时
        current_time = datetime.now()
        for task in state.get("assigned_tasks", []):
            if task.get("completed_at") is None:
                assigned_time = datetime.fromisoformat(task["assigned_at"])
                if (current_time - assigned_time).seconds > 300:  # 5分钟超时
                    # 重置任务并标记worker为空闲
                    task["assigned_to"] = None
                    task["assigned_at"] = None
                    state["worker_status"][task["assigned_to"]] = "idle"

        # 生成进度报告
        progress_report = f"任务进度: {completed_count}/{total_count} 完成"

        return {
            "messages": [HumanMessage(content=progress_report)]
        }

    async def aggregate_results(self, state: RecruitmentState):
        """聚合所有评估结果并生成最终推荐"""
        evaluations = state.get("evaluations", [])

        if not evaluations:
            return {"final_recommendations": [], "messages": [HumanMessage(content="尚无评估结果")]}

        # 按候选人分组评估结果
        candidate_evaluations = {}
        for eval_res in evaluations:
            candidate_id = eval_res.get("candidate_id")
            if candidate_id not in candidate_evaluations:
                candidate_evaluations[candidate_id] = []
            candidate_evaluations[candidate_id].append(eval_res)

        # 为每个候选人生成最终评估
        final_recommendations = []
        for candidate_id, evals in candidate_evaluations.items():
            # 使用LLM分析所有评估结果
            system_msg = SystemMessage(content="你是一个高级招聘专家，需要基于多个评估结果生成最终候选人推荐。")
            human_msg = HumanMessage(content=f"""
            基于以下评估结果，为候选人生成综合评分和最终推荐理由：
            {evals}

            请返回一个JSON格式的推荐，包含：
            - candidate_id: 候选人ID
            - overall_score: 综合评分 (0-100)
            - strengths: 优势列表
            - weaknesses: 弱点列表
            - recommendation: 最终推荐 (strong_yes/yes/no/strong_no)
            - reasoning: 推荐理由
            """)

            response = await self.llm.agenerate([[system_msg, human_msg]])
            print(response.generations[0][0].text)
            print(json.dumps(response.generations[0][0].text))
            recommendation = eval(json.dumps(response.generations[0][0].text))
            final_recommendations.append(json.dumps(recommendation))

        return {
            "final_recommendations": final_recommendations,
            "messages": [system_msg, human_msg, AIMessage(content=str(final_recommendations))]
        }


# 基础Worker类
class BaseWorker:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = llm
        self.name = config.get("name", "unnamed_worker")
        self.role = config.get("role", "通用评估员")
        self.task_type = config.get("task_type", "")

        # 根据配置选择工具
        tool_names = config.get("tools", [])
        self.tools = [tool for tool in tools if tool.name in tool_names]

        # 创建代理执行器
        self.agent_executor = self.create_agent_executor()

    def create_agent_executor(self):
        """创建代理执行器"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"你是一个{self.role}。{self.config.get('instructions', '')}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        # if is_empty(self.tools):
        #     agent =


        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)

    async def execute(self, task: Dict, candidate: Dict, previous_results: Dict) -> Dict:
        """执行任务"""
        # 构建输入
        input_text = f"""
        任务描述: {task['description']}
        职位描述: {task.get('job_description', '无')}
        候选人信息: {candidate}
        """

        # 添加前置任务结果
        if previous_results:
            input_text += f"\n前置任务结果: {json.dumps(previous_results, ensure_ascii=False)}"

        # 执行代理
        result = await self.agent_executor.ainvoke({"input": input_text})

        # 解析结果
        try:
            # 尝试解析JSON结果
            evaluation = eval(result["output"])
        except:
            # 如果无法解析，使用原始输出
            evaluation = {"raw_output": result["output"]}

        # 添加元数据
        evaluation["task_id"] = task["task_id"]
        evaluation["candidate_id"] = candidate["id"]
        evaluation["agent_name"] = self.name
        evaluation["task_type"] = self.task_type

        return evaluation


# Worker工厂
class WorkerFactory:
    def __init__(self):
        self.workers = {}

    def get_worker(self, task_type: str, config: Dict) -> BaseWorker:
        """获取或创建Worker"""
        if task_type not in self.workers:
            self.workers[task_type] = BaseWorker(config)
        return self.workers[task_type]


# 定义图节点
async def decompose_tasks_node(state: RecruitmentState):
    master = MasterAgent()
    result = await master.decompose_tasks(state)
    return result


async def assign_tasks_node(state: RecruitmentState):
    master = MasterAgent()
    result = await master.assign_tasks(state)
    return result


async def execute_tasks_node(state: RecruitmentState):
    """执行所有已分配的任务"""
    completed_tasks = state.get("completed_tasks", [])
    evaluations = state.get("evaluations", [])
    worker_status = state.get("worker_status", {})
    agent_configs = state.get("agent_configs", {})
    generated_documents = state.get("generated_documents", {})

    # 初始化Worker工厂
    worker_factory = WorkerFactory()
    rule_engine = RuleEngine()

    for task in state.get("assigned_tasks", []):
        if task.get("completed_at") is None:
            worker_name = task["assigned_to"]
            candidate_ids = task.get("candidate_ids", [])

            # 获取智能体配置
            config = agent_configs.get(task["task_id"], {})

            # 创建或获取Worker
            worker = worker_factory.get_worker(task["type"], config)

            # 获取前置任务结果
            previous_results = {}
            dependencies = state["task_dependencies"].get(task["task_id"], [])
            for dep_task_id in dependencies:
                # 查找依赖任务的评估结果
                for eval in evaluations:
                    if eval.get("task_id") == dep_task_id and eval.get("candidate_id") in candidate_ids:
                        candidate_id = eval.get("candidate_id")
                        if candidate_id not in previous_results:
                            previous_results[candidate_id] = {}
                        previous_results[candidate_id][dep_task_id] = eval

            # 为每个候选人执行任务
            task_results = {}
            for candidate_id in candidate_ids:
                # 获取候选人信息
                candidate = next((c for c in state["candidate_profiles"] if c["id"] == candidate_id), None)
                if not candidate:
                    continue

                # 添加职位描述到任务中
                task["job_description"] = state["job_description"]

                # 获取该候选人的前置任务结果
                candidate_previous_results = previous_results.get(candidate_id, {})

                # 执行任务
                result = await worker.execute(task, candidate, candidate_previous_results)

                # 保存评估结果
                evaluations.append(result)
                task_results[candidate_id] = result

            # 检查任务规则
            rules = task.get("rules", [])
            rule_evaluation = rule_engine.evaluate_all_rules(rules, task_results, previous_results)

            # 生成文档
            document_key = f"{task['task_id']}_document"
            if task["type"] == TaskType.REQUIREMENT_ANALYSIS.value:
                # 生成招聘需求文档
                doc = DocumentGenerator.generate_requirement_document(
                    state["job_description"], task_results
                )
                filename = doc.save(f"requirement_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                generated_documents[document_key] = filename
            else:
                # 为每个候选人生成评估文档
                for candidate_id, result in task_results.items():
                    candidate = next((c for c in state["candidate_profiles"] if c["id"] == candidate_id), None)
                    if candidate:
                        doc = DocumentGenerator.generate_evaluation_document(
                            task["type"], candidate, result
                        )
                        filename = doc.save(
                            f"{task['type']}_{candidate['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                        if document_key not in generated_documents:
                            generated_documents[document_key] = {}
                        generated_documents[document_key][candidate_id] = filename

            # 更新任务状态
            task["completed_at"] = datetime.now().isoformat()
            task["result"] = f"完成了{len(candidate_ids)}个候选人的评估"
            task["rule_evaluation"] = rule_evaluation
            completed_tasks.append(task)

            # 更新worker状态
            worker_status[worker_name] = "idle"

    return {
        "completed_tasks": completed_tasks,
        "evaluations": evaluations,
        "worker_status": worker_status,
        "generated_documents": generated_documents,
        "messages": [HumanMessage(content=f"执行了 {len(state.get('assigned_tasks', []))} 个任务")]
    }


async def monitor_progress_node(state: RecruitmentState):
    master = MasterAgent()
    result = await master.monitor_progress(state)
    return result


async def aggregate_results_node(state: RecruitmentState):
    master = MasterAgent()
    result = await master.aggregate_results(state)
    return result


def should_continue(state: RecruitmentState) -> str:
    """判断是否继续执行任务"""
    completed_count = len(state.get("completed_tasks", []))
    total_count = len(state.get("tasks", []))

    # 打印调试信息
    completed_task_ids = [t["task_id"] for t in state.get("completed_tasks", [])]
    all_task_ids = [t["task_id"] for t in state.get("tasks", [])]
    uncompleted_tasks = [task_id for task_id in all_task_ids if task_id not in completed_task_ids]
    # print(f"已完成任务: {completed_task_ids}")
    # print(f"未完成任务: {uncompleted_tasks}")

    if completed_count >= total_count:
        print("所有任务已完成，进入结果聚合阶段")
        return "aggregate"
    else:
        # 检查是否有任务可以继续执行
        assigned_tasks = state.get("assigned_tasks", [])
        if not assigned_tasks:
            # 如果没有已分配的任务，但任务未完成，尝试重新分配
            print("没有已分配的任务，尝试重新分配")
            return "continue"
        else:
            # 如果有已分配的任务，继续执行
            print(f"有 {len(assigned_tasks)} 个已分配的任务，继续执行")
            return "continue"


# 创建图
def create_recruitment_workflow():
    builder = StateGraph(RecruitmentState)

    # 添加节点
    builder.add_node("decompose_tasks", decompose_tasks_node)
    builder.add_node("assign_tasks", assign_tasks_node)
    builder.add_node("execute_tasks", execute_tasks_node)
    builder.add_node("monitor_progress", monitor_progress_node)
    builder.add_node("aggregate_results", aggregate_results_node)

    # 设置入口点
    builder.set_entry_point("decompose_tasks")

    # 添加边
    builder.add_edge("decompose_tasks", "assign_tasks")
    builder.add_edge("assign_tasks", "execute_tasks")
    builder.add_conditional_edges(
        "execute_tasks",
        should_continue,
        {
            "continue": "monitor_progress",
            "aggregate": "aggregate_results"
        }
    )
    builder.add_edge("monitor_progress", "assign_tasks")
    builder.add_edge("aggregate_results", END)

    # 编译图
    graph = builder.compile()
    return graph


# 示例使用
async def main():
    # 创建招聘工作流
    workflow = create_recruitment_workflow()

    # 定义SOP流程（包含规则）
    sop_definition = """
    任务1: 需求分析 - 分析职位描述，生成详细的候选人需求文档
    规则: 核心技能要求包含Python; 经验要求>=5年
    依赖: 无

    任务2: 简历筛选 - 评估候选人的基本资格和技能匹配度
    规则: 总体匹配度>=80%; 技能匹配度>=85%
    依赖: 1

    任务3: 技术能力评估 - 进行编程测试和技能验证
    规则: 得分>=80; 代码质量>=良好
    依赖: 2

    任务4: 文化匹配度评估 - 评估候选人与团队和公司文化的适配度
    规则: 文化匹配度评分>=4.0; 价值观匹配>=中等
    依赖: 2,3

    任务5: 背景核实 - 联系推荐人核实工作经历和表现
    规则: 推荐强度>=推荐; 可信度>=中
    依赖: 2,3,4

    任务6: 薪资谈判准备 - 分析市场薪资和候选人期望
    规则: 期望薪资<=市场薪资范围上限; 谈判空间!=无
    依赖: 2,3,4,5

    任务7: 最终评估 - 综合所有评估结果做出录用决定
    规则: 综合评分>=85; 最终推荐!=no
    依赖: 6
    """

    # 准备输入数据
    initial_state = RecruitmentState(
        job_description="我们需要招聘一名高级Python开发工程师，要求有5年以上经验，熟悉Django和FastAPI框架，有机器学习经验者优先。公司文化注重创新和团队合作。",
        candidate_profiles=[
            {
                "id": "candidate_1",
                "name": "张三",
                "experience": "6年Python开发经验",
                "skills": ["Python", "Django", "Flask", "机器学习"],
                "education": "计算机科学硕士",
                "previous_companies": ["公司A", "公司B"],
                "current_salary": "¥42,000",
                "expected_salary": "¥55,000 - ¥65,000"
            },
            {
                "id": "candidate_2",
                "name": "李四",
                "experience": "4年Java开发经验，2年Python经验",
                "skills": ["Java", "Python", "Spring", "微服务"],
                "education": "软件工程学士",
                "previous_companies": ["公司C", "公司D"],
                "current_salary": "¥32,000",
                "expected_salary": "¥40,000 - ¥45,000"
            }
        ],
        sop_definition=sop_definition,
        tasks=[],
        task_dependencies={},
        agent_configs={},
        assigned_tasks=[],
        completed_tasks=[],
        worker_status={},
        generated_documents={},
        evaluations=[],
        final_recommendations=[],
        messages=[]
    )

    # 执行工作流
    config = {"recursion_limit": 50}
    final_state = await workflow.ainvoke(initial_state,config)

    # 打印最终结果
    print("招聘评估完成!")
    print("=" * 50)
    print("最终推荐:")
    print(final_state)
    try:
        for recommendation in final_state["final_recommendations"]:
            print(f"候选人 {recommendation['candidate_id']}: {recommendation['recommendation']}")
            print(f"评分: {recommendation['overall_score']}")
            print(f"优势: {', '.join(recommendation['strengths'])}")
            print(f"弱点: {', '.join(recommendation['weaknesses'])}")
            print(f"理由: {recommendation['reasoning']}")
            print("---")

        # 打印任务执行详情
        print("\n任务执行详情:")
        print("=" * 50)
        for task in final_state["completed_tasks"]:
            print(f"任务ID: {task['task_id']}")
            print(f"任务类型: {task['type']}")
            print(f"执行者: {task['assigned_to']}")
            print(f"完成时间: {task['completed_at']}")
            print(f"规则评估: {task.get('rule_evaluation', {}).get('passed', '未知')}")
            if not task.get('rule_evaluation', {}).get('passed', True):
                print(f"失败规则: {', '.join(task['rule_evaluation'].get('failed_rules', []))}")
            print(f"结果: {task['result']}")
            print("---")

        # 打印生成的文档
        print("\n生成的文档:")
        print("=" * 50)
        for doc_key, doc_info in final_state["generated_documents"].items():
            print(f"文档键: {doc_key}")
            if isinstance(doc_info, dict):
                for candidate_id, filename in doc_info.items():
                    print(f"  候选人 {candidate_id}: {filename}")
            else:
                print(f"  文件: {doc_info}")
    except json.JSONDecodeError as e:
        print(f"JSON解码错误: {e}")
        return None


if __name__ == "__main__":
    asyncio.run(main())

# 需要在现有基础上，将模型调用改成deepseek，然后对每个任务Agent采用longCoT+ReAct的模式进行重构，并模拟必要的action执行结果
