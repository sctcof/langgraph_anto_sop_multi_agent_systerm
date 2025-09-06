import yaml
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json


class RecruitmentSOPParser:
    """招聘SOP流程解析器"""

    def __init__(self):
        # 预定义的关键词和模式
        self.keywords = {
            'position': ['职位', '岗位', '招聘', '聘请', '雇佣'],
            'department': ['部门', '团队', '组'],
            'requirement': ['要求', '条件', '资格', '需具备'],
            'responsibility': ['职责', '责任', '工作内容'],
            'process': ['流程', '步骤', '阶段'],
            'interview': ['面试', '面谈', '复试', '初试'],
            'evaluation': ['评估', '评价', '考核'],
            'offer': ['录用', '聘书', 'offer', '入职'],
            'onboarding': ['入职', '报到', '上岗']
        }

        # 任务节点模板
        self.task_templates = {
            'position_approval': {
                'name': '职位审批',
                'description': '招聘职位申请与审批流程',
                'responsible_party': '部门负责人,HRBP',
                'estimated_duration': '3-5个工作日',
                'prerequisites': [],
                'outputs': ['审批通过的职位需求表']
            },
            'jd_preparation': {
                'name': '职位描述准备',
                'description': '编写和确认职位描述(JD)',
                'responsible_party': '招聘专员,部门负责人',
                'estimated_duration': '1-2个工作日',
                'prerequisites': ['职位审批通过'],
                'outputs': ['正式的职位描述文档']
            },
            'channel_selection': {
                'name': '招聘渠道选择',
                'description': '确定并启动招聘渠道',
                'responsible_party': '招聘专员',
                'estimated_duration': '1个工作日',
                'prerequisites': ['职位描述准备完成'],
                'outputs': ['渠道发布计划', '职位发布链接']
            },
            'resume_screening': {
                'name': '简历筛选',
                'description': '筛选符合条件的候选人简历',
                'responsible_party': '招聘专员,部门代表',
                'estimated_duration': '3-7个工作日',
                'prerequisites': ['招聘渠道已启动'],
                'outputs': ['初选候选人名单']
            },
            'interview_arrangement': {
                'name': '面试安排',
                'description': '安排面试时间和面试官',
                'responsible_party': '招聘协调员',
                'estimated_duration': '1-3个工作日',
                'prerequisites': ['简历筛选完成'],
                'outputs': ['面试时间表', '面试官安排']
            },
            'interview_evaluation': {
                'name': '面试评估',
                'description': '进行面试并评估候选人',
                'responsible_party': '面试官,部门负责人',
                'estimated_duration': '1-2个工作日',
                'prerequisites': ['面试安排完成'],
                'outputs': ['面试评估表', '候选人评分']
            },
            'background_check': {
                'name': '背景调查',
                'description': '对候选人进行背景调查',
                'responsible_party': 'HR专员',
                'estimated_duration': '3-5个工作日',
                'prerequisites': ['面试评估通过'],
                'outputs': ['背景调查报告']
            },
            'offer_approval': {
                'name': '录用审批',
                'description': '审批录用决定和薪酬待遇',
                'responsible_party': '部门负责人,HR负责人',
                'estimated_duration': '2-3个工作日',
                'prerequisites': ['背景调查通过'],
                'outputs': ['审批通过的录用通知书']
            },
            'offer_issuance': {
                'name': '发放录用通知',
                'description': '向候选人发放正式录用通知',
                'responsible_party': '招聘专员',
                'estimated_duration': '1个工作日',
                'prerequisites': ['录用审批通过'],
                'outputs': ['签字的录用通知书']
            },
            'onboarding_preparation': {
                'name': '入职准备',
                'description': '准备新员工入职所需材料和设备',
                'responsible_party': 'HR,IT,行政部门',
                'estimated_duration': '3-5个工作日',
                'prerequisites': ['录用通知已接受'],
                'outputs': ['工位准备', '设备配置', '入职材料']
            }
        }

    def simulate_llm_analysis(self, text: str) -> Dict[str, Any]:
        """
        模拟大模型分析文本，提取SOP信息
        在实际应用中，这里会调用真正的大模型API
        """
        # 这里使用简单的规则模拟大模型分析
        result = {
            'process_name': '招聘流程',
            'department': None,
            'position_level': None,
            'tasks': []
        }

        # 提取部门信息
        department_patterns = [
            r'([\u4e00-\u9fa5]+)部门',
            r'([\u4e00-\u9fa5]+)团队',
            r'([\u4e00-\u9fa5]+)组'
        ]

        for pattern in department_patterns:
            match = re.search(pattern, text)
            if match:
                result['department'] = match.group(1)
                break

        # 提取职位级别
        level_patterns = [
            r'(初级|中级|高级|资深|专家|经理|总监|副总裁|总裁)',
            r'(助理|专员|主管|经理|总监|副总裁|总裁)'
        ]

        for pattern in level_patterns:
            match = re.search(pattern, text)
            if match:
                result['position_level'] = match.group(1)
                break

        # 识别提到的任务节点
        mentioned_tasks = []
        for task_key, task_info in self.task_templates.items():
            if any(keyword in text for keyword in task_info['name']):
                mentioned_tasks.append(task_key)

        # 如果没有明确提到任务，使用默认的任务流程
        if not mentioned_tasks:
            mentioned_tasks = list(self.task_templates.keys())

        # 构建任务列表
        for task_key in mentioned_tasks:
            task = self.task_templates[task_key].copy()
            result['tasks'].append(task)

        return result

    def generate_yaml_output(self, analysis_result: Dict[str, Any]) -> str:
        """生成YAML格式的输出"""

        # 构建YAML数据结构
        yaml_data = {
            'process': {
                'name': analysis_result['process_name'],
                'department': analysis_result['department'] or '待确定',
                'position_level': analysis_result['position_level'] or '待确定',
                'created_date': datetime.now().strftime('%Y-%m-%d'),
                'estimated_completion_days': self._calculate_total_duration(analysis_result['tasks'])
            },
            'tasks': []
        }

        # 添加任务信息
        for i, task in enumerate(analysis_result['tasks']):
            task_data = {
                'id': f"task_{i + 1:03d}",
                'name': task['name'],
                'description': task['description'],
                'responsible_party': task['responsible_party'].split(','),
                'estimated_duration': task['estimated_duration'],
                'prerequisites': task['prerequisites'],
                'outputs': task['outputs'],
                'status': 'pending',
                'start_date': None,
                'completion_date': None
            }
            yaml_data['tasks'].append(task_data)

        # 转换为YAML格式
        yaml_output = yaml.dump(
            yaml_data,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False
        )

        return yaml_output

    def _calculate_total_duration(self, tasks: List[Dict[str, Any]]) -> int:
        """估算总完成时间（工作日）"""
        total_days = 0

        for task in tasks:
            # 从字符串中提取数字
            duration_str = task['estimated_duration']
            numbers = re.findall(r'\d+', duration_str)

            if numbers:
                # 取最大值作为该任务的估计时间
                max_days = max(map(int, numbers))
                total_days += max_days

        return total_days

    def parse_sop_and_generate_yaml(self, sop_text: str) -> str:
        """解析SOP文本并生成YAML输出"""
        # 模拟大模型分析
        analysis_result = self.simulate_llm_analysis(sop_text)

        # 生成YAML
        yaml_output = self.generate_yaml_output(analysis_result)

        return yaml_output


# 示例使用
if __name__ == "__main__":
    # 示例招聘SOP文本
    sop_text = """
    技术部门招聘高级软件工程师的标准流程：
    1. 首先由技术总监审批职位需求
    2. HR与技术部门共同编写职位描述
    3. 选择招聘渠道并发布职位
    4. 招聘专员和技术团队共同筛选简历
    5. 安排技术面试和HR面试
    6. 面试通过后进行背景调查
    7. 发放录用通知书并确认入职时间
    8. 准备入职所需的设备和材料
    """

    # 创建解析器实例
    parser = RecruitmentSOPParser()

    # 解析SOP并生成YAML
    yaml_output = parser.parse_sop_and_generate_yaml(sop_text)

    print("生成的YAML结构化信息:")
    print(yaml_output)

    # 保存到文件
    with open('recruitment_sop.yaml', 'w', encoding='utf-8') as f:
        f.write(yaml_output)

    print("\nYAML文件已保存为 'recruitment_sop.yaml'")