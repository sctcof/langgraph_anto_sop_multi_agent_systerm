import json
import logging
import asyncio
import os
os.chdir(os.getcwd())

from utils.agent_dag_generator import DeepSeekAgentDAGGenerator
from utils.agent_dag_executor import DeepSeekAgentDAGExecutor
from src.utils.common import TaskStatus



# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DeepSeekAgentDAGExecutor")

# DeepSeek API配置
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"  # 根据实际情况调整模型名称
DEEPSEEK_KEY = "sk-350c789908d542b6a7a45bef1c212a98" ##task_招聘_ak


# 示例使用
def main():
    # 从环境变量获取DeepSeek API密钥
    api_key = DEEPSEEK_KEY
    if not api_key:
        logger.error("请设置DEEPSEEK_API_KEY环境变量")
        return

    yaml_file = "recruitment_sop.yaml"
    with open(yaml_file, 'r', encoding='utf-8') as f:
        yaml_content = f.read()

    # 示例YAML内容（替代从文件读取）

    # 创建DAG生成器
    dag_generator = DeepSeekAgentDAGGenerator()

    # 解析YAML
    yaml_data = dag_generator.parse_yaml(yaml_content)

    # 生成Agent
    agents = dag_generator.generate_agents_from_yaml(yaml_data)

    # 构建DAG
    nodes = dag_generator.build_dag_from_yaml(yaml_data, agents)

    # 生成可视化图表
    dag_description = dag_generator.generate_dag_visualization(nodes, "deepseek_dag.png")

    print(dag_description)

    # 保存DAG描述到文件
    with open("dag_description.txt", "w", encoding="utf-8") as f:
        f.write(dag_description)

    # 执行DAG
    executor = DeepSeekAgentDAGExecutor(dag_generator, api_key)

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
            if result.api_calls:
                print(f"   API调用: {result.api_calls}次")
            if result.resource_usage:
                print(f"   资源使用: {result.resource_usage}")

        print(f"\n总共API调用次数: {executor.total_api_calls}")

        # 保存详细结果
        with open("execution_results.json", "w", encoding="utf-8") as f:
            result_dict = {}
            for node_id, result in results.items():
                result_dict[node_id] = {
                    "status": result.status.value,
                    "execution_time": result.execution_time,
                    "api_calls": result.api_calls,
                    "resource_usage": {k.value: v for k, v in
                                       result.resource_usage.items()} if result.resource_usage else None,
                    "error": result.error
                }
            json.dump(result_dict, f, ensure_ascii=False, indent=2)

    # 运行异步执行
    asyncio.run(run_execution())


if __name__ == "__main__":
    main()