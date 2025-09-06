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
import aiohttp
import threading

import os

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

