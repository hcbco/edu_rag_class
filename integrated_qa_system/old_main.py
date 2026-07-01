"""
需求：实现V1.3的edu-rag项目
思路步骤：
1. 初始化：bm25search、大模型、rag_qa_system
2. 将用户query发送给faq模块进行处理，得到答案和是否进入RAG模块的标识
3. 如果答案可靠，直接返回
4. 如果答案不可靠，且需要进入RAG模块，调用rag模块，得到答案
5. 如果答案不可靠，且不需要进入RAG模块，直接走兜底话术
"""
import time

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from base.logger import get_module_logger
logger = get_module_logger('main')
from base.config import single_config as config

from mysql_qa.retreival.bm25_search import Bm25Search
from mysql_qa.cache.redis_client import RedisClient
from mysql_qa.db.mysql_client import MysqlClient

from rag_qa.core.rag_system import RagSystem


class IntegratedQASystem:
    # 1. 初始化：bm25search、大模型、rag_qa_system
    def __init__(self):
        # bm25search
        self.config = config
        self.logger = logger
        self.db = MysqlClient()
        self.cache = RedisClient()
        self.bm25_search = Bm25Search(
            mysql_client=self.db,
            redis_client=self.cache
        )

        # 大模型
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            api_key=config.DASHSCOPE_API_KEY,
            base_url=config.DASHSCOPE_BASE_URL,
            temperature=0.1,
        )

        # rag_qa_system
        self.rag_system = RagSystem()

    def _call_model(self, prompt, count_down=3):
        start_time = time.time()
        logger.debug(f'开始调用大模型，最大重试次数: {count_down}')
        for i in range(count_down):
            try:
                messages = [
                    SystemMessage('你是一个有用的助手，能够准确无误的完成用户的命令'),
                    HumanMessage(prompt),
                ]
                result = self.llm.invoke(messages).content
                elapsed = time.time() - start_time
                logger.debug(f'大模型调用成功，第 {i + 1} 次尝试, 耗时: {elapsed:.2f}s')
                return result
            except Exception as e:
                logger.error(f'大模型调用失败，第 {i + 1}/{count_down} 次尝试，原因: {e}')
                if i < count_down - 1:
                    logger.debug(f'等待 {1} 秒后重试...')
                    time.sleep(1)
                    continue
        elapsed = time.time() - start_time
        logger.warning(f'大模型调用失败，已重试 {count_down} 次，放弃调用, 总耗时: {elapsed:.2f}s')
        return None

    def _fallback_answer(self):
        """兜底话术：当所有模块都无法返回有效答案时使用"""
        phone = self.config.CUSTOMER_SERVICE_PHONE
        return f'抱歉，系统繁忙，暂时无法为您解答。请拨打客服电话 {phone} 获取帮助。'

    def search(self, query, subject_filter=None):
        try:
            # 2. 将用户query发送给faq模块进行处理，得到答案和是否进入RAG模块的标识
            answer, need_rag = self.bm25_search.search(query)
        except Exception as e:
            logger.error(f'FAQ模块调用异常: {e}')
            return self._fallback_answer()

        # 3. 如果答案可靠，直接返回
        if answer:
            return answer

        # 4. 如果答案不可靠，且需要进入RAG模块，调用rag模块，得到答案
        if need_rag:
            try:
                answer = self.rag_system.generate_answer(query, subject_filter)
            except Exception as e:
                logger.error(f'RAG模块调用异常: {e}')
                return self._fallback_answer()
            if answer:
                return answer

        # 5. 兜底话术
        return self._fallback_answer()


if __name__ == '__main__':
    qa_system = IntegratedQASystem()
    qa_system_search = qa_system.search('如何创建线程安全的单例对象')
    time.sleep(5)
    print(qa_system_search)