"""
实文档向量的存储功能
分为以下3个模块：
初始化与集合管理：创建或加载Milvus向量数据库集合。
    初始化方法：初始化VectorStore类的实例，设置基本参数并调用集合创建或加载方法
    创建或加载集合方法：检查并创建或加载Milvus集合，定义字段结构和索引参数
文档向量化与存储：将分块后的文档转换为向量并存储。
    添加文档方法：将分块后的文档转换为向量并存储到Milvus集合
混合检索与重排序：结合稠密和稀疏向量进行检索，并通过重排序优化结果。

"""
import datetime

from attr.validators import max_len
# 导入 BGE-M3 嵌入函数，用于生成文档和查询的向量表示
from milvus_model.hybrid import BGEM3EmbeddingFunction
# 导入 Milvus 相关类，用于操作向量数据库
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
# 导入 Document 类，用于创建文档对象
from langchain_core.documents import Document
# 导入 CrossEncoder，用于重排序和 NLI 判断
from sentence_transformers import CrossEncoder
# 导入 hashlib 模块，用于生成唯一 ID 的哈希值
import hashlib
from base.config import single_config as config
from base.logger import get_module_logger
logger = get_module_logger(__name__)
import sys
import os
import torch


class VectorStore:
    """
    需求：初始化VectorStore
    思路步骤：
    1. 构造milvus连接相关的参数: host、port、db、collection_name
    2. 构造嵌入模型
    3. 构造rerank模型
    4. 创建或加载milvus集合，保证milvus表存在

    """

    def __init__(self,
                 milvus_host=config.MILVUS_HOST
                 , milvus_port=config.MILVUS_PORT
                 , milvus_database_name=config.MILVUS_DATABASE_NAME
                 , milvus_collection_name=config.MILVUS_COLLECTION_NAME
                 , dim=1024
                 ):
        # 1. 构造milvus连接相关的参数: host、port、db、collection_name.
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.milvus_database_name = milvus_database_name
        self.milvus_collection_name = milvus_collection_name
        self.dim = dim

        self.milvus_client = MilvusClient(uri=f'http://{self.milvus_host}:{self.milvus_port}'
                                          , db_name=self.milvus_database_name)

        # 创建或加载milvus集合，保证milvus表存在
        self._create_or_load_collection()

        # 2. 构造嵌入模型
        embedding_model_path = os.path.join(config.MODELS_DIR, 'bge-m3')
        self.embedding_model = BGEM3EmbeddingFunction(model_name_or_path=embedding_model_path, use_fp16=False,
                                                      device='cuda' if torch.cuda.is_available() else 'cpu')
        logger.debug(f'加载embedding模型成功！')

        # 3. 构造rerank模型
        rerank_model_path = os.path.join(config.MODELS_DIR, 'bge-reranker-large')
        self.rerank_model = CrossEncoder(rerank_model_path,
                                         device='cuda' if torch.cuda.is_available() else 'cpu')
        logger.debug(f'加载rerank模型成功')

    """
    需求：创建或加载milvus集合，保证milvus表存在
    思路步骤：
    1. 判断milvus表是否存在
    2. 如果表不存在，创建milvus表： 
        2.1 建表： 主键、子块内容、稠密向量、稀疏向量、父块id、父块内容、学科、时间戳
        2.2 索引：稠密向量（IVF_FLAT + IP + nlist=128） 、稀疏向量（SPARSE_INVERTED_INDEX + IP + drop_radio_build ）
    3. 加载milvus集合到内存
    """

    def _create_or_load_collection(self):
        #  1. 判断milvus表是否存在
        if not self.milvus_client.has_collection(self.milvus_collection_name):
            logger.info(f'表{self.milvus_collection_name}不存在，执行建表逻辑')
            try:
                #  2. 如果表不存在，创建milvus表：
                #  2.1 建表： 主键、子块内容、稠密向量、稀疏向量、父块id、父块内容、学科、时间戳
                schema = self.milvus_client.create_schema(auto_id=False, enable_dynamic_field=True)
                # 主键
                schema.add_field(field_name='id', datatype=DataType.VARCHAR, max_length=128, is_primary=True)
                # 子块内容
                schema.add_field(field_name='content', datatype=DataType.VARCHAR, max_length=4096)
                # 稠密向量
                schema.add_field(field_name='dense_vector', datatype=DataType.FLOAT_VECTOR, dim=self.dim)
                # 稀疏向量
                schema.add_field(field_name='sparse_vector', datatype=DataType.SPARSE_FLOAT_VECTOR)
                # 父块id
                schema.add_field(field_name='parent_id', datatype=DataType.VARCHAR, max_length=32)
                # 父块内容
                schema.add_field(field_name='parent_content', datatype=DataType.VARCHAR, max_length=10240)
                # 学科
                # TODO 学科后续用来过滤，可以改成分区字段
                schema.add_field(field_name='subject', datatype=DataType.VARCHAR, max_length=32)
                #  时间戳
                schema.add_field(field_name='timestamp', datatype=DataType.VARCHAR, max_length=64)

                #  2.2 索引：稠密向量（IVF_FLAT + IP + nlist=128） 、稀疏向量（SPARSE_INVERTED_INDEX + IP + drop_ratio_build ）

                index_params = self.milvus_client.prepare_index_params()

                # 稠密向量（IVF_FLAT + IP + nlist=128
                index_params.add_index(
                    field_name='dense_vector'
                    , index_name='dense_vector_index'
                    , metric_type='IP'
                    , index_type='IVF_FLAT'
                    , params={
                        'nlist': 128
                    }
                )

                #  2.2 索引：稠密向量（IVF_FLAT + IP + nlist=128） 、
                index_params.add_index(
                    field_name='sparse_vector'
                    , index_name='sparse_vector_index'
                    , metric_type='IP'
                    , index_type='SPARSE_INVERTED_INDEX'
                    , params={
                        'drop_ratio_build': 0.2
                    }
                )

                # 建表
                self.milvus_client.create_collection(
                    collection_name=self.milvus_collection_name,
                    schema=schema,
                    index_params=index_params
                )

                logger.info(f'执行建表：{self.milvus_collection_name}成功')
            except Exception as e:
                logger.error(f'执行建表失败：{self.milvus_collection_name}, 原因：{e}')

        #  3. 加载milvus集合到内存
        try:
            self.milvus_client.load_collection(self.milvus_collection_name)
            logger.debug(f'集合:{self.milvus_collection_name}加载完毕！')
        except Exception as e:
            logger.error(f'执行加载集合{self.milvus_collection_name}失败, 原因{e}')

    """
    需求：将子块添加到milvus中
    思路：
    1. 从子块中获取到的所有的文本
    2. 将文本输入给embedding模型，得到稀疏向量和稠密向量
    3. 构造milvus要插入的数据
        3.1 主键：子块文本进行md5
        3.2 向量：稠密向量、稀疏向量
        3.3 其他信息：子块内容、父块ID、父块内容、时间戳、学科
    4. 使用分批upsert插入milvus (后续更新，直接把之前全部的文档+现有的文档一起写入向量库)
    """

    def add_documents(self, documents):
        # 1. 从子块中获取到的所有的文本
        texts = [document.page_content for document in documents]
        # 2. 将文本输入给embedding模型，得到稀疏向量和稠密向量
        embeddings = self.embedding_model(texts)

        data = []
        # 3. 构造milvus要插入的数据
        for i, document in enumerate(documents):
            # 3.1 主键：子块文本进行md5
            content = document.page_content
            logger.info(f'当前处理第{i}个子块， 大小：{len(content)}')
            id = hashlib.md5(content.encode('utf-8')).hexdigest()
            # 3.2 向量：稠密向量、稀疏向量
            dense_vector = embeddings['dense'][i]
            # 稀疏向量
            # token_id:权重
            sparse_vector = {}
            sparse_token_id_and_value = embeddings['sparse'][[i], :]
            sparse_col = sparse_token_id_and_value.indices
            sparse_data = sparse_token_id_and_value.data
            for token_id, value in zip(sparse_col, sparse_data):
                sparse_vector[token_id] = value
            # 3.3 其他信息：子块内容、父块ID、父块内容、时间戳、学科
            parent_id = document.metadata.get('parent_id', '')
            parent_content = document.metadata.get('parent_content', content)
            logger.info(f'当前处理第{i}个父块， 大小：{len(parent_content)}')

            timestamp = document.metadata.get('timestamp', datetime.datetime.now().isoformat())
            subject = document.metadata.get('subject', 'unknown')

            # document.metadata['parent_id']
            row = {
                'id': id,
                'content': content,
                'dense_vector': dense_vector,
                'sparse_vector': sparse_vector,
                'parent_id': parent_id,
                'parent_content': parent_content,
                'timestamp': timestamp,
                'subject': subject
            }

            data.append(row)

        # TODO 这里是全部把数据放到了一起，然后进行写入。 可以优化成分批次写入，防止内存溢出
        # 4. 使用upsert插入milvus (后续更新，直接把之前全部的文档+现有的文档一起写入向量库)
        if data:
            logger.info(f'开始往{self.milvus_collection_name}写入数据，数据条数:{len(data)}')
            self.milvus_client.upsert(collection_name=self.milvus_collection_name
                                      , data=data)

    """
    需求：实现输入query，返回排序后的topK个父块
    实现思路：
    1. 将query输入给embedding模型，得到稀疏和稠密向量
    2. 将稀疏和稠密向量构造检索对象
        2.1 稠密向量：稠密向量、度量类型（IP）、自定义参数（nprobe = 16）、 k、学科过滤条件
        2.2 稀疏向量：稀疏向量、度量类型（IP）、自定义参数（）、 k、学科过滤条件
    3. 执行混合检索，得到去重父块
        3.1 执行混合检索得到查询结果
        3.2 对父块进行去重复
    4. 将去重后的父块再进行重排序
    """
    def hybrid_search_with_rerank(self,
                                  query
                                  # k 粗排
                                  , k=config.RETRIEVAL_K
                                  # m 精排
                                  , m=config.CANDIDATE_M
                                  , subject_filter=None
                                  ):
        # 1. 将query输入给embedding模型，得到稀疏和稠密向量
        embeddings = self.embedding_model([str(query)])
        dense_vector = embeddings['dense'][0]
        sparse_vector = {}
        embeddings_sparse_ = embeddings['sparse'][[0], :]
        # col = embeddings_sparse_.col
        col = embeddings_sparse_.indices
        data = embeddings_sparse_.data
        for token_id, value in zip(col, data):
            sparse_vector[token_id] = value

        # 2. 将稀疏和稠密向量构造检索对象

        filter_expr = f"subject == {subject_filter}" if subject_filter else ""

        # 2.1 稠密向量：字段名、稠密向量、度量类型（IP）、自定义参数（nprobe = 16）、 k、学科过滤条件
        dense_search_request = AnnSearchRequest(anns_field='dense_vector', data=[dense_vector],
                                                param={'metric_type': 'IP', 'params': {'nprobe': 10}}, limit=k, expr=filter_expr)

        # 2.2 稀疏向量：字段名、稀疏向量、度量类型（IP）、自定义参数（）、 k、学科过滤条件
        sparse_search_request = AnnSearchRequest(anns_field='sparse_vector', data=[sparse_vector],
                                                 param={'metric_type': 'IP','params': {}},
                                                 limit=k, expr=filter_expr)
        # 3. 执行混合检索，得到去重父块
        # 3.1 执行混合检索得到查询结果
        results = self.milvus_client.hybrid_search(collection_name=self.milvus_collection_name,
                                                   reqs=[dense_search_request, sparse_search_request],
                                                   ranker=WeightedRanker(0.7, 1.0),
                                                   output_fields=['content', 'parent_content', 'timestamp', 'subject',
                                                                  'parent_id'], limit=k)[0]
        # 3.2 对父块进行去重复
        # hit: 命中的milvus数据，子块
        # hit {id , distance , entity : {content:   , parent_content: xxxx   }}
        child_chunks = [self._build_child_doc(hit['entity']) for hit in results]
        unique_parent_docs = self._get_unique_parent_docs(child_chunks)

        if unique_parent_docs:
            if len(unique_parent_docs) == 1:
                return unique_parent_docs

            # 4. 将去重后的父块再进行重排序
            # 4.1 构建pairs  [ [query, context1], [query, context2] , [query, context3]...]
            # 这里的parent_docs其实就是context
            # 如果父块超过一个，需要进行重排序： 基于query 和context的匹配程度做重排序
            # 构造： (query, context) 对
            # TODO 注意：这里的query是用户提出的原始的问题，context是查询到的相关上下文
            # rerank模型的作用就是基于rerank模型，再次计算query和context的相关性
            # pairs = [ [query, contex1], [query, contex2]  ,[query, contex3] ....]
            # pairs = [n ,2] , n = 参与rerank的父块的数量
            pairs = [[query, doc.page_content] for doc in unique_parent_docs]
            # 4.2 输入给cross-encoder得到相似度分数
            # scores: 几个父块就对应几个分数
            scores = self.rerank_model.predict(pairs)
            # 4.3 对分数进行排序
            # 排序，按照分数的大小进行倒排。 分数高的排到前面
            # zip: (0.3, 父块1), (0.5,父块2)
            # sorted -> (0.5, 父块2), (0.3,父块1) 。根据元祖的第一个元素进行排序, reverse=True
            sorted_parent_docs = [doc for _, doc in sorted(zip(scores, unique_parent_docs), reverse=True)]
        else:
            sorted_parent_docs = []

        # TODO 最后保留m个父块作为最终的context
        # TODO 切片操作[:m.] -> 切片 相当于 只保存列表中下标从0到m的，[0,3)
        # 长度为 10的 list -> 长度不超过m
        # 类比 字符串的sub_string(0, m)

        # TODO m？ 考虑模型能支持的输入大小
        # 1. 考虑上下文(父块）大小：一个父块1200， 3个父块3600
        # 2. 考虑多轮对话
        # 总之：父块大小 + 多轮对话 + 提示词模板 < 模型上下文大小

        # TODO 注意：这里基于rerank的score排序以后得结果无论和问题的相关性多么小，总是取相对比较大的最大值。
        # TODO 这样会存在一个问题：有可能会查出来和问题完全不相干的。 所以可以在前面增加一个阈值判断

        # 返回m条数据
        # ranked_parent_docs: 相似度分数从大到小的10个父块
        # 通过切片的方式，截取前m个
        #  ranked_parent_docs[0:config.m] -> ranked_parent_docs数组截取下标从0到m, [0,m)
        # TODO 先倒排，再切片 = topM
        return sorted_parent_docs[:m]

    def _build_child_doc(self, hit):
        return Document(
            page_content=hit['content'],
            metadata={
                'parent_id': hit['parent_id'],
                'parent_content': hit['parent_content'],
                'timestamp': hit['timestamp'],
                'subject': hit['subject']
            }
        )

    def _get_unique_parent_docs(self, child_docs):
        unique_parent_docs = []
        parent_content_set = set()
        # 1.遍历所有的子块
        for doc in child_docs:
            # 2.拿到子块中的父块内容
            parent_content = doc.metadata.get('parent_content', doc.page_content)
            # 3. 对父块内容进行去重，并构建父块的文档对象
            # 3.1 父块内容存在且没有重复
            if parent_content and parent_content not in parent_content_set:
                # 3.2 构建父块的对象
                unique_parent_docs.append(
                    Document(
                        page_content=parent_content,
                        metadata={
                            'parent_id': doc.metadata.get('parent_id'),
                            'subject': doc.metadata.get('subject'),
                            'timestamp': doc.metadata.get('timestamp')
                        }
                    )
                )

                # 标记这个父块的内容已经出现过了
                parent_content_set.add(parent_content)

        return unique_parent_docs


if __name__ == '__main__':
    # from rag_qa.core.document_processor import process_documents
    #
    # documents = process_documents(
    #     '/Users/itheima/Documents/黑马/讲课/就业班/edu-rag/北京31期/学生端/04-代码/00001.项目代码_课上/integrated_qa_system/rag_qa/data/ai_data')
    store = VectorStore()
    # store.add_documents(documents)

    results = store.hybrid_search_with_rerank('介绍一下大模型的发展史')
    print(results)