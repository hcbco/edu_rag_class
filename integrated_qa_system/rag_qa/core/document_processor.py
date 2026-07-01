"""
需求：实现文档从读取到分成子块的全流程
功能：
1. 加载文档并转成List[Document]
2. List[Document]文档切成父子块

"""

import os
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
# 这里为什么导包可以直接使用edu_document_loaders这一层，而不是进入到每个具体的文件中去导入？
# TODO 因为提前在__init__.py中已经把这个包下面所有的模块的类都统一做了导入
from rag_qa.edu_document_loaders import OCRPDFLoader, OCRIMGLoader, OCRDOCLoader, OCRPPTLoader

# 导入分割器
from rag_qa.edu_text_spliter.edu_chinese_recursive_text_splitter import ChineseRecursiveTextSplitter
from langchain_text_splitters import MarkdownTextSplitter

from base.logger import single_logger as logger
from base.config import single_config as config
from datetime import datetime

"""
需求：加载文档并转成List[Document]
思路步骤：
1. 遍历给定的目录
2. 判断目录下的文件是否是支持的
    2.1 对于支持的文件类型，使用对应的加载器进行加载
    2.2 对于不支持的文件类型，直接跳过
3. 给每个文档加入元数据：学科、路径（后缀名）、时间戳

"""

document_loaders = {
    '.txt': TextLoader,
    '.md': UnstructuredMarkdownLoader,
    '.doc': OCRDOCLoader,
    '.docx': OCRDOCLoader,
    '.ppt': OCRPPTLoader,
    '.pptx': OCRPPTLoader,
    '.jpg': OCRIMGLoader,
    '.png': OCRIMGLoader,
    '.pdf': OCRPDFLoader
}


def load_documents_from_directory(directory_path):
    documents = []
    supported_extensions = document_loaders.keys()
    # 1. 遍历给定的目录
    # os.walk：递归遍历指定的路径
    # root: 当前所在的绝对路径(/Users/itheima/Documents/黑马/讲课/就业班/edu-rag/北京31期/学生端/04-代码/00001.项目代码_课上/integrated_qa_system/rag_qa/core)
    # _(dirs): 当前目录下的文件夹(base)
    # files : 当前目录下的文件(LLM基础.pptx)

    # 获取source
    # directory_path -> (/Users/itheima/Documents/黑马/讲课/就业班/edu-rag/北京31期/ai_data
    # os.path.basename -> ai_data -> replace -> ai
    # 数据的路径要求： {学科名}_data

    subject = os.path.basename(directory_path).replace("_data", "")

    for root, _, files in os.walk(directory_path):
        for file_name in files:
            # 2. 判断目录下的文件是否是支持的
            # os.path.splitext: 分割文件名和后缀  split/ext -> (文件名,后缀)
            _, extension = os.path.splitext(file_name)
            file_path = os.path.join(root, file_name)
            if extension in supported_extensions:
                try:
                    # 2.1 对于支持的文件类型，使用对应的加载器进行加载
                    loader_class_name = document_loaders[extension]
                    # 拼接文件的完整路径
                    if '.txt' == extension:
                        loader = loader_class_name(file_path=file_path, encoding='utf-8')
                    else:
                        loader = loader_class_name(file_path)
                    # 每个文件返回的就是一个List[Document]
                    loaded_documents = loader.load()
                    # 3. 给每个文档加入元数据：学科、路径（后缀名）、时间戳
                    for loaded_document in loaded_documents:
                        # Document: page_content:str, 文档的原文; metadata(叫元数据，但是并不必纠结。就是给用户用来存放额外信息):dict, 文档的附加信息
                        loaded_document.metadata['timestamp'] = datetime.now().isoformat()
                        loaded_document.metadata['subject'] = subject
                        loaded_document.metadata['extension'] = extension

                    # 把每个文件加载的文档全部放到documents中返回
                    documents.extend(loaded_documents)

                except Exception as e:
                    logger.error(f'处理文件:{file_path}失败，原因:{e}')
            else:
                # 2.2 对于不支持的文件类型，直接跳过
                logger.warning(f'处理文件:{file_path}失败，不支持的类型')

    return documents


"""
需求：将文档切成父子块
步骤：
1. 调用加载文档的API，获取到所有的文档
2. 构造文档的spliter，有4种情况：md格式-父块、md格式-子块、txt格式-父块、txt格式子块
3. 按层次进行遍历和加载: 
    3.1 遍历所有的文档，分成父块
    3.2 遍历所有的父块，分成子块；给父块增加元数据
    3.3 遍历所有的子块，给子块增加元数据 

最终milvus表里的字段：
0. 主键          【子块提供-page_content】
1. 稠密向量（语义向量）, 固定长度列表 [0.006965979002416134,-0.019829535856842995,....]    【子块提供-page_content】
2. 稀疏向量（词频向量）, {"6":0.038418810814619064,"12":0.0549} 6:词表下标    【子块提供-page_content】
3. 父块内容     【子块提供-metadata】
4. 子块id       【子块提供-metadata】
5. 父块id       【子块提供-metadata】
6. 时间戳       【子块提供-metadata】
7. 学科         【子块提供-metadata】

"""


def process_documents(
        directory_path: str,
        parent_chunk_size=config.PARENT_CHUNK_SIZE,
        child_chunk_size=config.CHILD_CHUNK_SIZE,
        overlap_size=config.CHUNK_OVERLAP
):
    # 1. 调用加载文档的API，获取到所有的文档
    documents = load_documents_from_directory(directory_path)

    # 2. 构造文档的spliter，有4种情况：md格式-父块、md格式-子块、txt格式-父块、txt格式子块
    parent_markdown_spliter = MarkdownTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=overlap_size)
    child_markdown_spliter = MarkdownTextSplitter(chunk_size=child_chunk_size, chunk_overlap=overlap_size)
    parent_text_spiter = ChineseRecursiveTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=overlap_size)
    child_text_spiter = ChineseRecursiveTextSplitter(chunk_size=child_chunk_size, chunk_overlap=overlap_size)


    child_chunks = []

    # 3. 按层次进行遍历和加载:
    for i, document in enumerate(documents):
        # 3.1 遍历所有的文档，分成父块
        extension = document.metadata['extension']

        if_markdown = (extension == '.md')

        parent_spliter_to_use = parent_markdown_spliter if if_markdown else parent_text_spiter
        child_spliter_to_use = child_markdown_spliter if if_markdown else child_text_spiter

        parent_docs = parent_spliter_to_use.split_documents([document])

        # 3.2 遍历所有的父块，分成子块；给父块增加元数据
        for j , parent_doc in enumerate(parent_docs):
            parent_id = f"doc_{i}_parent_{j}"

            current_child_chunks = child_spliter_to_use.split_documents([parent_doc])
            # 3.3 遍历所有的子块，给子块增加元数据
            for k , child_chunk in enumerate(current_child_chunks):
                """
                最终milvus表里的字段：
                0. 主键          【子块提供-page_content】
                1. 稠密向量（语义向量）, 固定长度列表 [0.006965979002416134,-0.019829535856842995,....]    【子块提供-page_content】
                2. 稀疏向量（词频向量）, {"6":0.038418810814619064,"12":0.0549} 6:词表下标    【子块提供-page_content】
                3. 父块内容     【子块提供-metadata】
                4. 子块id       【子块提供-metadata】
                5. 父块id       【子块提供-metadata】
                6. 时间戳       【子块提供-metadata】
                7. 学科         【子块提供-metadata】

                """

                child_id = f"{parent_id}_child_{k}"
                child_chunk.metadata['child_id'] = child_id
                child_chunk.metadata['parent_id'] = parent_id
                child_chunk.metadata['parent_content'] = parent_doc.page_content
                child_chunk.metadata['process_time'] = document.metadata['timestamp']
                child_chunk.metadata['subject'] = document.metadata['subject']

            child_chunks.extend(current_child_chunks)

    return child_chunks


if __name__ == '__main__':
    # documents = load_documents_from_directory(
    #     '/Users/itheima/Documents/黑马/讲课/就业班/edu-rag/北京31期/学生端/04-代码/00001.项目代码_课上/integrated_qa_system/rag_qa/data')
    # print(len(documents))
    documents = process_documents(
        '/Users/itheima/Documents/黑马/讲课/就业班/edu-rag/北京31期/学生端/04-代码/00001.项目代码_课上/integrated_qa_system/rag_qa/data/ai_data')

    print(len(documents))