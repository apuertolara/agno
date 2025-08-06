from typing import Any, Callable, Dict, List, Optional

from agno.knowledge.agent import AgentKnowledge
from agno.knowledge.document import Document
from agno.utils.log import log_debug, logger


class LangChainKnowledgeBase(AgentKnowledge):
    loader: Optional[Callable] = None

    vectorstore: Optional[Any] = None
    search_kwargs: Optional[dict] = None

    knowledge_retriever: Optional[Any] = None

    def search(
        self, query: str, num_documents: Optional[int] = None, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Returns relevant documents matching the query"""

        try:
            from langchain_core.documents import Document as LangChainDocument
            from langchain_core.retrievers import BaseRetriever
        except ImportError:
            raise ImportError(
                "The `langchain` package is not installed. Please install it via `pip install langchain`."
            )

        if self.vectorstore is not None and self.knowledge_retriever is None:
            log_debug("Creating knowledge_retriever")
            if self.search_kwargs is None:
                self.search_kwargs = {"k": self.num_documents}
            if filters is not None:
                self.search_kwargs.update(filters)
            self.knowledge_retriever = self.vectorstore.as_retriever(search_kwargs=self.search_kwargs)

        if self.knowledge_retriever is None:
            logger.error("No knowledge_retriever provided")
            return []

        if not isinstance(self.knowledge_retriever, BaseRetriever):
            raise ValueError(f"Knowledge retriever is not of type BaseRetriever: {self.knowledge_retriever}")

        _num_documents = num_documents or self.num_documents
        log_debug(f"Getting {_num_documents} relevant documents for query: {query}")
        lc_documents: List[LangChainDocument] = self.knowledge_retriever.invoke(input=query)
        documents = []
        for lc_doc in lc_documents:
            documents.append(
                Document(
                    content=lc_doc.page_content,
                    meta_data=lc_doc.metadata,
                )
            )
        return documents

    def load(
        self,
        recreate: bool = False,
        upsert: bool = True,
        skip_existing: bool = True,
        filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.loader is None:
            logger.error("No loader provided for LangChainKnowledgeBase")
            return
        self.loader()

    def exists(self) -> bool:
        logger.warning("LangChainKnowledgeBase.exists() not supported - please check the vectorstore manually.")
        return True
