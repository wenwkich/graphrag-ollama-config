import asyncio
import os
import tiktoken
import pandas as pd
from dotenv import load_dotenv

from graphrag.query.indexer_adapters import read_indexer_entities, read_indexer_reports
from graphrag.query.structured_search.global_search.community_context import GlobalCommunityContext
from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.typing import OpenaiApiType

load_dotenv('.env')
join = os.path.join

async def global_search(query, input_dir, community_level=2, temperature=0.5, response_type="Multiple Paragraphs"):
        api_key = os.environ["GRAPHRAG_API_KEY"]
        llm_model = os.environ["GRAPHRAG_LLM_MODEL"]
        api_base = os.environ["GRAPHRAG_LLM_API_BASE"]

        llm = ChatOpenAI(
            api_key=api_key,
            api_base=api_base,
            model=llm_model,
            api_type=OpenaiApiType.OpenAI,  
            max_retries=10,
        )

        token_encoder = tiktoken.get_encoding("cl100k_base")

        COMMUNITY_REPORT_TABLE = "create_final_community_reports"
        ENTITY_TABLE = "create_final_nodes"
        ENTITY_EMBEDDING_TABLE = "create_final_entities"
        
        entity_df = pd.read_parquet(join(input_dir, f"{ENTITY_TABLE}.parquet"))
        report_df = pd.read_parquet(join(input_dir, f"{COMMUNITY_REPORT_TABLE}.parquet"))
        entity_embedding_df = pd.read_parquet(join(input_dir, f"{ENTITY_EMBEDDING_TABLE}.parquet"))

        reports = read_indexer_reports(report_df, entity_df, community_level)
        entities = read_indexer_entities(entity_df, entity_embedding_df, community_level)

        context_builder = GlobalCommunityContext(
            community_reports=reports,
            entities=entities,
            token_encoder=token_encoder,
        )

        context_builder_params = {
            "use_community_summary": False,  # False means using full community reports. True means using community short summaries.
            "shuffle_data": True,
            "include_community_rank": True,
            "min_community_rank": 0,
            "community_rank_name": "rank",
            "include_community_weight": True,
            "community_weight_name": "occurrence weight",
            "normalize_community_weight": True,
            "max_tokens": 4000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
            "context_name": "Reports",
        }

        map_llm_params = {
            "max_tokens": 1000,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }

        reduce_llm_params = {
            "max_tokens": 1000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 1000-1500)
            "temperature": temperature,
        }

        search_engine = GlobalSearch(
            llm=llm,
            context_builder=context_builder,
            token_encoder=token_encoder,
            max_data_tokens=5000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
            map_llm_params=map_llm_params,
            reduce_llm_params=reduce_llm_params,
            allow_general_knowledge=False,  # set this to True will add instruction to encourage the LLM to incorporate general knowledge in the response, which may increase hallucinations, but could be useful in some use cases.
            json_mode=True,  # set this to False if your LLM model does not support JSON mode.
            context_builder_params=context_builder_params,
            concurrent_coroutines=1,
            response_type=response_type,  # free form text describing the response type and format, can be anything, e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
        )

        result = await search_engine.asearch(query)
        return result.response

async def main():
    await global_search("give me a summary to the book", "output\\20240722-222016\\artifacts")

if __name__ == "__main__":
    asyncio.run(main())