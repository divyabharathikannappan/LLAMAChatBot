import json, re, time, logging
from typing import AsyncIterable, Dict
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts.chat import ChatPromptTemplate
from langchain.memory import ConversationSummaryBufferMemory

logger = logging.getLogger(__name__)

def validate_query(query: str, max_query_length=500) -> (bool, str):
    if not query.strip():
        return False, "Empty message. Try asking something about grants."
    if len(query) > max_query_length:
        return False, f"Query too long. Limit to {max_query_length} characters."
    if any(k in query.lower() for k in ["hate", "spam", "illegal", "violence"]):
        return False, "Inappropriate content detected."
    return True, "OK"

async def interact_with_user(
    session_id: str,
    query: str,
    chat_sessions: Dict,
    interaction_llm,
    response_llm,
    doc_db,
    interaction_system_prompt: str,
    response_system_prompt: str,
    retrieval_answer_fn
) -> AsyncIterable[str]:
    logger.info(f"[{session_id}] New query: {query}")

    is_valid, message = validate_query(query)
    if not is_valid:
        yield json.dumps({"data": message})
        return

    try:
        if session_id not in chat_sessions:
            memory = ConversationSummaryBufferMemory(
                llm=interaction_llm, memory_key="history", return_messages=True, max_token_limit=2000
            )
            chat_sessions[session_id] = {
                "memory": memory,
                "history": [],
                "user_info": {},
                "stage": "information_gathering",
                "last_search_query": None,
                "retrieved_context": None,
                "question_count": 0,
            }

        session = chat_sessions[session_id]
        memory = session["memory"]
        user_info = session["user_info"]

        prompt = f"""
{interaction_system_prompt}

USER INFORMATION:
{json.dumps(user_info, indent=2)}

CONVERSATION HISTORY:
{"".join([m.content for m in memory.chat_memory.messages])}

CURRENT QUERY: {query}
        """
        logger.debug(f"[{session_id}] Prompt built. Invoking LLM...")

        llm_response = await interaction_llm.ainvoke(prompt)
        session["history"].append(HumanMessage(content=query))
        session["history"].append(AIMessage(content=llm_response))
        memory.save_context({"input": query}, {"output": llm_response})

        logger.info(f"[{session_id}] LLM responded: {llm_response.strip()[:100]}...")

        if "SEARCH QUERY:" in llm_response:
            search_query = re.search(r"SEARCH QUERY:\s*(.*)", llm_response).group(1)
            session["stage"] = "search"
            session["last_search_query"] = search_query
            yield json.dumps({"data": "Searching grants...\n"})
            async for chunk in retrieval_answer_fn(session_id, search_query):
                try:
                    yield chunk
                except Exception as e:
                    logger.error(f"[{session_id}] Streaming error during RAG: {e}")
                    yield json.dumps({"data": "⚠️ Error while retrieving grants."})
            return

        try:
            yield f"data: {llm_response}\\n\\n"

        except Exception as e:
            logger.error(f"[{session_id}] Error during yield: {e}")
            yield json.dumps({"data": "⚠️ Internal error preparing response."})

    except Exception as e:
        logger.exception(f"[{session_id}] Critical failure in interact_with_user")
        yield json.dumps({"data": "⚠️ An unexpected error occurred. Please try again."})