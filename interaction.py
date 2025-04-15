import json, re, logging
from typing import AsyncIterable, Dict
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationSummaryBufferMemory

logger = logging.getLogger("interaction")

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
        yield f"data: {message}\n\n"
        return

    try:
        if session_id not in chat_sessions:
            memory = ConversationSummaryBufferMemory(
                llm=interaction_llm, memory_key="history", return_messages=True, max_token_limit=2000
            )
            chat_sessions[session_id] = {
                "memory": memory,
                "history": [],
                "question_count": 0,
                "stage": "information_gathering",
                "last_search_query": None,
                "retrieved_context": None,
            }

        session = chat_sessions[session_id]
        memory = session["memory"]
        question_count = session.get("question_count", 0)

        # Build prompt
        prompt = f"""
{interaction_system_prompt}

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

        # Smart fallback
        trigger_phrases = ["grants for", "funding for", "looking for grants", "apply for"]
        session["question_count"] += 1
        force_search = session["question_count"] >= 2 

        if force_search and "SEARCH QUERY:" not in llm_response:
            logger.warning(f"[{session_id}] Fallback search triggered — generating SEARCH QUERY from user input.")

            # Store LLM's actual message
            clarification = llm_response

            # Now set the new search query string
            llm_response = f"SEARCH QUERY: {query}"

            # Stream the LLM's message line-by-line before fallback triggers
            for line in clarification.splitlines():
                logger.debug(f"[{session_id}] YIELDING: {line!r}")
                yield f"{line}\n\n"

        if "SEARCH QUERY:" in llm_response:
            search_query = re.search(r"SEARCH QUERY:\s*(.*)", llm_response).group(1)
            session["stage"] = "search"
            session["last_search_query"] = search_query
            yield "data: Searching for relevant grants...\n\n"
            async for chunk in retrieval_answer_fn(session_id, search_query):
                try:
                    yield f"{chunk}\n\n"
                except Exception as e:
                    logger.error(f"[{session_id}] Error while streaming RAG: {e}")
                    yield "data: ⚠️ Error during document retrieval.\n\n"
            yield "data: [DONE]\n\n"
            return

        for line in llm_response.splitlines():
             logger.debug(f"[{session_id}] YIELDING (normal): {line!r}")
             yield f"{line}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.exception(f"[{session_id}] Unexpected failure in interact_with_user")
        yield "data: ⚠️ An unexpected error occurred.\n\n"
        yield "data: [DONE]\n\n"