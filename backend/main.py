import datetime
import uuid
import logging
import json
from fastapi import FastAPI, Depends, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import sqlalchemy as sa
from sqlalchemy import select, delete
from database import init_db, get_db_session, AsyncSession, engine
import loaders as ld
import tables as db
import aiohttp
from ollama_model_resolver import OllamaModelResolver
from ollama_token_tracker import OllamaTokenTracker

MODEL_NAME = "qwen3.5:9b"

logger = logging.getLogger(__name__)
def disable_sqlalchemy_logging():
    for l in ["sqlalchemy", "sqlalchemy.engine", "sqlalchemy.engine.Engine"]:
        sqlalchemy_logger = logging.getLogger(l)
        sqlalchemy_logger.disabled = True
        sqlalchemy_logger.propagate = False
disable_sqlalchemy_logging()

logging.basicConfig()
app = FastAPI(title="LLM Chat Backend")

# Configure CORS to allow Angular (Angular default port is 4200)
origins = [
    "http://localhost:4200", 
    "http://localhost:4200/*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    """Initialize the database when the server starts."""
    await init_db()
    print("Database initialized successfully.")

# === API ENDPOINTS ===

@app.get("/api/conversations")
async def list_conversations(sess: AsyncSession = Depends(get_db_session)):
    """Retrieves a list of all saved conversations."""
    
    result = await sess.execute(select(db.Conversation))
    return result.scalars().all()

@app.get("/api/conversations/{id}/messages")
async def list_conversations(id: str, sess: AsyncSession = Depends(get_db_session)):
    """Retrieves conversation messages."""
    
    req = select(db.Conversation).where(db.Conversation.id==id)
    result = (await sess.scalars(req)).first()
    if result is None:
        raise HTTPException(404)
    
    req2 = select(db.Message).where(db.Message.conversation_id == id)
    result = await sess.scalars(req2)
    return result.all()

@app.delete("/api/conversations/{id}")
async def delete_conversation(id: str, sess: AsyncSession = Depends(get_db_session)):
    """Creates and returns a new empty conversation object."""
    await sess.execute(delete(db.Message).where(db.Message.conversation_id == id))
    await sess.execute(delete(db.Conversation).where(db.Conversation.id == id))
    return ""


@app.post("/api/messages")
async def add_new_message(message: ld.Message, conversationId: str, sess: AsyncSession = Depends(get_db_session)):
    """add a new message to a conversation."""

    id = str(uuid.uuid4())
    sess.add(db.Message(
        id=id,
        conversation_id=conversationId, 
        content=message.content, 
        thinking=message.thinking,
        created_at=datetime.datetime.now(datetime.UTC).isoformat(),
        role=message.role
    ))
    await sess.flush()

    return {"id": id}


@app.post("/api/conversations")
async def create_new_conversation(input: ld.NewConversation, sess: AsyncSession = Depends(get_db_session)):
    """Creates and returns a new empty conversation object."""
    new_conv = db.Conversation(id=str(uuid.uuid4()), title=input.title)
    sess.add(new_conv)
    await sess.commit()
    await sess.refresh(new_conv)
    return new_conv





async def get_http_session():
    async with aiohttp.ClientSession(conn_timeout=10) as httpsess:
        yield httpsess

OLLAMA_URL = "http://localhost:11434/api/chat"

async def chat_endpoint_generator(http_sess: aiohttp.ClientSession, conversation: ld.Conversation):
    try:
        # Call Ollama with streaming
        query_body = {
            "model": MODEL_NAME,
            "messages": [m.model_dump() for m in conversation.messages],
            "stream": True
        }
        print("query_body=",query_body)
        async with http_sess.post(
            OLLAMA_URL,
            json=query_body,
        ) as response:

            # Process streaming response
            if response.status != 200:
                logger.error("response code %s", response.status)
                print("err!!")
                yield "{\"error\":true,\"done\":true}\n"
                return
                
            async for chunk_line in response.content:
                if chunk_line:
                    data = json.loads(chunk_line)
                    print(".", end="", flush=True)
                    message = data["message"]
                    if "done" in data and data["done"]:
                        print(json.dumps(data, indent=2))
                    response_chunk =  data
                    yield json.dumps(response_chunk)+"\n"

            # full_response = "".join(chunk["chunk"] for chunk in ollama_stream)
            # final_token_count = len(full_response) // 4

            # total_tokens = initial_context_tokens + final_token_count

            # # Save messages to database
            # if conversation_id:
            #     # Create assistant message
            #     assistant_msg = db.Message(
            #         conversation_id=conversation_id,
            #         role="assistant",
            #         content=full_response,
            #         timestamp=None
            #     )
            #     sess.add(assistant_msg)
            #     await sess.commit()    


    except Exception as e:
        logger.exception("error")
        print("err2!!")
        yield "{\"error\":true,\"done\":true}\n"
        return
    
@app.post("/api/chat")
async def chat_endpoint(
    conversation: ld.Conversation,
    conversation_id: int|None = None,
    sess: AsyncSession = Depends(get_db_session),
    http_sess: aiohttp.ClientSession = Depends(get_http_session)
):
    print(f"--- Received chat request for {len(conversation.messages)} messages ---")

    # Calculate initial context tokens (simplified - in production use tokenizer)
    token_count = sum(len(msg.content) for msg in conversation.messages)
    initial_context_tokens = token_count + 1245  # Base token overhead

    # Call Ollama API for streaming response
    
    ollama_stream = []
    print("conversation=",conversation)

    resolver = OllamaModelResolver()
    result = resolver.resolve_model(MODEL_NAME)
    gguf_path = result["blobs"][0]

    tracker = OllamaTokenTracker(model_path=gguf_path, max_context=65535)
    for msg in conversation.messages:
        if msg.role == 'assistant' and isinstance(msg.thinking , str) and len(msg.thinking) > 0:
            tracker.add_message(msg.role, msg.thinking)
        tracker.add_message(msg.role, msg.content)

    print("\n"*10)
    print(tracker.summary())
    print("\n"*10)

    if conversation_id:
        # Fetch conversation messages from DB
        result = await sess.execute(
            select(db.Message)
            .join(db.Conversation, db.Message.conversation_id == db.Conversation.id)
            .where(db.Conversation.id == conversation_id)
            .order_by(db.Message.id)
        )
        db_messages = result.scalars().all()
        for msg in db_messages:
            conversation.messages.append(ld.Message(role=msg.role, content=msg.content))

    return StreamingResponse(
        chat_endpoint_generator(http_sess, conversation), 
        media_type="application/x-ndjson"
    )





def print_content(data:bytes, point_in_code: str):
    try:
        print(f"{point_in_code=} data=",json.dumps(json.loads(data.decode()), indent=2))
    except Exception:
        data_s = [ d.strip() for d in data.decode().split("\n") if d.strip() != ""]
        for d in data_s:
            print(d)

    

@app.get("/{full_path:path}")
async def catch_all_get(full_path: str,response:Response, http_sess: aiohttp.ClientSession = Depends(get_http_session)):

    async with http_sess.get("http://localhost:11434/"+full_path) as r:
        data = await r.content.read()

        print_content(data, "RESPONSE catch_all_get")

        response.status_code = r.status
        return data


@app.post("/{full_path:path}")
async def catch_all_post(full_path: str, request:Request,response:Response, http_sess: aiohttp.ClientSession = Depends(get_http_session)):



    body = await request.body()
    print_content(body, "QUERY catch_all_post")
    async with http_sess.post("http://localhost:11434/"+full_path, data=body) as r:
        data = await r.content.read()

        print_content(data, "RESPONSE catch_all_post")
        response.status_code = r.status
        return data

@app.put("/{full_path:path}")
async def catch_all_put(full_path: str, request:Request,response:Response, http_sess: aiohttp.ClientSession = Depends(get_http_session)):
    body = await request.body()
    print_content(body, "QUERY catch_all_put")

    async with http_sess.put("http://localhost:11434/"+full_path, data=body) as r:
        data = await r.content.read()
        response.status_code = r.status
        print_content(data, "RESPONSE catch_all_put")

        return data
