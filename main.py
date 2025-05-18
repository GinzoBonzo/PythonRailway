# main.py
import os
import json
import time
import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import logging
import chromadb
from chromadb.config import Settings
import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from flask import Flask, request, jsonify
import asyncio

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
BOT_USERNAME = os.getenv("BOT_USERNAME")  # Your bot's username including the '@'
MAX_PROMPT_WORDS = 750
SHORT_TERM_DAYS = 2

# Initialize Flask app
app = Flask(__name__)

# Initialize bot application
bot_application = None

def init_bot():
    """Initialize the Telegram bot application."""
    global bot_application
    if bot_application is None:
        bot_application = Application.builder().token(TELEGRAM_TOKEN).build()
        
        # Add handlers
        bot_application.add_handler(CommandHandler("start", start))
        bot_application.add_handler(CommandHandler("help", help_command))
        bot_application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        
        # Start the bot in a separate thread
        import threading
        bot_thread = threading.Thread(target=bot_application.run_polling)
        bot_thread.daemon = True
        bot_thread.start()

# Initialize bot when the Flask app starts
@app.before_first_request
def before_first_request():
    init_bot()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy"}), 200

@app.route('/webhook', methods=['POST'])
def webhook():
    """Webhook endpoint for Telegram updates."""
    if request.method == "POST":
        update = Update.de_json(request.get_json(), bot_application.bot)
        asyncio.run(handle_message(update, None))
        return jsonify({"status": "ok"}), 200
    return jsonify({"status": "error"}), 400

# ChromaDB setup
def setup_chroma():
    """Set up and return the ChromaDB client and collection."""
    client = chromadb.HttpClient(
        host=os.getenv("CHROMA_HOST", "localhost"), 
        port=int(os.getenv("CHROMA_PORT", "8000"))
    )
    
    # Create collection if it doesn't exist
    try:
        collection = client.get_collection("chat_memory")
        logger.info("Connected to existing ChromaDB collection")
    except:
        collection = client.create_collection(
            name="chat_memory",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Created new ChromaDB collection")
    
    return client, collection

# Initialize ChromaDB
chroma_client, chat_memory_collection = setup_chroma()

# In-memory cache for short-term memory
short_term_memory = []

# Helper Functions
def get_embedding(text: str) -> List[float]:
    """Get embedding for text using OpenRouter's embeddings API."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Using OpenRouter's embedding endpoint (you could replace with direct API calls to a specific model)
    response = requests.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers=headers,
        json={"input": text, "model": "openai/text-embedding-ada-002"}
    )
    
    if response.status_code == 200:
        return response.json()["data"][0]["embedding"]
    else:
        logger.error(f"Error getting embedding: {response.text}")
        # Return a default embedding (all zeros) as fallback
        return [0.0] * 1536

def store_in_long_term_memory(message_id: str, content: str, user_id: str, timestamp: int):
    """Store a message in long-term memory (ChromaDB)."""
    embedding = get_embedding(content)
    
    metadata = {
        "user_id": user_id,
        "timestamp": timestamp,
        "content": content[:1000]  # Store truncated content in metadata for quick access
    }
    
    # Add to ChromaDB
    chat_memory_collection.add(
        ids=[message_id],
        embeddings=[embedding],
        metadatas=[metadata],
        documents=[content]
    )
    
    logger.info(f"Stored message {message_id} in long-term memory")

def add_to_short_term_memory(message: Dict[str, Any]):
    """Add a message to short-term memory (in-memory cache)."""
    short_term_memory.append(message)
    
    # Prune old messages from short-term memory
    cutoff_time = time.time() - (SHORT_TERM_DAYS * 24 * 60 * 60)
    global short_term_memory
    short_term_memory = [msg for msg in short_term_memory if msg["timestamp"] > cutoff_time]

def query_memory(query: str, limit: int = 5) -> List[str]:
    """Query both memory systems for relevant context."""
    # Get relevant messages from long-term memory
    embedding = get_embedding(query)
    results = chat_memory_collection.query(
        query_embeddings=[embedding],
        n_results=limit
    )
    
    long_term_results = []
    if results["documents"]:
        for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
            username = metadata.get("user_id", "Unknown")
            timestamp = metadata.get("timestamp", 0)
            date_str = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
            long_term_results.append(f"[{date_str}] {username}: {doc}")
    
    # Get recent messages from short-term memory, sorted by time
    recent_messages = sorted(short_term_memory, key=lambda x: x["timestamp"], reverse=True)[:limit]
    short_term_results = []
    for msg in recent_messages:
        date_str = datetime.datetime.fromtimestamp(msg["timestamp"]).strftime("%Y-%m-%d %H:%M")
        short_term_results.append(f"[{date_str}] {msg['user_id']}: {msg['content']}")
    
    # Combine results, prioritizing short-term memory
    all_results = short_term_results + [r for r in long_term_results if r not in short_term_results]
    return all_results[:limit]

def create_prompt(query: str, context: List[str], username: str) -> str:
    """Create a prompt for the LLM within the word limit."""
    base_prompt = f"""
You are a helpful assistant in a Telegram group chat. 
User {username} has asked: "{query}"

Here is some relevant context from the chat history:
{chr(10).join(context)}

Answer the question based on the context provided. If the context doesn't contain relevant information, 
say so and answer with your general knowledge. Keep your answer concise and helpful.
"""
    
    # Count words and trim if necessary
    words = base_prompt.split()
    if len(words) > MAX_PROMPT_WORDS:
        # Trim context to fit within word limit
        excess_words = len(words) - MAX_PROMPT_WORDS + 50  # Leave some buffer
        words_per_context = len(" ".join(context).split())
        
        if words_per_context > excess_words:
            # Recalculate context with fewer items
            context_to_keep = []
            current_words = 0
            for ctx in context:
                ctx_words = len(ctx.split())
                if current_words + ctx_words < words_per_context - excess_words:
                    context_to_keep.append(ctx)
                    current_words += ctx_words
                else:
                    break
            
            # Rebuild prompt with reduced context
            base_prompt = f"""
You are a helpful assistant in a Telegram group chat. 
User {username} has asked: "{query}"

Here is some relevant context from the chat history:
{chr(10).join(context_to_keep)}

Answer the question based on the context provided. If the context doesn't contain relevant information, 
say so and answer with your general knowledge. Keep your answer concise and helpful.
"""
    
    return base_prompt

def generate_response(prompt: str) -> str:
    """Generate a response using OpenRouter API."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "anthropic/claude-3-haiku-20240307",  # Use appropriate model
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500
    }
    
    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            logger.error(f"Error from OpenRouter: {response.text}")
            return "Sorry, I couldn't generate a response at this time."
    except Exception as e:
        logger.error(f"Exception when calling OpenRouter: {str(e)}")
        return "Sorry, I encountered an error while generating a response."

# Telegram Bot Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    await update.message.reply_text(
        "Hi! I'm your group chat assistant. I'll listen to the conversation and respond when mentioned."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text(
        "I'm a chat assistant with memory. Just mention me in your message, and I'll try to help!"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming messages."""
    if not update.message or not update.message.text:
        return
    
    message_text = update.message.text
    user_id = update.effective_user.username or str(update.effective_user.id)
    message_id = str(update.message.message_id)
    timestamp = int(update.message.date.timestamp())
    
    # Store message in memory systems
    message_data = {
        "id": message_id,
        "content": message_text,
        "user_id": user_id,
        "timestamp": timestamp
    }
    
    # Add to short-term memory
    add_to_short_term_memory(message_data)
    
    # Add to long-term memory
    store_in_long_term_memory(message_id, message_text, user_id, timestamp)
    
    # Check if bot is being addressed
    bot_mentioned = BOT_USERNAME in message_text
    
    if bot_mentioned:
        # Remove the bot's username from the query
        query = message_text.replace(BOT_USERNAME, "").strip()
        
        # Start typing indicator
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        # Get relevant context from memory
        context_messages = query_memory(query, limit=10)
        
        # Create prompt
        prompt = create_prompt(query, context_messages, user_id)
        
        # Generate response
        response_text = generate_response(prompt)
        
        # Send response
        await update.message.reply_text(response_text)
        
        # Also store the bot's response in memory
        bot_message_data = {
            "id": f"bot_response_{message_id}",
            "content": response_text,
            "user_id": BOT_USERNAME.replace("@", ""),
            "timestamp": int(time.time())
        }
        add_to_short_term_memory(bot_message_data)
        store_in_long_term_memory(
            bot_message_data["id"], 
            bot_message_data["content"], 
            bot_message_data["user_id"], 
            bot_message_data["timestamp"]
        )