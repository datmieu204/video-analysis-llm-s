# backend/app/api/db.py

from bson import ObjectId
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from backend.app.api.schemas import TranscriptResponse

MONGODB_URL = "mongodb://localhost:27017"
DB_NAME = "transcripts_db"
COLLECTION_NAME = "transcripts"
TARGET_COLLECTION_NAME = "target_transcripts"

client = AsyncIOMotorClient(MONGODB_URL)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]
target_collection = db[TARGET_COLLECTION_NAME]

async def save_transcript(transcript: str, config: dict) -> TranscriptResponse:
    """
    Save the transcript to the database and return a response model.
    """
    record = TranscriptResponse(
        source=config["type_of_source"],
        transcript=transcript,
        config=config,
    )
    result = await collection.insert_one(record.model_dump())
    record.id = str(result.inserted_id)

    return record

async def get_transcript_by_id(transcript_id: str) -> TranscriptResponse:
    """
    Retrieve a transcript by its ID.
    """
    record = await collection.find_one({"_id": ObjectId(transcript_id)})
    if record:
        return TranscriptResponse(**record)
    return None

async def update_transcript_fields(transcript_id: str, **kwargs) -> TranscriptResponse:
    """
    Update specific fields of a transcript by its ID.
    """
    kwargs["updated_at"] = datetime.now()
    result = await collection.update_one(
        {"_id": ObjectId(transcript_id)},
        {"$set": kwargs}
    )
    if result.modified_count == 1:
        return await get_transcript_by_id(transcript_id)
    return None