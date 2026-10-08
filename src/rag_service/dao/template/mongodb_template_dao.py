import logging

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import PyMongoError

from src.config import Config
from src.models.template import Template
from src.rag_service.dao.template.base import TemplateDAO


config = Config()
logger = logging.getLogger(__name__)


def _object_id(template_id: str) -> ObjectId | None:
    try:
        return ObjectId(template_id)
    except (InvalidId, TypeError):
        return None


class MongoDBTemplateDAO(TemplateDAO):
    def __init__(self):
        self.client = MongoClient(config.MONGODB_URI)
        self.db = self.client[config.MONGODB_DATABASE]
        self.collection = self.db[config.MONGODB_TEMPLATE_COLLECTION]
        self._create_indexes()

    def _create_indexes(self) -> None:
        try:
            self.collection.create_index([("owner_id", ASCENDING), ("type", ASCENDING)])
        except PyMongoError as e:
            logger.warning(f"Could not create template indexes: {e}")

    def _template_from_mongo(self, template_doc: dict) -> Template:
        template_doc = dict(template_doc)
        template_doc["id"] = str(template_doc.pop("_id"))
        return Template(**template_doc)

    def create(self, template: Template) -> Template:
        template_doc = template.model_dump(exclude={"id"})
        result = self.collection.insert_one(template_doc)
        return template.model_copy(update={"id": str(result.inserted_id)})

    def get_by_id(self, template_id: str) -> Template | None:
        object_id = _object_id(template_id)
        if object_id is None:
            return None
        template_doc = self.collection.find_one({"_id": object_id})
        return self._template_from_mongo(template_doc) if template_doc else None

    def get_by_owner(
        self, owner_id: str, template_type: str | None = None
    ) -> list[Template]:
        query: dict = {"owner_id": owner_id}
        if template_type is not None:
            query["type"] = template_type
        template_docs = self.collection.find(query).sort("created_at", DESCENDING)
        return [self._template_from_mongo(doc) for doc in template_docs]

    def update(self, template: Template) -> Template:
        object_id = _object_id(template.id or "")
        if object_id is None:
            raise ValueError(f"Template ID '{template.id}' is not a valid ObjectId.")

        result = self.collection.update_one(
            {"_id": object_id}, {"$set": template.model_dump(exclude={"id"})}
        )
        if result.matched_count == 0:
            raise ValueError(f"Template with ID {template.id} not found")
        return template

    def delete(self, template_id: str) -> bool:
        object_id = _object_id(template_id)
        if object_id is None:
            return False
        return self.collection.delete_one({"_id": object_id}).deleted_count == 1

    def is_reachable(self) -> bool:
        try:
            self.client.admin.command("ping")
            return True
        except PyMongoError as e:
            logger.error(f"Failed to ping MongoDB: {e}")
            return False
