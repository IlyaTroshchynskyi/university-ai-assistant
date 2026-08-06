from types_aiobotocore_dynamodb import DynamoDBClient

from app.api.v1.faculty.repository import FacultyRepository
from app.settings import get_settings


async def delete_test_faculty(faculty_id: str, db_client: DynamoDBClient) -> None:
    repo = FacultyRepository(db_client, get_settings())
    await repo.delete_faculty(faculty_id)
