"""Tests for the teacher client."""
import os
from teacher.client import TeacherClient


def test_client_initialization():
    """Test that the client initializes with API key from env."""
    if not os.getenv("GROQ_API_KEY"):
        from dotenv import load_dotenv
        load_dotenv()

    client = TeacherClient()
    assert client.model == "llama-3.3-70b-versatile"
    assert client.client is not None


def test_simple_query():
    """Test a simple query to the teacher model."""
    if not os.getenv("GROQ_API_KEY"):
        from dotenv import load_dotenv
        load_dotenv()

    client = TeacherClient()
    response = client.query("What is 2 + 2? Reply with just the number.")
    assert response is not None
    assert len(response) > 0
    print(f"Teacher response: {response}")


if __name__ == "__main__":
    test_client_initialization()
    print("Client initialization test passed!")

    test_simple_query()
    print("Simple query test passed!")

    print("All teacher client tests passed!")
