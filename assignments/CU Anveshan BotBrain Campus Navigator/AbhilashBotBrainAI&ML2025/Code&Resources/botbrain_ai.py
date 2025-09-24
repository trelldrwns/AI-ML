import ollama
import json

# A list of the official, exact names of locations used in your graph.
# This is crucial for validating the LLM's output.
VALID_LOCATIONS = [
    "Main Gate", "Junction1", "Admin Block", "Library", "Stamba", "AcadA",
    "AcadB", "AcadC", "Junction2", "Faculty Housing", "Foodcourt & Laundry",
    "Hostel", "Sports Complex", "Cricket Ground", "Other Sports Grounds",
    "OuterJunction1", "OuterJunction2", "OuterJunction3", "OuterJunction4",
    "OuterJunction5", "OuterJunction6"
]

def parse_user_query(user_text: str) -> tuple[str | None, str | None]:
    """
    Uses a local LLM (phi-3:mini) to parse a natural language query and extract
    a valid source and destination.

    Args:
        user_text: The natural language input from the user.

    Returns:
        A tuple containing (source, destination) if successful, otherwise (None, None).
    """
    # This detailed prompt guides the LLM to give us a clean, predictable JSON output.
    prompt = f"""
    You are a parsing assistant for a campus navigation bot. Your only job is to extract the source and destination from the user's text.
    The locations MUST be one of these exact names: {VALID_LOCATIONS}.
    Map common names to their official names (e.g., "academic block a" -> "AcadA", "hostel" -> "Hostel").
    Respond ONLY with a single, clean JSON object containing "source" and "destination" keys. Do not add any other text or explanations.

    Example 1:
    User: "Show me the way from the main gate to the admin building"
    {{"source": "Main Gate", "destination": "Admin Block"}}

    Example 2:
    User: "I'm at the hostel and need to go to the library"
    {{"source": "Hostel", "destination": "Library"}}
    
    Example 3:
    User: "route from food court to cricket ground"
    {{"source": "Foodcourt & Laundry", "destination": "Cricket Ground"}}

    Now, parse this user's text:
    User: "{user_text}"
    """
    try:
        response = ollama.chat(
            model='phi3:mini',
            messages=[{'role': 'user', 'content': prompt}],
            format='json', # Force the model to output valid JSON
        )
        data = json.loads(response['message']['content'])
        
        source = data.get('source')
        destination = data.get('destination')

        # Final validation to ensure the LLM didn't hallucinate a location
        if source in VALID_LOCATIONS and destination in VALID_LOCATIONS:
            return source, destination
        return None, None
    except Exception as e:
        print(f"An error occurred while parsing the query: {e}")
        return None, None

def get_building_info(location_name: str) -> str:
    """
    Retrieves information about a building from the campus_info.json file.
    This acts as the 'Retrieval' step in a simple RAG system.
    """
    try:
        with open('campus_info.json', 'r') as f:
            knowledge_base = json.load(f)
        return knowledge_base.get(location_name, "No specific information available.")
    except FileNotFoundError:
        return "Knowledge base file (campus_info.json) not found."
