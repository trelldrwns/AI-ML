from browser_use_sdk import BrowserUseSdk

TODAY = "2025-08-15"

PEOPLE = [
    # ...
    {
        "name": "Abhilash N S Reddy",
        "location": "Bangalore, Karnataka",
        "needs": "Information about AI & ML",
    },
]

sdk = BrowserUseSdk(api_key="insert your own API key here")

for person in PEOPLE:
    result = sdk.run(
        llm_model="o3",
        task=f"""
You need info about Introduction to AI & ML

Find different sources that provide a great deal of knowledge on this topic.

  - from {person['location']}
  - on {TODAY}
  - for {person['name']}
  - who needs {person['needs']}.

List the top 6 sites that provide a great deal of information on that topic and also why them.
""",
    )