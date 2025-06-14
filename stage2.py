import csv
import json
import os
import dotenv
import requests
import re

dotenv.load_dotenv()

def extract_json_from_policies(policies):
    """
    Extract the first JSON object from a string, ignoring any markdown/code block wrappers or extra text.
    """
    if not policies:
        return {}
    # Remove markdown code block markers and leading/trailing whitespace
    cleaned = policies.strip()
    cleaned = re.sub(r'^```json', '', cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'^```', '', cleaned).strip()
    cleaned = re.sub(r'```$', '', cleaned).strip()
    # Find the first JSON object in the string
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except Exception as e:
            return {}
    return {}

def process_csv(input_csv, output_file, prompt, api_key):
    """Process the CSV file and generate output based on the given prompt."""
    try:
        with open(input_csv, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
    except FileNotFoundError:
        print(f"Error: The file {input_csv} was not found.")
        return
    except Exception as e:
        print(f"Error reading the file {input_csv}: {e}")
        return

    results = []
    for row in rows:
        conversation_id = row.get('Conversation Id', 'Unknown')
        messages = row.get('Messages', '')
        policies = row.get('Policies', '')

        try:
            # Use the new extraction function
            policies_data = extract_json_from_policies(policies)
            if not policies_data:
                raise ValueError("Could not extract valid JSON from policies field.")
            api_output = call_openai_api(messages, policies_data, prompt, api_key)
            # --- Fix: Remove markdown code block before parsing ---
            cleaned_output = api_output.strip()
            if cleaned_output.startswith("```json"):
                cleaned_output = cleaned_output[len("```json"):].strip()
            if cleaned_output.startswith("```"):
                cleaned_output = cleaned_output[len("```"):].strip()
            if cleaned_output.endswith("```"):
                cleaned_output = cleaned_output[:-3].strip()
            try:
                parsed_output = json.loads(cleaned_output)
            except json.JSONDecodeError as e:
                print(f"✗ Error parsing API response for conversation {conversation_id}: {e}")
                parsed_output = {"error": f"Error parsing API response: {e}", "raw": api_output}
            results.append({
                "conversation_id": conversation_id,
                "output": parsed_output
            })
            print(f"✓ Processed conversation {conversation_id}")
        except Exception as e:
            error_msg = f"Error: {e}"
            print(f"✗ Error processing conversation {conversation_id}: {e}")
            results.append({
                "conversation_id": conversation_id,
                "output": {"error": error_msg}
            })

    # Write results to output file in JSON format
    try:
        with open(output_file, 'w', encoding='utf-8') as jsonfile:
            json.dump(results, jsonfile, indent=4, ensure_ascii=False)
        print(f"✓ Results successfully saved to {output_file}")
    except Exception as e:
        print(f"Error writing to JSON file {output_file}: {e}")

def call_openai_api(messages, policies, prompt, api_key):
    """Call the OpenAI API with the given messages, policies, and prompt."""
    api_url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Format the input for the API
    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Messages: {messages}\nPolicies: {json.dumps(policies, indent=2)}"},
        ],
        "temperature": 0,
    }

    try:
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"
    except KeyError as e:
        return f"Error parsing API response: {e}"

def generate_output_from_csv(input_csv, output_file):
    """Main function to process the CSV and generate output."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: Please set your OPENAI_API_KEY environment variable")
        exit(1)

    # PROMPT
    prompt = """
    You are an AI assistant tasked with analyzing conversations between a bot and users. Your goal is to evaluate whether the bot's responses in the conversation adhere to the provided company policies. If any violations are detected, you must identify the violated policies and summarize the nature of the violation.
    DO NOT OVERLOOK EXCEPTIONS in the policies, they are crucial for determining if a violation has occurred.
    YOU ONLY HAVE TO ASSESS THE BOT'S RESPONSES, NOT THE AGENT'S MESSAGES. If a bot violates a policy, you must detect the violation, even if the agent happened to correct that violation.
    For each conversation, provide the output in JSON format with the following structure:
    {
    "policy_violated": <true/false>,
    "policies_violated": [
        {

        "title": "<policy_title>",
        "description": "<policy_description>"
        },
        ...
    ],
    "violation_summary": "<summary_of_violation>"
    }

    ### Guidelines:
    1. **Policy Evaluation**: Compare the bot's responses in the conversation against the provided policies. Determine if the bot's responses are incorrect, or misleading based on the policies. NOTE: some conversations have one or more agent's messages which are human input; do not compare these to policies because the task is to assess bot performance; they are only used for context and a peek to what the conversation progressed into. Add agent's intervention to the summary (to be explained) if it is relevant.
    2. **Policy Details**: For each violated policy, include the `title` and `description` in the `policies_violated` array.
    3. **Violation Summary**: Provide a concise summary explaining how the bot's responses violated the policies.
    4. **Policy Exceptions**: For every policy, carefully read and consider the `exceptions` field in the policy JSON. If a policy has exceptions, ensure that you do not mark a violation if the bot's behavior is allowed by any exception. For example, if the exception says "when speaking to a client about a sick maid, ask them to speak to the maid directly before collecting symptoms," and the bot does this, do NOT mark a violation for not collecting symptoms first. Always reference the exception text when making your decision.
    5. **No Violations**: If no violations are detected, set `"policy_violated": false`, leave the `policies_violated` array empty, and make the summary  "No policy violations detected in the conversation."
    6. **DO NOT OVERLOOK EXCEPTIONS** in the policies, they are crucial for determining if a violation has occurred.

    ### Example Outputs:

    #### Example 1: Policy Violated
    {
    "conversation_id": "CH38242fdf##########",
    "policy_violated": true,
    "policies_violated": [
        {
        "title": "Redirecting to Medical Facilities & Providing Links",
        "description": "Provide the clinic link only when you have assessed the symptoms and determined that a clinic visit is necessary."
        }
    ],
    "violation_summary": "The bot provided clinic links prematurely."
    }

    #### Example 2: No Policy Violations
    {
    "conversation_id": "CHf8dbf4131b###########",
    "policy_violated": false,
    "policies_violated": [],
    "violation_summary": "No policy violations detected in the conversation."
    }

    ### Input Format:
    - **Messages**: The conversation between the bot and the user.
    - **Policies**: A list of company policies in JSON format; the JSON includes the exceptions for each policy, please read them carefully.

    Analyze the conversation carefully and provide the output in the specified JSON format. Do not include any extra characters, markdown, or explanations outside the JSON object.
    """

    process_csv(input_csv, output_file, prompt, api_key)

def create_final_policies_csv(input_csv, processed_json, final_csv_path="final_policies.csv"):
    """
    Create a CSV with columns:
    conv_id, Messages, policies_related, policies_violated, policies_high_relevance
    - policies_related: full value of the "policies" key (as JSON string) from input_csv
    - policies_violated: full value of the "output" key (as JSON string) from processed_output.json
    - policies_high_relevance: comma-separated policy titles from input_csv with relevance_score > 0.9
    """
    # Load processed output
    with open(processed_json, 'r', encoding='utf-8') as f:
        processed = json.load(f)

    # Map conversation_id to full output JSON
    output_map = {}
    for item in processed:
        conv_id = item.get("conversation_id")
        output = item.get("output", {})
        output_map[conv_id] = json.dumps(output, ensure_ascii=False)

    # Prepare and write the final CSV
    with open(input_csv, 'r', encoding='utf-8') as infile, \
         open(final_csv_path, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = [
            "conv_id",
            "Messages",
            "policies_related",
            "policies_violated",
            "policies_high_relevance"
        ]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            conv_id = row.get("Conversation Id")
            messages = row.get("Messages", "")
            policies_field = row.get("Policies", "")
            # Parse policies JSON and extract full "policies" value and high relevance titles
            try:
                cleaned = re.sub(r'^```json', '', policies_field.strip(), flags=re.IGNORECASE).strip()
                cleaned = re.sub(r'^```', '', cleaned).strip()
                cleaned = re.sub(r'```$', '', cleaned).strip()
                policies_json = json.loads(cleaned)
                policies_list = policies_json.get("policies", [])
                # Pretty-print policies_related
                policies_related = json.dumps(policies_list, indent=2, ensure_ascii=False)
                high_rel_titles = [p.get("title", "") for p in policies_list if p.get("relevance_score", 0) > 0.9]
            except Exception:
                policies_related = ""
                high_rel_titles = []

            # Pretty-print policies_violated
            policies_violated_raw = output_map.get(conv_id, "")
            try:
                policies_violated_json = json.loads(policies_violated_raw)
                policies_violated = json.dumps(policies_violated_json, indent=2, ensure_ascii=False)
            except Exception:
                policies_violated = policies_violated_raw

            writer.writerow({
                "conv_id": conv_id,
                "Messages": messages,
                "policies_related": policies_related,
                "policies_violated": policies_violated,
                "policies_high_relevance": ",".join(high_rel_titles)
            })

# Usage
if __name__ == "__main__":
    input_csv = 'output_14th_June.csv'
    output_file = '14th_Jun_processed_output.json'
    generate_output_from_csv(input_csv, output_file)
    create_final_policies_csv(input_csv, output_file)
