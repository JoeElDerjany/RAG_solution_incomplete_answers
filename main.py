from graphrag_agent import createAgent
import csv
import re


def extract_conversations(csv_file_path):
    conversation_dict = {}
    with open(csv_file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for index, row in enumerate(reader):
            conv_id = row.get("conversation_id")
            message = row.get("messages")
            if conv_id and message:
                conversation_dict[index] = (conv_id, message)
    return conversation_dict


def save_dict_to_csv(data_dict):
    with open('output_8th_June.csv', mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        
        # Write the header
        writer.writerow(['Conversation Id','Messages', 'Policies'])
        
        # Write each key-value pair
        for key, value in data_dict.items():
            writer.writerow([value[0], value[1]["input"], value[1]["output"]])


if __name__ == "__main__":
    csv_path = "8th Jun - Sheet1.csv"
    conversations = extract_conversations(csv_path)
    policies_dict = {}
    agent = createAgent()

    for key, value in conversations.items():
        input = re.sub(r'[^a-zA-Z0-9\s:]', '', value[1])
        policies = agent.invoke({"input": input})
        policies_dict[key] = (value[0], policies)

    save_dict_to_csv(policies_dict)
