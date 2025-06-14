import pandas as pd
import csv

# Number of messages to append from above/below
n = 5

def split_messages(messages):
    # Split by newlines, ignore empty lines
    return [m for m in messages.split('\n') if m.strip()]

def join_messages(messages):
    return '\n'.join(messages)

def process_group(df_group):
    rows = df_group.reset_index(drop=True)
    processed_rows = []

    i = 0
    while i < len(rows):
        row = rows.iloc[i]
        # Merge neighboring BOT rows
        if row['Agent Name'] == 'BOT':
            merged_messages = split_messages(row['Messages'])
            j = i + 1
            while j < len(rows) and rows.iloc[j]['Agent Name'] == 'BOT':
                merged_messages += split_messages(rows.iloc[j]['Messages'])
                j += 1
            # Now, append n messages from above and below if possible
            # Above
            if i > 0 and rows.iloc[i-1]['Conversation Id'] == row['Conversation Id']:
                above_msgs = split_messages(rows.iloc[i-1]['Messages'])
                merged_messages = above_msgs[-n:] + merged_messages
            # Below
            if j < len(rows) and rows.iloc[j]['Conversation Id'] == row['Conversation Id']:
                below_msgs = split_messages(rows.iloc[j]['Messages'])
                merged_messages = merged_messages + below_msgs[:n]
            # Create a new row with merged messages
            new_row = row.copy()
            new_row['Messages'] = join_messages(merged_messages)
            processed_rows.append(new_row)
            i = j
        else:
            processed_rows.append(row)
            i += 1
    return pd.DataFrame(processed_rows)

def fix_garbled_text(text):
    """
    Fix garbled text by re-encoding and decoding using utf-8.
    """
    try:
        return text.encode('latin-1').decode('utf-8')
    except Exception:
        return text

def main():
    df = pd.read_csv('segmented conversations 14th june - Sheet1.csv', encoding='utf-8')
    processed = []
    for conv_id, group in df.groupby('Conversation Id', sort=False):
        # Skip groups where all Agent Name == 'BOT'
        if (group['Agent Name'] == 'BOT').all():
            continue
        processed_group = process_group(group)
        processed.append(processed_group)
    result = pd.concat(processed, ignore_index=True)
    result = result[result['Agent Name'] == 'BOT'] # keep only BOT convos
    result = result[['Conversation Id', 'Last Skill', 'Agent Name', 'Messages']] # keeps only these cols

    # Fix garbled text in Messages column
    result['Messages'] = result['Messages'].apply(fix_garbled_text)

    # Save to new CSV
    result.to_csv('segmented_processed.csv', index=False, quoting=csv.QUOTE_ALL)

if __name__ == "__main__":
    main()
