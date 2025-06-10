import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_neo4j import Neo4jGraph
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from pydantic import BaseModel, Field
from typing import List
from langchain_community.vectorstores import Neo4jVector

def createAgent():
 # retrieve Neo4j credentials and configuration from environment variables
    NEO4J_URI = os.environ["NEO4J_URI"]
    NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
    NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
    NEO4J_DATABASE = os.environ["NEO4J_DATABASE"]

    # retrieve OpenAI API key from environment variables
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # initialize OpenAI embeddings and the GPT-4 model
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, model="gpt-4o")
    # connect to the Neo4j graph database
    kg = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
    )
    # VERY IMPORTANT: If the Neo4j database instance is idle for a while, it'll be paused. If this code is run while the instance is paused,
    # it'll throw an error. In case you encounter an error related to the Neo4j database, contact the person with access to the database
    # so that he/she can resume running the instance. As of now, that person is Joe El-Derjany.

    # initialize a Neo4j vector index for hybrid search
    textbook_graphrag_vector_index = Neo4jVector.from_existing_graph(
        embeddings,
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding",
    )

    # define a Pydantic model for entity extraction
    class Entities(BaseModel):
        """Identifying information about entities."""

        names: List[str] = Field(
            ...,
            description=
            """All the actions, addresses, amounts, arguments, body parts, branches, buildings, building addresses, clinics, concepts,
            conditions, contacts, contact information, coordinates, countries, dates, day surgery centers, dental services, diagnostics,
            documents, entities, events, facilities, facility types, faxes, google ratings, hospitals, identifiers, insurances, licenses,
            license numbers, lists, locations, medical facility types, methods, objects, organizations, parameters, percentages, 
            persons, pharmacies, phone numbers, places, poboxes, policies, processes, providers, provider groups, provider ids, 
            provider start dates, ratings, regions, roles, services, specialties, start dates, streets, subregions, symptoms, teams,
            telephones, and tools that appear in the conversations"""
        )

    # create a chain to extract structured entity information with the LLM
    entity_chain_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """All the actions, addresses, amounts, arguments, body parts, branches, buildings, building addresses, clinics, concepts,
                conditions, contacts, contact information, coordinates, countries, dates, day surgery centers, dental services, diagnostics,
                documents, entities, events, facilities, facility types, faxes, google ratings, hospitals, identifiers, insurances, licenses,
                license numbers, lists, locations, medical facility types, methods, objects, organizations, parameters, percentages, 
                persons, pharmacies, phone numbers, places, poboxes, policies, processes, providers, provider groups, provider ids, 
                provider start dates, ratings, regions, roles, services, specialties, start dates, streets, subregions, symptoms, teams,
                telephones, and tools that appear in the conversations""",
            ),
            (
                "human",
                "Use the given format to extract information from the following "
                "input: {question}",
            ),
        ]
    )
    entity_chain = entity_chain_prompt | llm.with_structured_output(Entities)

    # Create a full-text index named 'entity' on the 'id' property of nodes labeled '__Entity__'
    # This index enables efficient text-based searches on the 'id' property.
    # The 'IF NOT EXISTS' clause ensures the index is only created if it doesn't already exist.
    kg.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

    # function to generate a full-text search query for Neo4j, to be used in the structured retriever function below
    def generate_full_text_query(input: str) -> str:
        """
        Generate a full-text search query for a given input string.

        This function constructs a query string suitable for a full-text search.
        It processes the input string by splitting it into words and appending a
        similarity threshold (~2 changed characters) to each word, then combines
        them using the AND operator. Useful for mapping entities from user questions
        to database values, and allows for some misspelings.
        """
        full_text_query = ""
        words = [el for el in remove_lucene_chars(input).split() if el]
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        full_text_query += f" {words[-1]}~2"
        return full_text_query.strip()

    # function to retrieve structured data from Neo4j based on entities
    def structured_retriever(question: str) -> str:
        """
        Collects the neighborhood of entities mentioned
        in the question
        """
        result = ""
        entities = entity_chain.invoke({"question": question})
        for entity in entities.names:
            print(f" Getting Entity: {entity}")
            response = kg.query(
                """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                YIELD node,score
                CALL {
                WITH node
                MATCH (node)-[r:!MENTIONS]->(neighbor)
                RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                UNION ALL
                WITH node
                MATCH (node)<-[r:!MENTIONS]-(neighbor)
                RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
                }
                RETURN output LIMIT 50
                """,
                {"query": generate_full_text_query(entity)},
            )
            result += "\n".join([el["output"] for el in response])
        return result

    # function to retrieve unstructured data from Neo4j
    def unstructured_retriever(question: str) -> str:
        unstructured_data = [
            el.page_content for el in textbook_graphrag_vector_index.similarity_search(question)
        ]
        return "".join(word.replace("\n", " ") for word in unstructured_data)

    # create a custom GraphRAG prompt based on the predefined ReAct prompt 
    textbook_graphrag_template = """
You are an AI assistant specialized in analyzing WhatsApp customer service conversations to identify relevant company policies. Your task is to process conversations from CSV data and match them with policies stored in a Neo4j graph database for a domestic services company.Add commentMore actions

        ## Input Format
        Conversation Data: Text block containing back and forth messages.
        Message Structure: Customer/Consumer and bot messages.
        Platform: WhatsApp support conversations
        Domain: Domestic worker services (maids, drivers, nannies) and visa processing

        # Analysis Process
        ## Step 1: Policy Retrieval
        Use the `Structured_GraphRAG` tool to retrieve policies using entity-relationship mapping.
        Use the `Unstructured_GraphRAG` tool to retrieve policies using semantic similarity search.
        Combine the outputs from both tools to ensure comprehensive policy coverage.

        ## Step 2: Relevance Assessment
        For each policy retrieved, calculate a `relevance_score` between 0.0 (not relevant) and 1.0 (perfectly relevant) based on the following criteria. A score of 1.0 indicates a policy that directly and completely addresses the core issue of the conversation.
        - **Service Type Alignment:** How well the policy covers the services mentioned (maid, driver, visa, etc.).
        - **Customer Intent Matching:** Whether the policy addresses the customer's specific inquiry, complaint, or issue.Add commentMore actions
        - **Keyword Overlap:** Presence of relevant terms (sickness, pain, hospitals, symptoms, doctor, visa processing, service fees, salaries, etc.).
        - **Bot Response Appropriateness:** How well the policy supports or validates the customer service responses provided.
        - **Communication Style:** Policy alignment with professional WhatsApp standards, including the use of emojis (indicated by `::` like `:happy:`), jargon, or informal language.
        - **Process Relevance:** How well the policy covers relevant processes mentioned in the conversation.
        - **Regulatory Compliance:** Adherence to UAE labor laws, visa requirements, or other regulations.

        ## Step 3: Policy Selection & Filtering
        - Select **only** policies with a `relevance_score` of **0.85 or higher**.
        - After filtering, sort the selected policies in descending order based on their `relevance_score`.
        - Prioritize policies that directly address customer concerns, complaints, and intents when assigning scores.
        - Include policies that guide appropriate bot behavior and service standards.
        - Consider policies related to service guarantees, refunds, or problem resolution.

        ## Step 4: Content Extraction
        For each selected policy that meets the threshold:
        - **Policy ID:** Extract the unique identifier from the database.
        - **Title:** Use the policy's official title.
        - **Relevance Score:** The calculated score (float between 0.85 and 1.0).
        - **Excerpt:** Extract the most relevant portion explaining the core policy rule or guideline.

        # Output Requirements and schema
        ## JSON Format (Strict)
        Provide your final output in a clean JSON object.
        Example Output:

        {
        "policies": [
        {
        "policy_id": "policy_101",
        "title": "Service Replacement Policy",
        "relevance_score": 0.95,
        "excerpt": "If a domestic worker fails to meet service standards within the first 30 days, customers are entitled to a free replacement at no additional cost."
        },
        {
        "policy_id": "policy_305",
        "title": "Symptom Collection Priority",
        "relevance_score": 0.88,
        "excerpt": "Understanding the maid's symptoms is the best way to help determine the right course of action. Collect information about the patient's symptoms before providing clinic information."
        }
        ]
        }

        ## Quality StandardsAdd commentMore actions
        - **Clean JSON only:** Do not include Markdown fences (like ```
        - **Precise excerpts:** Focus on actionable policy content that directly addresses customer concerns.
        - **Accurate scoring:** Ensure the score accurately reflects the true relevance between the conversation context and the policy.
        - **Consistent formatting:** Maintain the exact JSON structure for downstream processing.

        ## Edge CasesAdd commentMore actions
        - **No policies meet threshold:** If no policies have a score of 0.87 or higher, return `{"policies": []}`.
        - **Mixed service inquiries:** Include all policies that meet the threshold for any relevant service types mentioned.
        - **Complaint escalations:** Prioritize policies related to issue resolution and customer satisfaction when calculating scores.

        ## Available ToolsAdd commentMore actions
        - `Structured_GraphRAG`: Retrieves policies through entity-relationship traversal in the Neo4j graph.
        - `Unstructured_GraphRAG`: Retrieves policies through vector similarity search.

        ## Important: Always use BOTH tools to ensure comprehensive policy retrieval. Pay special attention to policies that help resolve customer issues or guide appropriate service recovery actions.        
    """

    react_prompt = hub.pull("langchain-ai/react-agent-template")
    custom_textbook_graphrag_prompt = react_prompt.partial(instructions=textbook_graphrag_template)

    # create tools for structured and unstructured retrieval
    structured_retrieval_tool = Tool(
        name="Structured_GraphRAG",
        func=structured_retriever,
        description="Retrieves relevant documents from the database in a structured form based on the Conversation Messages.",
    )
    unstructured_retrieval_tool = Tool(
        name="Unstructured_GraphRAG",
        func=unstructured_retriever,
        description="Retrieves relevant documents from the database in an unstructured form based on the Conversation Messages.",
    )

    # create a GraphRAG agent using the tools, LLM, and custom prompt
    textbook_graphrag_agent = create_react_agent(
            tools=[structured_retrieval_tool, unstructured_retrieval_tool], llm=llm, prompt=custom_textbook_graphrag_prompt
        )

    # create an executor to run the GraphRAG agent
    textbook_graphrag_agent_executor = AgentExecutor(
        agent=textbook_graphrag_agent,
        tools=[structured_retrieval_tool, unstructured_retrieval_tool],
        verbose=True,
        handle_parsing_errors=True,
        output_parser = StrOutputParser()
    )

    return textbook_graphrag_agent_executor
