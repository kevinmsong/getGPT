import os
import requests
import json
import chainlit as cl
from openai import OpenAI
from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain_community.utilities import PubMedAPIWrapper
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import time
import logging
from Bio import Entrez
from Bio import Medline
import io
import re
from collections import Counter
from langchain.schema import HumanMessage, SystemMessage
import csv

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize the API key (replace with your actual API key)
os.environ["OPENAI_API_KEY"] = "sk-None-PVtbsS676iagEfhPaYUqT3BlbkFJqtiyuLiLA9sfmc7h6Syg"
llm = ChatOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    temperature=0.7,
    max_tokens=1024
)

# OpenTargets API URL
graphql_url = "https://api.platform.opentargets.org/api/v4/graphql"
genetics_graphql_url = "https://api.genetics.opentargets.org/graphql"

# Set up Entrez email (replace with your email)
Entrez.email = "your.email@example.com"

def execute_query(query, variables=None, max_retries=3, initial_delay=1):
    payload = {"query": query, "variables": variables} if variables else {"query": query}
    
    logger.debug(f"Sending query to OpenTargets API: {json.dumps(payload, indent=2)}")
    
    for attempt in range(max_retries):
        try:
            response = requests.post(graphql_url, json=payload)
            logger.debug(f"Full API Response: {response.text}")
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 400:
                logger.error(f"Bad Request (400) - Full response: {response.text}")
                error_message = f"Bad Request (400) - API Response: {response.text}"
                raise ValueError(error_message)
            else:
                response.raise_for_status()
        
        except requests.RequestException as e:
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                raise
            time.sleep(initial_delay * (2 ** attempt))  # Exponential backoff

    raise Exception("Max retries reached")

def execute_genetics_query(query, variables=None, max_retries=3, initial_delay=1):
    payload = {"query": query, "variables": variables} if variables else {"query": query}
    
    logger.debug(f"Sending query to OpenTargets Genetics API: {json.dumps(payload, indent=2)}")
    
    for attempt in range(max_retries):
        try:
            response = requests.post(genetics_graphql_url, json=payload)
            logger.debug(f"Full Genetics API Response: {response.text}")
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 400:
                logger.error(f"Bad Request (400) - Full response: {response.text}")
                error_message = f"Bad Request (400) - Genetics API Response: {response.text}"
                raise ValueError(error_message)
            else:
                response.raise_for_status()
        
        except requests.RequestException as e:
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                raise
            time.sleep(initial_delay * (2 ** attempt))  # Exponential backoff

    raise Exception("Max retries reached")

def extract_gene_names(text):
    gene_pattern = r'\b[A-Z][A-Z0-9]+\b'
    potential_genes = re.findall(gene_pattern, text)
    return potential_genes

async def query_pubmed_for_abstracts(disease_name, max_results=5):
    try:
        handle = Entrez.esearch(db="pubmed", term=f"{disease_name} differentially expressed genes", retmax=max_results, sort="relevance")
        record = Entrez.read(handle)
        handle.close()

        if not record["IdList"]:
            return "No relevant PubMed articles found.", []

        handle = Entrez.efetch(db="pubmed", id=record["IdList"], rettype="medline", retmode="text")
        records = Medline.parse(handle)
        articles = list(records)
        handle.close()

        abstracts = []
        for article in articles:
            title = article.get('TI', 'N/A')
            abstract = article.get('AB', 'N/A')
            abstracts.append(f"Title: {title}\n\nAbstract: {abstract}\n\n")

        return "\n".join(abstracts)

    except Exception as e:
        logger.exception(f"Error querying PubMed: {str(e)}")
        return f"An error occurred while querying PubMed: {str(e)}"
 
async def query_opentargets(disease_name):
    try:
        # Search for disease ID
        search_query = """
        query($diseaseText: String!) {
          search(queryString: $diseaseText, entityNames: ["disease"], page: {index: 0, size: 1}) {
            hits {
              id
              name
            }
          }
        }
        """
        search_data = execute_query(search_query, {"diseaseText": disease_name})
        
        if not search_data.get("data", {}).get("search", {}).get("hits"):
            return f"No disease found for '{disease_name}'"
        
        disease_id = search_data["data"]["search"]["hits"][0]["id"]
        disease_name = search_data["data"]["search"]["hits"][0]["name"]
        logger.info(f"Disease search successful. ID: {disease_id}, Name: {disease_name}")

        result = f"Information for {disease_name} (ID: {disease_id}):\n\n"

        # 1. OpenTargets Genetics results
        variants_query = """
        query StudyVariants($studyId: String!) {
          manhattan(studyId: $studyId) {
            associations {
              variant {
                id
                rsId
              }
              pval
              bestGenes {
                score
                gene {
                  id
                  symbol
                }
              }
            }
          }
        }
        """
        
        study_id = "GCST90002369"  # This is a placeholder and should be replaced with a method to find the correct study ID for the disease
        
        variants_data = execute_genetics_query(variants_query, {"studyId": study_id})
        
        if 'errors' in variants_data:
            logger.error(f"GraphQL errors in variants query: {json.dumps(variants_data['errors'], indent=2)}")
            result += f"\nError fetching variants data: {variants_data['errors'][0]['message']}\n"
        else:
            result += "OpenTargets Genetics Results:\n"
            genetics_genes = {}
            associations = variants_data["data"]["manhattan"]["associations"]
            for association in associations[:10]:  # Limit to top 10 for brevity
                variant = association["variant"]
                result += f"Variant: {variant['id']} (rsID: {variant['rsId']})\n"
                result += f"p-value: {association['pval']}\n"
                result += "Best Genes:\n"
                for best_gene in association["bestGenes"]:
                    gene = best_gene["gene"]
                    genetics_genes[gene['symbol']] = best_gene['score']
                    result += f"  - {gene['symbol']} (ID: {gene['id']}, Score: {best_gene['score']})\n"
                result += "\n"
        
        # 2. PubMed and ChatGPT results
        abstracts = await query_pubmed_for_abstracts(disease_name)
        chatgpt_genes_text = await extract_genes_with_chatgpt(abstracts, disease_name)

        result += "\nPubMed and ChatGPT Results:\n"
        result += "Differentially expressed genes extracted by ChatGPT from PubMed abstracts:\n"
        result += chatgpt_genes_text + "\n"

        # Extract gene symbols from ChatGPT response
        chatgpt_genes = set(re.findall(r'\b[A-Z][A-Z0-9]+\b', chatgpt_genes_text))

        # 3. OpenTargets results
        disease_query = """
        query($diseaseId: String!) {
          disease(efoId: $diseaseId) {
            id
            name
            associatedTargets(page: {index: 0, size: 100}) {
              count
              rows {
                target {
                  id
                  approvedSymbol
                  approvedName
                }
                score
              }
            }
          }
        }
        """
        disease_data = execute_query(disease_query, {"diseaseId": disease_id})
        
        if 'errors' in disease_data:
            logger.error(f"GraphQL errors in disease query: {json.dumps(disease_data['errors'], indent=2)}")
            result += f"\nError fetching disease data: {disease_data['errors'][0]['message']}\n"
        else:
            disease_info = disease_data["data"]["disease"]
            result += "\nOpenTargets Results:\n"
            result += f"Total associated targets: {disease_info['associatedTargets']['count']}\n"
            result += "Top 100 associated targets from OpenTargets:\n"
            opentargets_genes = {}
            for i, row in enumerate(disease_info['associatedTargets']['rows'], 1):
                target = row['target']
                opentargets_genes[target['approvedSymbol']] = row['score']
                result += f"{i}. {target['approvedSymbol']} ({target['approvedName']})\n"
                result += f"   ID: {target['id']}, Association Score: {row['score']:.4f}\n"

        # Compile all genes and create unique gene lists
        all_genes = set(opentargets_genes.keys()) | set(genetics_genes.keys()) | chatgpt_genes

        result += "\nUnique Gene Analysis:\n"
        result += f"Total unique genes found: {len(all_genes)}\n\n"

        genetics_unique = set(genetics_genes.keys()) - set(opentargets_genes.keys()) - chatgpt_genes
        chatgpt_unique = chatgpt_genes - set(opentargets_genes.keys()) - set(genetics_genes.keys())
        opentargets_unique = set(opentargets_genes.keys()) - set(genetics_genes.keys()) - chatgpt_genes

        result += f"Genes unique to OpenTargets Genetics ({len(genetics_unique)}):\n"
        for gene in sorted(genetics_unique):
            result += f"- {gene} (Score: {genetics_genes[gene]:.4f})\n"
        result += "\n"

        result += f"Genes unique to ChatGPT analysis ({len(chatgpt_unique)}):\n"
        for gene in sorted(chatgpt_unique):
            result += f"- {gene}\n"
        result += "\n"

        result += f"Genes unique to OpenTargets ({len(opentargets_unique)}):\n"
        for gene in sorted(opentargets_unique):
            result += f"- {gene} (Score: {opentargets_genes[gene]:.4f})\n"
        result += "\n"

        # Prepare data for CSV download
        gene_data = []
        for gene in sorted(all_genes):
            gene_data.append({
                "Gene": gene,
                "OpenTargets Score": opentargets_genes.get(gene, "N/A"),
                "Genetics API Score": genetics_genes.get(gene, "N/A"),
                "Found in ChatGPT": "Yes" if gene in chatgpt_genes else "No"
            })

        # Add a button to download the CSV
        cl.user_session.set("gene_data", gene_data)
        await cl.Message(content="Gene analysis complete. Click the button below to download the results as a CSV file.").send()
        await cl.Message(content="Download Gene Data", actions=[
            cl.Action(name="download_csv", value="download", label="Download CSV")
        ]).send()

        return result

    except Exception as e:
        logger.exception(f"Error in query_opentargets: {str(e)}")
        return f"An error occurred while querying OpenTargets and extracting genes: {str(e)}"

async def extract_genes_with_chatgpt(abstracts, disease_name):
    try:
        prompt = f"""As an expert in genomics and bioinformatics, analyze the following abstracts about differentially expressed genes in {disease_name}. 
        Identify and list the top differentially expressed genes mentioned across these abstracts. 
        If possible, indicate whether each gene is upregulated or downregulated. 
        Present the results in a clear, numbered list format. 
        If no specific genes are mentioned, provide a summary of the key findings related to gene expression in {disease_name}.

        Abstracts:
        {abstracts}

        Top differentially expressed genes in {disease_name}:
        """

        messages = [
            SystemMessage(content="You are an expert in genomics and bioinformatics, skilled at extracting key information about differentially expressed genes from scientific abstracts."),
            HumanMessage(content=prompt)
        ]

        response = await llm.agenerate([messages])
        return response.generations[0][0].text.strip()
    except Exception as e:
        logger.exception(f"Error in ChatGPT gene extraction: {str(e)}")
        return f"An error occurred while extracting genes with ChatGPT: {str(e)}"
    
tools = [
    Tool(
        name="PubMed Search",
        func=PubMedAPIWrapper().run,
        description="Useful for when you need to answer questions about medical or biological topics. Use this to search for scientific papers on PubMed."
    ),
    Tool(
        name="OpenTargets API",
        func=query_opentargets,
        description="Useful for querying the OpenTargets GraphQL API to find genetic variants, associated genes, and drug targets related to a disease."
    )
]

prompt = PromptTemplate.from_template(
    "You are an AI assistant. You have access to the following tools: {tools}\n\n"
    "Tool names: {tool_names}\n\n"
    "To use a tool, please use the following format:\n"
    "Thought: Do I need to use a tool? Yes\n"
    "Action: the action to take, should be one of {tool_names}\n"
    "Action Input: the input to the action\n"
    "Observation: the result of the action\n\n"
    "When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:\n"
    "Thought: Do I need to use a tool? No\n"
    "AI: [provide a detailed and elaborate response here]\n\n"
    "Begin!\n\n"
    "Human: {input}\n"
    "Thought: {agent_scratchpad}"
)

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

experts = {
    "Biologist": "You are an expert biologist. Provide detailed and comprehensive scientific explanations on biological topics to a postdoctoral scientific and technical audience. Use the PubMed Search tool and other scholarly sources like Google Scholar, JSTOR, and university publications to find and cite recent research.",
    "Informatician": "You are an expert in bioinformatics. Explain biological concepts from a data analysis perspective in depth to a postdoctoral scientific and technical audience.",
    "Computer Scientist": "You are a computer scientist specializing in computational biology. Discuss biological topics in terms of algorithms and computational methods with detailed explanations to a postdoctoral scientific and technical audience.",
    "General Expert": "You are a general science expert. Provide comprehensive explanations on scientific topics, relating them to other fields when relevant, to a postdoctoral scientific and technical audience."
}

expert_introductions = {
    "Biologist": "I am a Biologist, an expert in biological sciences. I can provide detailed scientific explanations on biological topics.",
    "Informatician": "I am an Informatician, specializing in bioinformatics. I can explain biological concepts from a data analysis perspective.",
    "Computer Scientist": "I am a Computer Scientist specializing in computational biology. I can discuss biological topics in terms of algorithms and computational methods.",
    "General Expert": "I am a General Expert in science. I can provide comprehensive explanations on scientific topics and relate them to other fields when relevant."
}

def chat_with_gpt(prompt, expert):
    try:
        if expert in ["Biologist", "General Expert"]:
            try:
                response = agent_executor.run(input=prompt)
            except ValueError as ve:
                logger.error(f"AgentExecutor error: {str(ve)}")
                response = llm.predict(prompt)
        else:
            response = llm.chat.create(
                messages=[
                    {"role": "system", "content": experts[expert]},
                    {"role": "user", "content": prompt}
                ]
            )
            response = response.choices[0].message.content.strip()
        return response
    except Exception as e:
        logger.error(f"Error in chat_with_gpt: {str(e)}")
        return f"An error occurred while processing your request: {str(e)}"

async def handle_file_upload(file: cl.File):
    try:
        pdf_path = file.path
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()

        full_text = "\n".join([page.page_content for page in pages])

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_text(full_text)

        embeddings = OpenAIEmbeddings()
        db = FAISS.from_texts(texts, embeddings)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
        )

        cl.user_session.set("qa_chain", qa_chain)
        await cl.Message(content=f"PDF '{file.name}' uploaded and processed. You can now ask questions about the content.").send()
        
        summary_prompt = f"Please provide a brief summary of the following text, which is the content of the uploaded PDF titled '{file.name}':\n\n{full_text[:2000]}"
        summary = llm.predict(summary_prompt)
        await cl.Message(content=f"Summary of '{file.name}':\n\n{summary}").send()
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        await cl.Message(content=f"An error occurred while processing the PDF: {str(e)}").send()

async def test_opentargets_api():
    try:
        # Test Platform API
        test_query = """
        {
          meta {
            apiVersion {
              x
              y
              z
            }
          }
        }
        """
        response = execute_query(test_query)
        if 'errors' in response:
            logger.error(f"GraphQL errors: {json.dumps(response['errors'], indent=2)}")
            await cl.Message(content=f"Error querying OpenTargets Platform API: {response['errors'][0]['message']}").send()
            return
        
        version = response["data"]["meta"]["apiVersion"]
        logger.info(f"Successfully connected to OpenTargets Platform API. Version: {version['x']}.{version['y']}.{version['z']}")
        await cl.Message(content=f"Successfully connected to OpenTargets Platform API. Version: {version['x']}.{version['y']}.{version['z']}").send()

        # Test Genetics API
        test_genetics_query = """
        query {
          meta {
            apiVersion {
              major
              minor
              patch
            }
          }
        }
        """
        genetics_response = execute_genetics_query(test_genetics_query)
        if 'errors' in genetics_response:
            logger.error(f"GraphQL errors in Genetics API: {json.dumps(genetics_response['errors'], indent=2)}")
            await cl.Message(content=f"Error querying OpenTargets Genetics API: {genetics_response['errors'][0]['message']}").send()
            return
        
        genetics_version = genetics_response["data"]["meta"]["apiVersion"]
        logger.info(f"Successfully connected to OpenTargets Genetics API. Version: {genetics_version['major']}.{genetics_version['minor']}.{genetics_version['patch']}")
        await cl.Message(content=f"Successfully connected to OpenTargets Genetics API. Version: {genetics_version['major']}.{genetics_version['minor']}.{genetics_version['patch']}").send()

        # Test Genetics API with a sample study
        test_study_query = """
        query StudyVariants($studyId: String!) {
          manhattan(studyId: $studyId) {
            associations {
              variant {
                id
                rsId
              }
              pval
              beta
              oddsRatio
              bestGenes {
                score
                gene {
                  id
                  symbol
                }
              }
            }
          }
        }
        """
        test_study_id = "GCST90002369"  # Using a placeholder study ID
        study_response = execute_genetics_query(test_study_query, {"studyId": test_study_id})
        if 'errors' in study_response:
            logger.error(f"GraphQL errors in study query: {json.dumps(study_response['errors'], indent=2)}")
            await cl.Message(content=f"Error querying study data: {study_response['errors'][0]['message']}").send()
            return
        
        associations = study_response['data']['manhattan']['associations']
        if associations:
            association = associations[0]  # Take the first association
            logger.info(f"Successfully queried study data. Found variant: {association['variant']['id']}")
            await cl.Message(content=f"Successfully queried study data. Found variant: {association['variant']['id']}").send()
            
            # Log the number of associations found
            logger.info(f"Total associations found: {len(associations)}")
            await cl.Message(content=f"Total associations found: {len(associations)}").send()
        else:
            logger.warning(f"No associations found for study ID: {test_study_id}")
            await cl.Message(content=f"No associations found for study ID: {test_study_id}").send()

    except Exception as e:
        logger.exception("Error testing OpenTargets APIs")
        await cl.Message(content=f"Error testing OpenTargets APIs: {str(e)}").send()

@cl.on_chat_start
async def start():
    cl.user_session.set("expert", "General Expert")
    actions = [
        cl.Action(name="change_expert", value=expert, label=f"Switch to {expert}")
        for expert in experts.keys()
    ]
    actions.append(cl.Action(name="get_gene_list", value="get_gene_list", label="Get GET List"))
    await cl.Message(content="Select an expert, upload a PDF for analysis, or get a gene list:", actions=actions).send()
    
    try:
        await test_opentargets_api()
    except Exception as e:
        error_message = f"Error initializing OpenTargets API: {str(e)}\n\nPlease try again later or contact support if the issue persists."
        await cl.Message(content=error_message).send()
        logger.exception("Error during OpenTargets API initialization")

@cl.action_callback("get_gene_list")
async def on_get_gene_list(action):
    await cl.Message(content="Please enter the name of the disease you want to query:").send()
    cl.user_session.set("awaiting_disease_name", True)

@cl.on_message
async def main(message: cl.Message):
    try:
        expert = cl.user_session.get("expert")

        if cl.user_session.get("awaiting_disease_name"):
            disease_name = message.content
            cl.user_session.set("awaiting_disease_name", False)
            await cl.Message(content=f"Querying OpenTargets, OpenTargets Genetics, and PubMed abstracts for disease: {disease_name}").send()
            response = await query_opentargets(disease_name)
            await cl.Message(content=response).send()
            return

        # Check if a file was uploaded
        if message.content == "" and message.elements:
            for element in message.elements:
                if isinstance(element, cl.File) and element.name.lower().endswith('.pdf'):
                    await handle_file_upload(element)
                    return
            await cl.Message(content="Please upload a PDF file for analysis.").send()
            return

        qa_chain = cl.user_session.get("qa_chain")
        if qa_chain:
            response = qa_chain({"query": message.content})
            await cl.Message(content=f"Answer: {response['result']}").send()
        else:
            if message.content.lower() in ["who are you", "what is your expertise"]:
                await cl.Message(content=expert_introductions[expert]).send()
            else:
                response = chat_with_gpt(message.content, expert)
                formatted_response = f"## Response from {expert}\n\n{response}"
                await cl.Message(content=formatted_response).send()
    except Exception as e:
        error_message = f"An error occurred: {str(e)}\nType: {type(e).__name__}\nDetails: {e.args}"
        logger.error(error_message)
        await cl.Message(content=error_message).send()

@cl.action_callback("download_csv")
async def on_action(action):
    gene_data = cl.user_session.get("gene_data")
    if gene_data:
        csv_file = io.StringIO()
        fieldnames = ["Gene", "OpenTargets Score", "Genetics API Score", "Found in ChatGPT"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(gene_data)
        
        # Create a new message to attach the file to
        msg = cl.Message(content="Here's your gene analysis CSV file:")
        await msg.send()

        # Now send the file, associated with the message we just created
        await cl.File(name="gene_analysis.csv", content=csv_file.getvalue().encode(), mime="text/csv").send(msg.id)
    else:
        await cl.Message(content="No gene data available for download.").send()