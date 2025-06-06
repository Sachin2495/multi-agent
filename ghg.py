# main.py - Multi-Agent System with Streaming

import os
import json
import asyncio
import threading
from typing import Dict, Any, Generator, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import easyocr
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LangChainException

# =========================
# STREAMING CALLBACK HANDLER
# =========================
class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.tokens = []
        
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.tokens.append(token)
        print(f"[{self.agent_name}] {token}", end="", flush=True)

# =========================
# INDIVIDUAL AGENTS
# =========================
class BaseAgent:
    def __init__(self, name: str, llm, prompt_template: str):
        self.name = name
        self.llm = llm
        self.prompt_template = PromptTemplate.from_template(prompt_template)
        self.streaming_handler = StreamingCallbackHandler(name)
        
    def process(self, query: str) -> str:
        try:
            print(f"\n🤖 [{self.name}] Processing...")
            formatted_prompt = self.prompt_template.format(input=query)
            
            # Create LLM with streaming callback
            streaming_llm = ChatOllama(
                model="llama3", 
                streaming=True,
                callbacks=[self.streaming_handler]
            )
            
            result = streaming_llm.invoke(formatted_prompt)
            response = result.content if hasattr(result, 'content') else str(result)
            print(f"\n✅ [{self.name}] Complete\n")
            return response
            
        except Exception as e:
            error_msg = f"Error in {self.name}: {str(e)}"
            print(f"\n❌ [{self.name}] {error_msg}\n")
            return error_msg

class ResearchAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(
            "Research", 
            llm, 
            """You are a research specialist. Conduct thorough research on the topic and provide comprehensive insights.
            Research Topic: {input}
            
            Provide detailed research findings with:
            - Key concepts and definitions
            - Current trends and developments  
            - Important facts and statistics
            - Relevant examples and case studies
            
            Research Response:"""
        )

class GeneralAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(
            "General", 
            llm, 
            """You are a helpful conversational assistant. Respond naturally and conversationally.
            User Input: {input}
            
            Provide a friendly, informative response that addresses the user's query in a conversational manner.
            
            Response:"""
        )

class ExplainerAgent(BaseAgent):
    def __init__(self, llm, vectorstore=None):
        super().__init__(
            "Explainer", 
            llm, 
            """You are an expert explainer. Provide detailed, educational explanations.
            Topic to Explain: {input}
            
            Provide a comprehensive explanation including:
            - Clear definition and overview
            - Step-by-step breakdown
            - Key principles and concepts
            - Practical examples
            - Common misconceptions (if any)
            
            Detailed Explanation:"""
        )
        self.vectorstore = vectorstore
        
    def process(self, query: str) -> str:
        try:
            print(f"\n🤖 [{self.name}] Processing with Knowledge Base...")
            
            if self.vectorstore:
                # Try to use vector store for enhanced explanations
                try:
                    retriever = self.vectorstore.as_retriever()
                    chain = RetrievalQA.from_chain_type(
                        llm=self.llm, 
                        retriever=retriever,
                        callbacks=[self.streaming_handler]
                    )
                    result = chain.invoke({"query": query})
                    response = result.get("result", str(result))
                    print(f"\n✅ [{self.name}] Complete (with KB)\n")
                    return response
                except Exception as e:
                    print(f"⚠️ Vector store error, falling back to basic explanation: {e}")
            
            # Fallback to basic explanation
            return super().process(query)
            
        except Exception as e:
            error_msg = f"Error in {self.name}: {str(e)}"
            print(f"\n❌ [{self.name}] {error_msg}\n")
            return error_msg

class SolutionAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(
            "Solution", 
            llm, 
            """You are a coding expert. Provide practical code solutions.
            Problem: {input}
            
            Provide:
            - Clean, well-commented Python code
            - Explanation of the approach
            - Usage examples
            - Error handling where appropriate
            
            Code Solution:
            ```python"""
        )

# =========================
# CONTROLLER AGENT
# =========================
class MultiAgentController:
    def __init__(self):
        # Initialize base LLM
        self.base_llm = ChatOllama(model="llama3", streaming=False)
        
        # Initialize vector store
        self.vectorstore = self._initialize_vectorstore()
        
        # Initialize agents
        self.agents = {
            "Research": ResearchAgent(self.base_llm),
            "General": GeneralAgent(self.base_llm),
            "Explainer": ExplainerAgent(self.base_llm, self.vectorstore),
            "Solution": SolutionAgent(self.base_llm)
        }
        
    def _initialize_vectorstore(self) -> Optional[Chroma]:
        try:
            embedding = OllamaEmbeddings(model="llama3")
            vectorstore = Chroma(
                persist_directory="./vectordb", 
                embedding_function=embedding
            )
            print("✅ Vector store initialized successfully")
            return vectorstore
        except Exception as e:
            print(f"⚠️ Vector store initialization failed: {e}")
            return None
    
    def process_with_streaming(self, query: str) -> Dict[str, str]:
        """Process query with all agents using concurrent execution and streaming"""
        print(f"\n🎯 MULTI-AGENT CONTROLLER ACTIVATED")
        print(f"📝 Query: {query}")
        print("="*60)
        
        results = {}
        
        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all agent tasks
            future_to_agent = {
                executor.submit(agent.process, query): agent_name 
                for agent_name, agent in self.agents.items()
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_agent):
                agent_name = future_to_agent[future]
                try:
                    result = future.result()
                    results[agent_name] = result
                except Exception as exc:
                    results[agent_name] = f"Agent {agent_name} generated an exception: {exc}"
        
        return results
    
    def display_structured_output(self, results: Dict[str, str]):
        """Display results in structured format"""
        print("\n" + "="*60)
        print("🏆 STRUCTURED MULTI-AGENT OUTPUT")
        print("="*60)
        
        for agent_name, response in results.items():
            print(f"\n🔹 {agent_name.upper()}:")
            print("-" * 40)
            print(f"{response}")
            print("-" * 40)
        
        # Also return as JSON for programmatic use
        return json.dumps(results, indent=2)

# =========================
# INPUT PARSERS
# =========================
class InputParser:
    @staticmethod
    def parse_pdf(file_path: str) -> str:
        try:
            if not os.path.exists(file_path):
                return f"Error: PDF file not found at {file_path}"
            
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            content = "\n".join([doc.page_content for doc in docs])
            print(f"✅ PDF parsed successfully: {len(docs)} pages")
            return content
        except Exception as e:
            return f"Error parsing PDF: {str(e)}"

    @staticmethod
    def parse_image(file_path: str) -> str:
        try:
            if not os.path.exists(file_path):
                return f"Error: Image file not found at {file_path}"
            
            reader = easyocr.Reader(['en'])
            result = reader.readtext(file_path, detail=0)
            content = "\n".join(result)
            print(f"✅ Image OCR completed: {len(result)} text blocks extracted")
            return content
        except Exception as e:
            return f"Error parsing image: {str(e)}"

# =========================
# MAIN APPLICATION
# =========================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Agent System with Streaming")
    parser.add_argument("--text", type=str, help="Direct text input")
    parser.add_argument("--pdf", type=str, help="PDF file path")
    parser.add_argument("--img", type=str, help="Image file path")
    parser.add_argument("--output", type=str, help="Output file path (optional)")
    args = parser.parse_args()
    
    # Initialize controller
    controller = MultiAgentController()
    
    # Process input
    if args.text:
        user_input = args.text
        print(f"📝 Processing text input...")
    elif args.pdf:
        user_input = InputParser.parse_pdf(args.pdf)
        print(f"📄 Processing PDF: {args.pdf}")
    elif args.img:
        user_input = InputParser.parse_image(args.img)
        print(f"🖼️ Processing image: {args.img}")
    else:
        print("❌ Error: Please provide input using --text, --pdf, or --img")
        print("Example: python main.py --text 'Explain machine learning'")
        return
    
    # Check if input parsing was successful
    if user_input.startswith("Error:"):
        print(f"❌ {user_input}")
        return
    
    # Process with multi-agent system
    results = controller.process_with_streaming(user_input)
    
    # Display structured output
    json_output = controller.display_structured_output(results)
    
    # Save output if requested
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(json_output)
            print(f"\n💾 Output saved to: {args.output}")
        except Exception as e:
            print(f"❌ Error saving output: {e}")

if __name__ == "__main__":
    main()