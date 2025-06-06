# main.py - Multi-Agent System with Streaming (Fixed Dependencies)

import os
import json
import asyncio
import threading
from typing import Dict, Any, Generator, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import easyocr
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import BaseMessage, HumanMessage, AIMessage

# Try to import vector store components, but don't fail if they're problematic
try:
    from langchain_chroma import Chroma
    from langchain_community.embeddings import OllamaEmbeddings
    VECTOR_STORE_AVAILABLE = True
except ImportError:
    print("⚠️ Vector store components not available, running without vector DB")
    VECTOR_STORE_AVAILABLE = False

# =========================
# STREAMING CALLBACK HANDLER
# =========================
class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.tokens = []
        self.is_streaming = True
        
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if self.is_streaming:
            self.tokens.append(token)
            print(f"[{self.agent_name}] {token}", end="", flush=True)
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        print(f"\n🤖 [{self.agent_name}] Starting generation...")
        
    def on_llm_end(self, response, **kwargs: Any) -> None:
        print(f"\n✅ [{self.agent_name}] Complete\n")

# =========================
# SIMPLE VECTOR STORE ALTERNATIVE
# =========================
class SimpleDocumentStore:
    def __init__(self):
        self.documents = []
        self.embeddings_available = VECTOR_STORE_AVAILABLE
        
    def add_documents(self, docs: List[str]):
        self.documents.extend(docs)
        
    def similarity_search(self, query: str, k: int = 3) -> List[str]:
        # Simple keyword-based search as fallback
        query_words = set(query.lower().split())
        scored_docs = []
        
        for doc in self.documents:
            doc_words = set(doc.lower().split())
            similarity = len(query_words.intersection(doc_words)) / len(query_words.union(doc_words))
            scored_docs.append((similarity, doc))
        
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        return [doc for _, doc in scored_docs[:k]]

# =========================
# INDIVIDUAL AGENTS
# =========================
class BaseAgent:
    def __init__(self, name: str, prompt_template: str):
        self.name = name
        self.prompt_template = PromptTemplate.from_template(prompt_template)
        self.streaming_handler = StreamingCallbackHandler(name)
        
    def process(self, query: str) -> str:
        try:
            print(f"\n🎯 [{self.name}] Processing query...")
            formatted_prompt = self.prompt_template.format(input=query)
            
            # Create LLM with streaming callback
            streaming_llm = ChatOllama(
                model="llama3", 
                streaming=True,
                callbacks=[self.streaming_handler],
                temperature=0.7
            )
            
            result = streaming_llm.invoke([HumanMessage(content=formatted_prompt)])
            response = result.content if hasattr(result, 'content') else str(result)
            
            return response
            
        except Exception as e:
            error_msg = f"Error in {self.name}: {str(e)}"
            print(f"\n❌ [{self.name}] {error_msg}")
            return error_msg

class ResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            "Research", 
            """You are a research specialist AI. Conduct thorough research and analysis on the given topic.

Topic: {input}

Please provide a comprehensive research response that includes:
1. **Overview**: Brief introduction to the topic
2. **Key Concepts**: Important definitions and terminology
3. **Current Trends**: Latest developments and innovations
4. **Facts & Statistics**: Relevant data and numbers
5. **Applications**: Real-world use cases and examples
6. **Future Outlook**: Predictions and emerging trends

Make your response detailed, informative, and well-structured.

Research Analysis:"""
        )

class GeneralAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            "General", 
            """You are a helpful, friendly conversational AI assistant.

User Query: {input}

Provide a natural, conversational response that:
- Addresses the user's question directly
- Is friendly and approachable in tone
- Includes helpful context when appropriate
- Offers additional insights or related information
- Maintains a warm, supportive conversation style

Response:"""
        )

class ExplainerAgent(BaseAgent):
    def __init__(self, document_store=None):
        super().__init__(
            "Explainer", 
            """You are an expert educational AI that provides detailed, clear explanations.

Topic to Explain: {input}

Provide a comprehensive educational explanation that includes:
1. **Definition**: Clear, simple definition of the concept
2. **Core Principles**: Fundamental ideas and principles
3. **Step-by-Step Breakdown**: How it works or functions
4. **Examples**: Concrete, relatable examples
5. **Common Misconceptions**: What people often get wrong
6. **Practical Applications**: How it's used in real life
7. **Key Takeaways**: Most important points to remember

Make your explanation accessible to learners while being thorough and accurate.

Detailed Explanation:"""
        )
        self.document_store = document_store
        
    def process(self, query: str) -> str:
        try:
            # Try to enhance with document store if available
            context = ""
            if self.document_store and self.document_store.documents:
                relevant_docs = self.document_store.similarity_search(query)
                if relevant_docs:
                    context = "\n\nAdditional Context from Knowledge Base:\n" + "\n".join(relevant_docs[:2])
            
            enhanced_query = query + context
            return super().process(enhanced_query)
            
        except Exception as e:
            print(f"⚠️ Explainer context enhancement failed: {e}")
            return super().process(query)

class SolutionAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            "Solution", 
            """You are a coding expert AI that provides practical programming solutions.

Problem/Task: {input}

Provide a complete code solution that includes:
1. **Problem Analysis**: Understanding of what needs to be solved
2. **Approach**: Strategy and algorithm explanation
3. **Complete Code**: Well-commented, working Python code
4. **Usage Example**: How to use the code
5. **Error Handling**: Robust error management
6. **Optimization Notes**: Performance considerations

Format your code properly with syntax highlighting and include all necessary imports.

Code Solution:

```python"""
        )

# =========================
# CONTROLLER AGENT
# =========================
class MultiAgentController:
    def __init__(self):
        print("🚀 Initializing Multi-Agent System...")
        
        # Initialize document store
        self.document_store = SimpleDocumentStore()
        
        # Initialize vector store if available
        self.vectorstore = self._initialize_vectorstore()
        
        # Initialize agents
        self.agents = {
            "Research": ResearchAgent(),
            "General": GeneralAgent(),
            "Explainer": ExplainerAgent(self.document_store),
            "Solution": SolutionAgent()
        }
        
        print("✅ Multi-Agent System initialized successfully!")
        
    def _initialize_vectorstore(self) -> Optional[Any]:
        if not VECTOR_STORE_AVAILABLE:
            print("⚠️ Vector store not available, using simple document store")
            return None
            
        try:
            embedding = OllamaEmbeddings(model="llama3")
            vectorstore = Chroma(
                persist_directory="./vectordb", 
                embedding_function=embedding
            )
            print("✅ ChromaDB vector store initialized")
            return vectorstore
        except Exception as e:
            print(f"⚠️ Vector store initialization failed: {e}")
            return None
    
    def load_documents_to_store(self, documents: List[str]):
        """Load documents into the document store for enhanced explanations"""
        self.document_store.add_documents(documents)
        print(f"📚 Loaded {len(documents)} documents into knowledge base")
    
    def process_with_streaming(self, query: str) -> Dict[str, str]:
        """Process query with all agents using concurrent execution and streaming"""
        print(f"\n{'='*80}")
        print(f"🎯 MULTI-AGENT CONTROLLER ACTIVATED")
        print(f"📝 Processing Query: {query[:100]}{'...' if len(query) > 100 else ''}")
        print(f"{'='*80}")
        
        results = {}
        
        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="Agent") as executor:
            print(f"⚡ Launching {len(self.agents)} agents concurrently...")
            
            # Submit all agent tasks
            future_to_agent = {
                executor.submit(agent.process, query): agent_name 
                for agent_name, agent in self.agents.items()
            }
            
            # Collect results as they complete
            completed_count = 0
            for future in as_completed(future_to_agent):
                agent_name = future_to_agent[future]
                completed_count += 1
                
                try:
                    result = future.result(timeout=120)  # 2-minute timeout per agent
                    results[agent_name] = result
                    print(f"🎉 Agent {completed_count}/{len(self.agents)} completed: {agent_name}")
                except Exception as exc:
                    error_msg = f"Agent {agent_name} generated an exception: {exc}"
                    results[agent_name] = error_msg
                    print(f"❌ Agent {agent_name} failed: {exc}")
        
        print(f"\n🏁 All agents completed processing!")
        return results
    
    def display_structured_output(self, results: Dict[str, str]) -> str:
        """Display results in structured format"""
        print(f"\n{'='*80}")
        print("🏆 STRUCTURED MULTI-AGENT OUTPUT")
        print(f"{'='*80}")
        
        # Define display order for better readability
        display_order = ["General", "Explainer", "Research", "Solution"]
        
        for agent_name in display_order:
            if agent_name in results:
                response = results[agent_name]
                print(f"\n🔹 {agent_name.upper()} AGENT:")
                print(f"{'='*50}")
                print(f"{response}")
                print(f"{'='*50}")
        
        # Handle any agents not in the display order
        for agent_name, response in results.items():
            if agent_name not in display_order:
                print(f"\n🔹 {agent_name.upper()} AGENT:")
                print(f"{'='*50}")
                print(f"{response}")
                print(f"{'='*50}")
        
        # Create JSON output
        json_output = {
            "timestamp": str(asyncio.get_event_loop().time()) if hasattr(asyncio, 'get_event_loop') else "N/A",
            "agents": results,
            "summary": {
                "total_agents": len(results),
                "successful_agents": len([r for r in results.values() if not r.startswith("Error")]),
                "failed_agents": len([r for r in results.values() if r.startswith("Error")])
            }
        }
        
        return json.dumps(json_output, indent=2, ensure_ascii=False)

# =========================
# INPUT PARSERS
# =========================
class InputParser:
    @staticmethod
    def parse_pdf(file_path: str) -> str:
        try:
            if not os.path.exists(file_path):
                return f"Error: PDF file not found at {file_path}"
            
            print(f"📄 Parsing PDF: {file_path}")
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            content = "\n".join([doc.page_content for doc in docs])
            print(f"✅ PDF parsed successfully: {len(docs)} pages, {len(content)} characters")
            return content
        except Exception as e:
            return f"Error parsing PDF: {str(e)}"

    @staticmethod
    def parse_image(file_path: str) -> str:
        try:
            if not os.path.exists(file_path):
                return f"Error: Image file not found at {file_path}"
            
            print(f"🖼️ Performing OCR on image: {file_path}")
            reader = easyocr.Reader(['en'], gpu=False)  # Disable GPU for compatibility
            result = reader.readtext(file_path, detail=0)
            content = "\n".join(result)
            print(f"✅ OCR completed: {len(result)} text blocks, {len(content)} characters")
            return content
        except Exception as e:
            return f"Error parsing image: {str(e)}"

# =========================
# MAIN APPLICATION
# =========================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Multi-Agent System with Streaming Output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --text "Explain machine learning algorithms"
  python main.py --pdf "document.pdf" --output "results.json"
  python main.py --img "screenshot.png"
        """
    )
    parser.add_argument("--text", type=str, help="Direct text input")
    parser.add_argument("--pdf", type=str, help="PDF file path")
    parser.add_argument("--img", type=str, help="Image file path")
    parser.add_argument("--output", type=str, help="Output JSON file path (optional)")
    args = parser.parse_args()
    
    # Initialize controller
    try:
        controller = MultiAgentController()
    except Exception as e:
        print(f"❌ Failed to initialize Multi-Agent Controller: {e}")
        return
    
    # Process input
    user_input = None
    if args.text:
        user_input = args.text
        print(f"📝 Processing direct text input")
    elif args.pdf:
        user_input = InputParser.parse_pdf(args.pdf)
        print(f"📄 Processing PDF: {args.pdf}")
    elif args.img:
        user_input = InputParser.parse_image(args.img)
        print(f"🖼️ Processing image: {args.img}")
    else:
        print("❌ Error: Please provide input using --text, --pdf, or --img")
        print("\nExamples:")
        print("  python main.py --text 'Explain neural networks'")
        print("  python main.py --pdf 'document.pdf'")
        print("  python main.py --img 'image.png' --output 'results.json'")
        return
    
    # Check if input parsing was successful
    if user_input and user_input.startswith("Error:"):
        print(f"❌ {user_input}")
        return
    
    if not user_input or len(user_input.strip()) == 0:
        print("❌ Error: No valid input content found")
        return
    
    # Process with multi-agent system
    try:
        results = controller.process_with_streaming(user_input)
        
        # Display structured output
        json_output = controller.display_structured_output(results)
        
        # Save output if requested
        if args.output:
            try:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(json_output)
                print(f"\n💾 Complete output saved to: {args.output}")
            except Exception as e:
                print(f"❌ Error saving output: {e}")
        
        print(f"\n🎉 Multi-Agent processing completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"❌ Error during processing: {e}")

if __name__ == "__main__":
    main()