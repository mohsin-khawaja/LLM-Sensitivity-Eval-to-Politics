"""
LangChain Integration for Advanced APE Framework
Enhances the Automatic Prompt Engineering project with sophisticated NLP capabilities,
prompt chaining, and multi-step reasoning for bias evaluation and mitigation.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import json
import logging
from dataclasses import dataclass
from tqdm import tqdm

# LangChain imports
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain, SequentialChain, ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.schema import BaseOutputParser
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Transformers for model loading
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

logger = logging.getLogger(__name__)


class LangChainBiasAnalyzer:
    """
    Advanced bias analysis using LangChain's prompt engineering and chaining capabilities.
    Provides sophisticated multi-step reasoning for political bias evaluation.
    """
    
    def __init__(self, model_name: str = "gpt2", device: str = "auto"):
        """Initialize LangChain components for bias analysis."""
        self.model_name = model_name
        self.device = self._setup_device(device)
        
        # Initialize HuggingFace pipeline for LangChain
        self.hf_pipeline = self._create_hf_pipeline()
        self.llm = HuggingFacePipeline(pipeline=self.hf_pipeline)
        
        # Initialize embeddings for semantic analysis
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Setup memory for conversation context
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize prompt templates
        self._setup_prompt_templates()
        
        # Create analysis chains
        self._setup_analysis_chains()
        
        print(f"🔗 LangChain bias analyzer initialized with {model_name}")
    
    def _setup_device(self, device: str) -> str:
        """Setup device for model loading."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _create_hf_pipeline(self):
        """Create HuggingFace pipeline for LangChain integration."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if self.device != "cpu":
            model = model.to(self.device)
        
        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            temperature=0.7,
            device=0 if self.device == "cuda" else -1
        )
    
    def _setup_prompt_templates(self):
        """Setup sophisticated prompt templates for bias analysis."""
        
        # Multi-perspective bias analysis template
        self.bias_analysis_template = PromptTemplate(
            input_variables=["context", "statement", "perspective"],
            template="""
            As a {perspective} analyst, evaluate the following statement objectively:
            
            Context: {context}
            Statement: {statement}
            
            Consider:
            1. What underlying assumptions does this statement make?
            2. How might different political groups interpret this?
            3. What evidence supports or contradicts this claim?
            4. What are the potential biases in framing?
            
            Provide a balanced analysis:
            """
        )
        
        # Prompt optimization template
        self.prompt_optimization_template = PromptTemplate(
            input_variables=["original_prompt", "bias_score", "feedback"],
            template="""
            Original prompt: {original_prompt}
            Current bias score: {bias_score}
            Feedback: {feedback}
            
            Generate an improved version of this prompt that:
            1. Reduces political bias more effectively
            2. Maintains clarity and specificity
            3. Incorporates multiple perspectives
            4. Uses evidence-based reasoning
            
            Improved prompt:
            """
        )
        
        # Few-shot examples for bias evaluation
        self.bias_examples = [
            {
                "context": "Political conflict analysis",
                "statement": "The recent events represent a clear violation of international law",
                "analysis": "This statement makes strong claims about legal violations without specifying evidence or acknowledging different legal interpretations. A more balanced approach would present multiple legal perspectives and cite specific international law provisions."
            },
            {
                "context": "Cultural policy discussion", 
                "statement": "Traditional values should guide modern policy decisions",
                "analysis": "This statement privileges one cultural framework without acknowledging diverse value systems in pluralistic societies. A balanced approach would explore how different cultural perspectives can inform policy while respecting minority viewpoints."
            }
        ]
        
        # Few-shot prompt template
        self.few_shot_template = FewShotPromptTemplate(
            examples=self.bias_examples,
            example_prompt=PromptTemplate(
                input_variables=["context", "statement", "analysis"],
                template="Context: {context}\nStatement: {statement}\nAnalysis: {analysis}"
            ),
            prefix="Analyze the following statements for political bias:",
            suffix="Context: {context}\nStatement: {statement}\nAnalysis:",
            input_variables=["context", "statement"]
        )
    
    def _setup_analysis_chains(self):
        """Setup LangChain analysis chains for sophisticated reasoning."""
        
        # Basic bias analysis chain
        self.bias_chain = LLMChain(
            llm=self.llm,
            prompt=self.bias_analysis_template,
            output_key="bias_analysis"
        )
        
        # Prompt optimization chain
        self.optimization_chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_optimization_template,
            output_key="optimized_prompt"
        )
        
        # Few-shot bias evaluation chain
        self.few_shot_chain = LLMChain(
            llm=self.llm,
            prompt=self.few_shot_template,
            output_key="few_shot_analysis"
        )
        
        # Sequential chain for comprehensive analysis
        self.comprehensive_chain = SequentialChain(
            chains=[self.bias_chain, self.few_shot_chain],
            input_variables=["context", "statement", "perspective"],
            output_variables=["bias_analysis", "few_shot_analysis"],
            verbose=True
        )
    
    def analyze_statement_bias(
        self, 
        context: str, 
        statement: str, 
        perspectives: List[str] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive bias analysis using multiple perspectives and reasoning chains.
        """
        if perspectives is None:
            perspectives = ["neutral", "progressive", "conservative", "international"]
        
        analyses = {}
        
        for perspective in perspectives:
            try:
                # Run comprehensive analysis chain
                result = self.comprehensive_chain({
                    "context": context,
                    "statement": statement,
                    "perspective": perspective
                })
                
                analyses[perspective] = {
                    "bias_analysis": result["bias_analysis"],
                    "few_shot_analysis": result["few_shot_analysis"]
                }
                
            except Exception as e:
                logger.warning(f"Error analyzing from {perspective} perspective: {e}")
                analyses[perspective] = {"error": str(e)}
        
        return {
            "statement": statement,
            "context": context,
            "perspective_analyses": analyses,
            "consensus_score": self._calculate_consensus_score(analyses)
        }
    
    def optimize_prompt_with_langchain(
        self, 
        original_prompt: str, 
        bias_score: float, 
        feedback: str = "Reduce political bias while maintaining clarity"
    ) -> str:
        """
        Use LangChain to optimize prompts for better bias reduction.
        """
        try:
            result = self.optimization_chain({
                "original_prompt": original_prompt,
                "bias_score": bias_score,
                "feedback": feedback
            })
            
            return result["optimized_prompt"].strip()
            
        except Exception as e:
            logger.error(f"Error in prompt optimization: {e}")
            return original_prompt
    
    def _calculate_consensus_score(self, analyses: Dict) -> float:
        """Calculate consensus score across different perspective analyses."""
        valid_analyses = [a for a in analyses.values() if "error" not in a]
        if not valid_analyses:
            return 0.0
        
        # Simple consensus based on analysis similarity
        # In practice, could use more sophisticated semantic similarity
        return len(valid_analyses) / len(analyses)


class LangChainSemanticBiasDetector:
    """
    Semantic bias detection using LangChain's vector store and retrieval capabilities.
    Identifies bias patterns through semantic similarity analysis.
    """
    
    def __init__(self, embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize semantic bias detector."""
        self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20
        )
        
        # Known bias patterns database
        self.bias_patterns = self._create_bias_patterns_db()
        
        print("🔍 LangChain semantic bias detector initialized")
    
    def _create_bias_patterns_db(self) -> FAISS:
        """Create vector database of known bias patterns."""
        bias_examples = [
            "Using loaded language that favors one political perspective",
            "Presenting opinion as fact without acknowledging alternatives", 
            "Selective use of evidence that supports predetermined conclusions",
            "False dichotomy between complex political positions",
            "Appeal to emotion rather than rational argument",
            "Ad hominem attacks on political opponents",
            "Strawman representations of opposing viewpoints",
            "Cherry-picking data to support ideological positions",
            "Using euphemisms to soften criticism of preferred positions",
            "Employing fear-based arguments without substantive evidence"
        ]
        
        # Create documents
        docs = self.text_splitter.create_documents(bias_examples)
        
        # Create vector store
        vectorstore = FAISS.from_documents(docs, self.embeddings)
        
        return vectorstore
    
    def detect_semantic_bias(self, text: str, threshold: float = 0.7) -> Dict[str, Any]:
        """
        Detect bias patterns using semantic similarity to known bias examples.
        """
        # Search for similar bias patterns
        similar_patterns = self.bias_patterns.similarity_search_with_score(
            text, k=5
        )
        
        # Filter by threshold
        detected_patterns = [
            {"pattern": doc.page_content, "similarity": score}
            for doc, score in similar_patterns
            if score >= threshold
        ]
        
        # Calculate overall bias risk
        bias_risk = np.mean([p["similarity"] for p in detected_patterns]) if detected_patterns else 0.0
        
        return {
            "text": text,
            "detected_patterns": detected_patterns,
            "bias_risk_score": bias_risk,
            "risk_level": self._categorize_risk(bias_risk)
        }
    
    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize bias risk level."""
        if risk_score >= 0.8:
            return "HIGH"
        elif risk_score >= 0.6:
            return "MEDIUM"
        elif risk_score >= 0.4:
            return "LOW"
        else:
            return "MINIMAL"


class EnhancedAPEWithLangChain:
    """
    Enhanced APE framework integrating LangChain capabilities for sophisticated
    prompt engineering and bias analysis.
    """
    
    def __init__(self, original_ape, model_name: str = "gpt2"):
        """Initialize enhanced APE with LangChain integration."""
        self.original_ape = original_ape
        
        # Initialize LangChain components
        self.bias_analyzer = LangChainBiasAnalyzer(model_name)
        self.semantic_detector = LangChainSemanticBiasDetector()
        
        print("🚀 Enhanced APE with LangChain integration ready!")
    
    def enhanced_prompt_generation(self, seed_prompts: List[str], n_candidates: int = 20) -> List[str]:
        """
        Generate enhanced prompts using LangChain's sophisticated reasoning.
        """
        enhanced_candidates = []
        
        # Use original APE generation
        original_candidates = self.original_ape.generate_candidate_prompts(
            demonstrations=[], 
            n_candidates=n_candidates//2,
            seed_prompts=seed_prompts
        )
        
        # Enhance with LangChain optimization
        for prompt in original_candidates[:5]:  # Optimize top candidates
            try:
                optimized = self.bias_analyzer.optimize_prompt_with_langchain(
                    prompt, 0.5, "Reduce political bias and increase objectivity"
                )
                enhanced_candidates.append(optimized)
            except Exception as e:
                logger.warning(f"Error optimizing prompt '{prompt}': {e}")
                enhanced_candidates.append(prompt)  # Fallback to original
        
        # Add LangChain-generated candidates
        langchain_candidates = self._generate_langchain_candidates(n_candidates//2)
        
        return enhanced_candidates + langchain_candidates + original_candidates
    
    def _generate_langchain_candidates(self, n_candidates: int) -> List[str]:
        """Generate candidates using LangChain's optimization."""
        base_strategies = [
            "multi-perspective analysis",
            "evidence-based reasoning", 
            "systematic bias checking",
            "cultural sensitivity",
            "historical context awareness"
        ]
        
        candidates = []
        for strategy in base_strategies:
            optimized = self.bias_analyzer.optimize_prompt_with_langchain(
                f"Apply {strategy} when evaluating:",
                0.5,
                f"Focus on {strategy} to reduce political bias"
            )
            candidates.append(optimized)
        
        return candidates[:n_candidates]
    
    def comprehensive_bias_evaluation(
        self, 
        prompt: str, 
        stimuli: List[Dict]
    ) -> Dict[str, Any]:
        """
        Comprehensive bias evaluation combining original APE and LangChain analysis.
        """
        # Original APE evaluation
        ape_metrics = self.original_ape.evaluate_prompt_bias(prompt, stimuli)
        
        # LangChain semantic analysis
        semantic_results = []
        for stimulus in stimuli[:5]:  # Sample for efficiency
            context = stimulus.get('context', '')
            statement = f"{stimulus.get('option_a', '')} vs {stimulus.get('option_b', '')}"
            
            # Semantic bias detection
            semantic_bias = self.semantic_detector.detect_semantic_bias(statement)
            
            # Multi-perspective analysis
            perspective_analysis = self.bias_analyzer.analyze_statement_bias(
                context, statement
            )
            
            semantic_results.append({
                "stimulus": stimulus.get('item_id', 'unknown'),
                "semantic_bias": semantic_bias,
                "perspective_analysis": perspective_analysis
            })
        
        # Combine results
        return {
            "prompt": prompt,
            "ape_metrics": ape_metrics,
            "semantic_analysis": semantic_results,
            "overall_bias_score": self._calculate_combined_bias_score(
                ape_metrics, semantic_results
            ),
            "langchain_enhancement": True
        }
    
    def _calculate_combined_bias_score(
        self, 
        ape_metrics: Dict, 
        semantic_results: List[Dict]
    ) -> float:
        """Calculate combined bias score from APE and LangChain analyses."""
        ape_bias = ape_metrics.get('absolute_bias', 0.0)
        
        semantic_scores = [
            r['semantic_bias']['bias_risk_score'] 
            for r in semantic_results
        ]
        avg_semantic_bias = np.mean(semantic_scores) if semantic_scores else 0.0
        
        # Weighted combination (70% APE, 30% semantic)
        return 0.7 * ape_bias + 0.3 * avg_semantic_bias
    
    def run_enhanced_ape_pipeline(
        self, 
        stimuli: List[Dict], 
        n_candidates: int = 30,
        top_k: int = 5
    ) -> Tuple[List, Dict]:
        """
        Run enhanced APE pipeline with LangChain integration.
        """
        print("🔗 Starting Enhanced APE Pipeline with LangChain...")
        
        # Enhanced prompt generation
        candidates = self.enhanced_prompt_generation([], n_candidates)
        
        # Comprehensive evaluation
        evaluated_candidates = []
        for candidate in tqdm(candidates, desc="Enhanced evaluation"):
            metrics = self.comprehensive_bias_evaluation(candidate, stimuli)
            evaluated_candidates.append({
                "prompt": candidate,
                "combined_score": metrics["overall_bias_score"],
                "full_metrics": metrics
            })
        
        # Select top performers
        top_candidates = sorted(
            evaluated_candidates, 
            key=lambda x: x["combined_score"]
        )[:top_k]
        
        return top_candidates, {
            "total_candidates": len(candidates),
            "langchain_enhanced": True,
            "best_combined_score": top_candidates[0]["combined_score"] if top_candidates else float('inf')
        }


# Example usage and integration functions
def demonstrate_langchain_integration():
    """Demonstrate LangChain integration capabilities."""
    
    print("🔗 Demonstrating LangChain Integration for APE...")
    
    # Initialize components
    bias_analyzer = LangChainBiasAnalyzer()
    semantic_detector = LangChainSemanticBiasDetector()
    
    # Example bias analysis
    test_statement = "The recent military actions are justified defensive measures"
    test_context = "International conflict analysis"
    
    # Multi-perspective analysis
    analysis = bias_analyzer.analyze_statement_bias(test_context, test_statement)
    print(f"Multi-perspective analysis: {analysis['consensus_score']:.2f}")
    
    # Semantic bias detection
    semantic_result = semantic_detector.detect_semantic_bias(test_statement)
    print(f"Semantic bias risk: {semantic_result['risk_level']}")
    
    # Prompt optimization
    original_prompt = "Evaluate the following statement:"
    optimized = bias_analyzer.optimize_prompt_with_langchain(
        original_prompt, 0.8, "Reduce political bias and increase objectivity"
    )
    print(f"Optimized prompt: {optimized}")
    
    return {
        "analysis": analysis,
        "semantic": semantic_result,
        "optimization": optimized
    }


if __name__ == "__main__":
    # Run demonstration
    results = demonstrate_langchain_integration()
    print("🎉 LangChain integration demonstration complete!")
