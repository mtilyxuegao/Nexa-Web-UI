"""
Nexa AI LLM Adapter for Web-UI
Handles Nexa SDK integration with LangChain, supporting multimodal content conversion
"""

import requests
import json
import base64
import tempfile
import os
import asyncio
from pathlib import Path
from typing import List, Optional, Any, Dict, Union, AsyncIterator, Iterator
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration, LLMResult
from langchain_core.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.runnables import RunnableConfig
from pydantic import Field
import logging

logger = logging.getLogger(__name__)


class NexaChatLLM(ChatOpenAI):
    """
    Nexa AI Chat Model Adapter
    Extends ChatOpenAI to intercept multimodal requests and convert base64 images to file paths
    """
    
    nexa_base_url: str = Field(description="Nexa API base URL")
    nexa_model: str = Field(description="Nexa model name")
    temp_dir: str = Field(default_factory=lambda: tempfile.mkdtemp(prefix="nexa_images_"))
    
    def __init__(self, model: str, base_url: str, temperature: float = 0.0, **kwargs):
        # Initialize parent class with dummy API key (Nexa doesn't require one)
        super().__init__(
            model=model,
            base_url=base_url,
            api_key="not-required",  # Nexa doesn't need API key
            temperature=temperature,
            nexa_base_url=base_url.rstrip('/'),
            nexa_model=model,
            **kwargs
        )
        
        # Create temp directory for storing converted images
        os.makedirs(self.temp_dir, exist_ok=True)
        logger.info(f"Nexa adapter initialized with temp directory: {self.temp_dir}")
    
    def __del__(self):
        """Clean up temporary directory"""
        try:
            import shutil
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temp directory: {self.temp_dir}")
        except:
            pass
    
    def _convert_base64_to_file(self, base64_data: str, file_extension: str = "png") -> str:
        """Convert base64 image data to local file path"""
        try:
            # Remove data URL prefix if present
            if base64_data.startswith('data:'):
                # Extract file type
                if 'image/jpeg' in base64_data or 'image/jpg' in base64_data:
                    file_extension = "jpg"
                elif 'image/png' in base64_data:
                    file_extension = "png"
                elif 'image/webp' in base64_data:
                    file_extension = "webp"
                
                base64_data = base64_data.split(',', 1)[1]
            
            # Decode base64 data
            image_data = base64.b64decode(base64_data)
            
            # Create file in temp directory
            import time
            filename = f"image_{int(time.time() * 1000)}.{file_extension}"
            file_path = os.path.join(self.temp_dir, filename)
            
            # Write file
            with open(file_path, 'wb') as f:
                f.write(image_data)
            
            logger.info(f"✅ Converted base64 to file: {file_path} ({len(image_data)} bytes)")
            return file_path
            
        except Exception as e:
            logger.error(f"❌ Base64 conversion failed: {e}")
            raise ValueError(f"Failed to convert base64 image: {e}")
    
    def _process_multimodal_content(self, content: Any) -> Any:
        """Process multimodal content, converting base64 images to file paths"""
        if isinstance(content, list):
            # Process multimodal content list
            processed_content = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    image_url = item["image_url"]["url"]
                    if image_url.startswith("data:"):
                        # Convert base64 to local file path
                        local_file_path = self._convert_base64_to_file(image_url)
                        processed_content.append({
                            "type": "image_url",
                            "image_url": {"url": local_file_path}
                        })
                    else:
                        # Already a file path or URL, use directly
                        processed_content.append(item)
                else:
                    processed_content.append(item)
            return processed_content
        else:
            # Plain text content
            return content
    
    def _convert_messages_to_openai_format(self, messages: List[BaseMessage]) -> List[Dict]:
        """Convert LangChain message format to OpenAI format, handling multimodal content"""
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                formatted_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                # Process multimodal content
                processed_content = self._process_multimodal_content(msg.content)
                formatted_messages.append({"role": "user", "content": processed_content})
            elif isinstance(msg, AIMessage):
                formatted_messages.append({"role": "assistant", "content": msg.content})
        
        return formatted_messages
    
    def _fix_action_names(self, json_str: str) -> str:
        """Fix common action name errors (e.g., click_element -> click_element_by_index)"""
        try:
            data = json.loads(json_str)
            
            # Action name mapping
            action_mappings = {
                "click_element": "click_element_by_index",
                "open_new_tab": "open_tab",
                "new_tab": "open_tab",
            }
            
            # Check and fix action names
            if "action" in data and isinstance(data["action"], list):
                for action_item in data["action"]:
                    if isinstance(action_item, dict):
                        for wrong_name, correct_name in action_mappings.items():
                            if wrong_name in action_item:
                                action_item[correct_name] = action_item.pop(wrong_name)
                                logger.info(f"🔧 Fixed action name: {wrong_name} -> {correct_name}")
                                return json.dumps(data, ensure_ascii=False)
            
            return json_str
            
        except Exception as e:
            logger.warning(f"⚠️ Action name correction failed: {e}")
            return json_str
    
    def _call_nexa_api(self, messages: List[BaseMessage], **kwargs) -> str:
        """Core method for calling Nexa API"""
        # Convert message format
        formatted_messages = self._convert_messages_to_openai_format(messages)
        
        # Prepare request payload
        payload = {
            "model": self.nexa_model,
            "messages": formatted_messages,
            "temperature": 0.5,
            "max_tokens": kwargs.get("max_tokens", 2048)
        }
        
        try:
            # Call chat/completions endpoint
            response = requests.post(
                f"{self.nexa_base_url}/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            logger.info(f"✅ Nexa API call successful, content length: {len(content)}")
            
            # Fix common action name errors
            processed_content = self._fix_action_names(content)
            
            return processed_content
            
        except Exception as e:
            logger.error(f"❌chat/completions call failed: {e}")
            # Fallback to completions endpoint
            try:
                prompt = self._messages_to_prompt(formatted_messages)
                payload = {
                    "model": self.nexa_model,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "max_tokens": kwargs.get("max_tokens", 1024)
                }
                
                response = requests.post(
                    f"{self.nexa_base_url}/completions",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()
                
                result = response.json()
                content = result["choices"][0]["text"].strip()
                logger.info(f"Nexa completions call successful, content length: {len(content)}")
                
                # Fix common action name errors
                processed_content = self._fix_action_names(content)
                
                return processed_content
                
            except Exception as fallback_e:
                logger.error(f"completions fallback failed: {fallback_e}")
                raise ValueError(f"Nexa API call completely failed. Chat error: {e}, Completions error: {fallback_e}")
    
    # Override all possible invocation methods
    
    def invoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AIMessage:
        """Synchronous invocation method"""
        logger.info("NexaChatLLM.invoke called")
        
        # Convert input to message list
        if isinstance(input, str):
            messages = [HumanMessage(content=input)]
        else:
            messages = input
        
        content = self._call_nexa_api(messages, **kwargs)
        return AIMessage(content=content)
    
    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AIMessage:
        """Asynchronous invocation method"""
        logger.info(f"NexaChatLLM.ainvoke called")
        
        # Filter out unsupported parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['response_format']}
        
        # Call sync method in async context
        loop = asyncio.get_event_loop()
        
        # Create wrapper function to pass parameters correctly
        def sync_call():
            return self.invoke(input, config, stop=stop, **filtered_kwargs)
        
        return await loop.run_in_executor(None, sync_call)
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous generation method"""
        logger.info("NexaChatLLM._generate called")
        
        content = self._call_nexa_api(messages, **kwargs)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronous generation method"""
        logger.info(f"NexaChatLLM._agenerate called")
        
        # Filter out unsupported parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['response_format']}
        
        # Call sync method in async context
        loop = asyncio.get_event_loop()
        
        def sync_call():
            return self._call_nexa_api(messages, **filtered_kwargs)
        
        content = await loop.run_in_executor(None, sync_call)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )
    
    def batch(
        self,
        inputs: List[LanguageModelInput],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> List[AIMessage]:
        """Batch processing method"""
        logger.info(f"NexaChatLLM.batch called with {len(inputs)} inputs")
        
        results = []
        for input_item in inputs:
            try:
                result = self.invoke(input_item, **kwargs)
                results.append(result)
            except Exception as e:
                if return_exceptions:
                    results.append(e)
                else:
                    raise e
        
        return results
    
    async def abatch(
        self,
        inputs: List[LanguageModelInput],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> List[AIMessage]:
        """Asynchronous batch processing method"""
        logger.info(f"NexaChatLLM.abatch called with {len(inputs)} inputs")
        
        tasks = []
        for input_item in inputs:
            task = self.ainvoke(input_item, **kwargs)
            tasks.append(task)
        
        if return_exceptions:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            results = await asyncio.gather(*tasks)
        
        return results
    
    def stream(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[AIMessage]:
        """Streaming method (Nexa doesn't support streaming, returns full response)"""
        logger.info("NexaChatLLM.stream called (converting to non-streaming)")
        
        result = self.invoke(input, config, stop=stop, **kwargs)
        yield result
    
    async def astream(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[AIMessage]:
        """Asynchronous streaming method (Nexa doesn't support streaming, returns full response)"""
        logger.info("NexaChatLLM.astream called (converting to non-streaming)")
        
        result = await self.ainvoke(input, config, stop=stop, **kwargs)
        yield result
    
    
    def _messages_to_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Convert message list to single prompt string (for completions endpoint)"""
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"System: {msg['content']}\n"
            elif msg["role"] == "user":
                if isinstance(msg["content"], list):
                    # Multimodal content, extract text parts only
                    text_parts = [item["text"] for item in msg["content"] if item.get("type") == "text"]
                    content = " ".join(text_parts) if text_parts else "Image description request"
                    prompt += f"User: {content}\n"
                else:
                    prompt += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                prompt += f"Assistant: {msg['content']}\n"
        prompt += "Assistant: "
        return prompt


def create_nexa_llm(model: str, base_url: str, temperature: float = 0.0, **kwargs) -> NexaChatLLM:
    """
    Factory function to create Nexa LLM instance
    """
    logger.info(f"Creating Nexa LLM instance: model={model}, base_url={base_url}, temperature={temperature}")
    return NexaChatLLM(model=model, base_url=base_url, temperature=temperature, **kwargs)
