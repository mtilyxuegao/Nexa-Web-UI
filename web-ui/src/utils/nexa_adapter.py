"""
Nexa AI LLM Adapter for Web-UI
专门处理 Nexa SDK 的 LLM 集成，支持多模态内容转换
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
    Nexa AI 聊天模型适配器
    继承自 ChatOpenAI，重写所有调用方法来拦截多模态请求并转换 base64 图片
    """
    
    nexa_base_url: str = Field(description="Nexa API 基础URL")
    nexa_model: str = Field(description="Nexa 模型名称")
    temp_dir: str = Field(default_factory=lambda: tempfile.mkdtemp(prefix="nexa_images_"))
    
    def __init__(self, model: str, base_url: str, temperature: float = 0.0, **kwargs):
        # 初始化父类，但使用虚假的 API key 因为 Nexa 不需要
        super().__init__(
            model=model,
            base_url=base_url,
            api_key="not-required",  # Nexa 不需要 API key
            temperature=temperature,
            nexa_base_url=base_url.rstrip('/'),
            nexa_model=model,
            **kwargs
        )
        
        # 创建临时目录用于存储转换的图片
        os.makedirs(self.temp_dir, exist_ok=True)
        logger.info(f"🖼️ Nexa adapter 初始化完成，图片临时目录：{self.temp_dir}")
    
    def __del__(self):
        """清理临时目录"""
        try:
            import shutil
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"🗑️ 清理临时目录：{self.temp_dir}")
        except:
            pass
    
    def _convert_base64_to_file(self, base64_data: str, file_extension: str = "png") -> str:
        """将 base64 图片数据转换为本地文件路径"""
        try:
            logger.info(f"🔄 开始转换 base64 图片数据...")
            
            # 移除 data URL 前缀（如果存在）
            if base64_data.startswith('data:'):
                # 提取文件类型
                if 'image/jpeg' in base64_data or 'image/jpg' in base64_data:
                    file_extension = "jpg"
                elif 'image/png' in base64_data:
                    file_extension = "png"
                elif 'image/webp' in base64_data:
                    file_extension = "webp"
                
                base64_data = base64_data.split(',', 1)[1]
            
            # 解码 base64 数据
            image_data = base64.b64decode(base64_data)
            
            # 在临时目录中创建文件
            import time
            filename = f"image_{int(time.time() * 1000)}.{file_extension}"
            file_path = os.path.join(self.temp_dir, filename)
            
            # 写入文件
            with open(file_path, 'wb') as f:
                f.write(image_data)
            
            logger.info(f"✅ base64 图片已转换为：{file_path} ({len(image_data)} bytes)")
            return file_path
            
        except Exception as e:
            logger.error(f"❌ base64 转换失败: {e}")
            raise ValueError(f"无法转换 base64 图片数据: {e}")
    
    def _process_multimodal_content(self, content: Any) -> Any:
        """处理多模态内容，将 base64 图片转换为文件路径"""
        if isinstance(content, list):
            # 处理多模态内容列表
            processed_content = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    image_url = item["image_url"]["url"]
                    if image_url.startswith("data:"):
                        # 转换 base64 为本地文件路径
                        local_file_path = self._convert_base64_to_file(image_url)
                        processed_content.append({
                            "type": "image_url",
                            "image_url": {"url": local_file_path}
                        })
                        logger.info(f"🔄 base64 图片已转换为本地路径：{local_file_path}")
                    else:
                        # 已经是文件路径或 URL，直接使用
                        processed_content.append(item)
                else:
                    processed_content.append(item)
            return processed_content
        else:
            # 纯文本内容
            return content
    
    def _convert_messages_to_openai_format(self, messages: List[BaseMessage]) -> List[Dict]:
        """转换 LangChain 消息格式为 OpenAI 格式，处理多模态内容"""
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                # 保持系统消息原样，不添加增强内容
                formatted_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                # 处理多模态内容
                processed_content = self._process_multimodal_content(msg.content)
                formatted_messages.append({"role": "user", "content": processed_content})
            elif isinstance(msg, AIMessage):
                formatted_messages.append({"role": "assistant", "content": msg.content})
        
        return formatted_messages
    
    def _extract_json_from_markdown(self, content: str) -> str:
        """从 markdown 代码块中提取 JSON 内容并修复格式问题"""
        import re
        import json
        
        original_content = content.strip()
        
        # 尝试匹配 ```json ... ``` 格式
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            extracted = json_match.group(1).strip()
            logger.info("🔧 从 markdown ```json``` 代码块中提取了 JSON 内容")
            fixed_json = self._fix_malformed_json(extracted)
            return fixed_json
        
        # 尝试匹配 ``` ... ``` 格式
        code_match = re.search(r'```\s*(.*?)\s*```', content, re.DOTALL)
        if code_match:
            extracted = code_match.group(1).strip()
            # 简单检查是否看起来像 JSON
            if extracted.strip().startswith('{') and extracted.strip().endswith('}'):
                logger.info("🔧 从 markdown ``` 代码块中提取了 JSON 内容")
                fixed_json = self._fix_malformed_json(extracted)
                return fixed_json
        
        # 如果没有代码块，尝试修复原始内容
        logger.info("📝 无需从 markdown 提取，检查并修复原始 JSON")
        fixed_json = self._fix_malformed_json(original_content)
        return fixed_json
    
    def _fix_malformed_json(self, json_str: str) -> str:
        """修复常见的 JSON 格式问题"""
        import json
        import re
        
        # 首先尝试直接解析
        try:
            json.loads(json_str)
            logger.info("✅ JSON 格式正确，无需修复")
            return json_str
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ 检测到 JSON 格式问题: {e}")
            
            # 修复常见问题：多余的逗号和结构错误
            fixed = json_str
            
            # 修复模式: {"current_state": {...}}, "action": [...] -> {"current_state": {...}, "action": [...]}
            # 使用正则表达式来精确匹配和修复结构问题
            import re
            pattern = r'("current_state": \{.*?\})\}, "action":'
            replacement = r'\1, "action":'
            
            if re.search(pattern, fixed):
                logger.info("🔧 修复 current_state 和 action 之间的结构错误")
                fixed = re.sub(pattern, replacement, fixed)
            
            # 尝试解析修复后的 JSON
            try:
                json.loads(fixed)
                logger.info("✅ JSON 修复成功")
                return fixed
            except json.JSONDecodeError:
                logger.warning("❌ JSON 修复失败，返回原始内容")
                return json_str
    
    def _call_nexa_api(self, messages: List[BaseMessage], **kwargs) -> str:
        """调用 Nexa API 的核心方法"""
        # 转换消息格式
        formatted_messages = self._convert_messages_to_openai_format(messages)
        
        # 准备请求载荷，使用适中温度以获得更灵活的输出
        payload = {
            "model": self.nexa_model,
            "messages": formatted_messages,
            "temperature": 0.5,  # 提高温度以增强指令遵循性
            "max_tokens": kwargs.get("max_tokens", 2048)  # 增加最大 token 数
        }
        
        # 调试日志
        logger.info(f"🔍 处理前的消息数量: {len(messages)}")
        logger.info(f"🔍 处理后的载荷: {json.dumps(payload, indent=2, ensure_ascii=False)[:1000]}...")
        
        try:
            # 尝试使用 chat/completions 端点
            response = requests.post(
                f"{self.nexa_base_url}/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            logger.info(f"✅ Nexa API 调用成功，返回内容长度: {len(content)}")
            logger.info(f"📝 模型原始输出: {content[:500]}...")
            
            # 从 markdown 代码块中提取 JSON（如果有的话）
            processed_content = self._extract_json_from_markdown(content)
            return processed_content
            
        except Exception as e:
            logger.error(f"❌ chat/completions 调用失败: {e}")
            # 回退到 completions 端点
            try:
                # 将消息转换为单一提示
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
                logger.info(f"✅ Nexa completions 调用成功，返回内容长度: {len(content)}")
                logger.info(f"📝 模型原始输出 (completions): {content[:500]}...")
                
                # 从 markdown 代码块中提取 JSON（如果有的话）
                processed_content = self._extract_json_from_markdown(content)
                return processed_content
                
            except Exception as fallback_e:
                logger.error(f"❌ completions 回退也失败: {fallback_e}")
                raise ValueError(
                    f"Nexa API 调用完全失败。Chat错误: {e}, Completions错误: {fallback_e}"
                )
    
    # 重写所有可能的调用方法
    
    def invoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AIMessage:
        """同步调用方法"""
        logger.info("📞 NexaChatLLM.invoke 被调用")
        
        # 转换输入为消息列表
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
        """异步调用方法"""
        logger.info(f"📞 NexaChatLLM.ainvoke 被调用，参数: {list(kwargs.keys())}")
        
        # 过滤掉不支持的参数
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['response_format']}
        
        # 在异步环境中调用同步方法
        loop = asyncio.get_event_loop()
        
        # 创建一个包装函数来正确传递参数
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
        """同步生成方法"""
        logger.info("📞 NexaChatLLM._generate 被调用")
        
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
        """异步生成方法"""
        logger.info(f"📞 NexaChatLLM._agenerate 被调用，参数: {list(kwargs.keys())}")
        
        # 过滤掉不支持的参数
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['response_format']}
        
        # 在异步环境中调用同步方法
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
        """批量处理方法"""
        logger.info(f"📞 NexaChatLLM.batch 被调用，输入数量: {len(inputs)}")
        
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
        """异步批量处理方法"""
        logger.info(f"📞 NexaChatLLM.abatch 被调用，输入数量: {len(inputs)}")
        
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
        """流式处理方法（Nexa 当前不支持流式，返回完整响应）"""
        logger.info("📞 NexaChatLLM.stream 被调用（转换为非流式）")
        
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
        """异步流式处理方法（Nexa 当前不支持流式，返回完整响应）"""
        logger.info("📞 NexaChatLLM.astream 被调用（转换为非流式）")
        
        result = await self.ainvoke(input, config, stop=stop, **kwargs)
        yield result
    
    
    def _messages_to_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """将消息列表转换为单一提示字符串（用于 completions 端点）"""
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"System: {msg['content']}\n"
            elif msg["role"] == "user":
                if isinstance(msg["content"], list):
                    # 多模态内容，只提取文本部分
                    text_parts = [item["text"] for item in msg["content"] if item.get("type") == "text"]
                    content = " ".join(text_parts) if text_parts else "图片描述请求"
                    prompt += f"User: {content}\n"
                else:
                    prompt += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                prompt += f"Assistant: {msg['content']}\n"
        prompt += "Assistant: "
        return prompt


def create_nexa_llm(model: str, base_url: str, temperature: float = 0.0, **kwargs) -> NexaChatLLM:
    """
    创建 Nexa LLM 实例的工厂函数
    """
    logger.info(f"🏭 创建 Nexa LLM 实例: model={model}, base_url={base_url}, temperature={temperature}")
    return NexaChatLLM(model=model, base_url=base_url, temperature=temperature, **kwargs)