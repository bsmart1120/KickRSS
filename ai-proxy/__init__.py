"""
AI Proxy — thinking-disable 透明代理
======================================
接收 OpenAI 兼容的 /v1/chat/completions 请求，
根据 model 字段自动匹配后端配置，注入 thinking-disable 参数，
转发到真实 AI 后端，支持流式/非流式双模式及 400 重试。
"""
