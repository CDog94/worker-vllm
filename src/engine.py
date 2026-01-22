import os
import logging
import json
import asyncio
import time

from dotenv import load_dotenv
from typing import AsyncGenerator

from vllm import AsyncLLMEngine
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_models import (
    BaseModelPath,
    LoRAModulePath,
    OpenAIServingModels,
)

from utils import DummyRequest, JobInput, BatchSize, create_error_response
from constants import (
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BATCH_SIZE_GROWTH_FACTOR,
    DEFAULT_MIN_BATCH_SIZE,
)
from tokenizer import TokenizerWrapper
from engine_args import get_engine_args


class vLLMEngine:
    def __init__(self, engine=None):
        load_dotenv()  # For local development
        self.engine_args = get_engine_args()
        logging.info(f"Engine args: {self.engine_args}")

        # Initialize vLLM engine first
        self.llm = self._initialize_llm() if engine is None else engine.llm

        # Only create custom tokenizer wrapper if not using mistral tokenizer mode
        # For mistral models, let vLLM handle tokenizer initialization
        if getattr(self.engine_args, "tokenizer_mode", None) != "mistral":
            self.tokenizer = TokenizerWrapper(
                self.engine_args.tokenizer or self.engine_args.model,
                self.engine_args.tokenizer_revision,
                self.engine_args.trust_remote_code,
            )
        else:
            # For mistral models, we'll get the tokenizer from transformers as a fallback (for chat templates)
            self.tokenizer = None

        self.max_concurrency = int(
            os.getenv("MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY)
        )
        self.default_batch_size = int(
            os.getenv("DEFAULT_BATCH_SIZE", DEFAULT_BATCH_SIZE)
        )
        self.batch_size_growth_factor = int(
            os.getenv("BATCH_SIZE_GROWTH_FACTOR", DEFAULT_BATCH_SIZE_GROWTH_FACTOR)
        )
        self.min_batch_size = int(
            os.getenv("MIN_BATCH_SIZE", DEFAULT_MIN_BATCH_SIZE)
        )

    def _get_tokenizer_for_chat_template(self):
        """Get tokenizer for chat template application."""
        if self.tokenizer is not None:
            return self.tokenizer

        # Fallback: for tokenizer_mode=mistral (or if TokenizerWrapper isn't present),
        # create a minimal wrapper around HF tokenizer to apply chat templates.
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                self.engine_args.tokenizer or self.engine_args.model,
                revision=self.engine_args.tokenizer_revision or "main",
                trust_remote_code=self.engine_args.trust_remote_code,
            )

            class MinimalTokenizerWrapper:
                def __init__(self, tok):
                    self.tokenizer = tok
                    self.custom_chat_template = os.getenv("CUSTOM_CHAT_TEMPLATE")
                    self.has_chat_template = bool(
                        getattr(self.tokenizer, "chat_template", None)
                    ) or bool(self.custom_chat_template)
                    if self.custom_chat_template and isinstance(
                        self.custom_chat_template, str
                    ):
                        self.tokenizer.chat_template = self.custom_chat_template

                def apply_chat_template(self, input_):
                    if isinstance(input_, list):
                        if not self.has_chat_template:
                            raise ValueError(
                                "Chat template does not exist for this model; "
                                "provide a single string instead of a list of messages."
                            )
                        messages = input_
                    elif isinstance(input_, str):
                        messages = [{"role": "user", "content": input_}]
                    else:
                        raise ValueError("Input must be a string or a list of messages")

                    return self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

            return MinimalTokenizerWrapper(tokenizer)
        except Exception as e:
            logging.error(f"Failed to create fallback tokenizer: {e}")
            raise

    def dynamic_batch_size(self, current_batch_size, batch_size_growth_factor):
        return min(
            current_batch_size * batch_size_growth_factor, self.default_batch_size
        )

    async def generate(self, job_input: JobInput):
        try:
            async for batch in self._generate_vllm(
                llm_input=job_input.llm_input,
                validated_sampling_params=job_input.sampling_params,
                batch_size=job_input.max_batch_size,
                stream=job_input.stream,
                apply_chat_template=job_input.apply_chat_template,
                request_id=job_input.request_id,
                batch_size_growth_factor=job_input.batch_size_growth_factor,
                min_batch_size=job_input.min_batch_size,
            ):
                yield batch
        except Exception as e:
            yield {"error": create_error_response(str(e)).model_dump()}

    async def _generate_vllm(
        self,
        llm_input,
        validated_sampling_params,
        batch_size,
        stream,
        apply_chat_template,
        request_id,
        batch_size_growth_factor,
        min_batch_size: str,
    ) -> AsyncGenerator[dict, None]:
        if apply_chat_template or isinstance(llm_input, list):
            tokenizer_wrapper = self._get_tokenizer_for_chat_template()
            llm_input = tokenizer_wrapper.apply_chat_template(llm_input)

        results_generator = self.llm.generate(
            llm_input, validated_sampling_params, request_id
        )

        n_responses = validated_sampling_params.n
        n_input_tokens = 0
        is_first_output = True

        last_output_texts = ["" for _ in range(n_responses)]
        token_counters = {"batch": 0, "total": 0}

        batch_obj = {"choices": [{"tokens": []} for _ in range(n_responses)]}

        max_batch_size = batch_size or self.default_batch_size
        batch_size_growth_factor = (
            batch_size_growth_factor or self.batch_size_growth_factor
        )
        min_batch_size = min_batch_size or self.min_batch_size
        batch_size_state = BatchSize(
            max_batch_size, min_batch_size, batch_size_growth_factor
        )

        async for request_output in results_generator:
            if is_first_output:
                n_input_tokens = len(request_output.prompt_token_ids)
                is_first_output = False

            for output in request_output.outputs:
                output_index = output.index
                token_counters["total"] += 1

                if stream:
                    new_output = output.text[len(last_output_texts[output_index]) :]
                    batch_obj["choices"][output_index]["tokens"].append(new_output)
                    token_counters["batch"] += 1

                    if token_counters["batch"] >= batch_size_state.current_batch_size:
                        batch_obj["usage"] = {
                            "input": n_input_tokens,
                            "output": token_counters["total"],
                        }
                        yield batch_obj
                        batch_obj = {
                            "choices": [{"tokens": []} for _ in range(n_responses)]
                        }
                        token_counters["batch"] = 0
                        batch_size_state.update()

                last_output_texts[output_index] = output.text

        if not stream:
            for output_index, output_text in enumerate(last_output_texts):
                batch_obj["choices"][output_index]["tokens"] = [output_text]
            token_counters["batch"] += 1

        if token_counters["batch"] > 0:
            batch_obj["usage"] = {
                "input": n_input_tokens,
                "output": token_counters["total"],
            }
            yield batch_obj

    def _initialize_llm(self):
        try:
            start = time.time()
            engine = AsyncLLMEngine.from_engine_args(self.engine_args)
            end = time.time()
            logging.info(f"Initialized vLLM engine in {end - start:.2f}s")
            return engine
        except Exception as e:
            logging.error("Error initializing vLLM engine: %s", e)
            raise


class OpenAIvLLMEngine(vLLMEngine):
    def __init__(self, vllm_engine):
        super().__init__(vllm_engine)

        self.served_model_name = (
            os.getenv("OPENAI_SERVED_MODEL_NAME_OVERRIDE") or self.engine_args.model
        )
        self.response_role = os.getenv("OPENAI_RESPONSE_ROLE") or "assistant"
        self.lora_adapters = self._load_lora_adapters()

        # NOTE: asyncio.run() is only safe if you are not already inside an event loop.
        # If this is constructed inside an async context (e.g. FastAPI lifespan), call
        # `await engine.async_init()` instead of constructing with asyncio.run().
        asyncio.run(self._initialize_engines())

        # Handle both integer and boolean string values for RAW_OPENAI_OUTPUT
        raw_output_env = os.getenv("RAW_OPENAI_OUTPUT", "1")
        if raw_output_env.lower() in ("true", "false"):
            self.raw_openai_output = raw_output_env.lower() == "true"
        else:
            self.raw_openai_output = bool(int(raw_output_env))

    def _load_lora_adapters(self):
        adapters = []
        try:
            adapters = json.loads(os.getenv("LORA_MODULES", "[]"))
        except Exception as e:
            logging.info(f"---Initialized adapter json load error: {e}")

        out = []
        for adapter in adapters:
            try:
                out.append(LoRAModulePath(**adapter))
                logging.info(f"---Initialized adapter: {adapter}")
            except Exception as e:
                logging.info(f"---Initialized adapter not worked: {e}")
                continue
        return out

    async def _initialize_engines(self):
        # vLLM 0.14.0 compatibility:
        # - OpenAIServingModels(engine_client, base_model_paths, *, lora_modules=None)
        # - OpenAIServingChat(engine_client, models, response_role, *, request_logger, chat_template, ...)
        # - OpenAIServingCompletion(engine_client, models, *, request_logger, ...)

        self.base_model_paths = [
            BaseModelPath(name=self.engine_args.model, model_path=self.engine_args.model)
        ]

        self.serving_models = OpenAIServingModels(
            engine_client=self.llm,
            base_model_paths=self.base_model_paths,
            lora_modules=self.lora_adapters,
        )
        await self.serving_models.init_static_loras()

        # Get chat template from your TokenizerWrapper if present
        chat_template = None
        if self.tokenizer and hasattr(self.tokenizer, "tokenizer"):
            chat_template = getattr(self.tokenizer.tokenizer, "chat_template", None)

        self.chat_engine = OpenAIServingChat(
            engine_client=self.llm,
            models=self.serving_models,
            response_role=self.response_role,
            request_logger=None,
            chat_template=chat_template,
            chat_template_content_format="auto",
            reasoning_parser=os.getenv("REASONING_PARSER", ""),  # must be str
            enable_auto_tools=os.getenv("ENABLE_AUTO_TOOL_CHOICE", "false").lower()
            == "true",
            tool_parser=os.getenv("TOOL_CALL_PARSER") or None,
            enable_prompt_tokens_details=False,
        )

        self.completion_engine = OpenAIServingCompletion(
            engine_client=self.llm,
            models=self.serving_models,
            request_logger=None,
        )

    async def generate(self, openai_request: JobInput):
        if openai_request.openai_route == "/v1/models":
            yield await self._handle_model_request()
        elif openai_request.openai_route in ("/v1/chat/completions", "/v1/completions"):
            async for response in self._handle_chat_or_completion_request(openai_request):
                yield response
        else:
            yield create_error_response("Invalid route").model_dump()

    async def _handle_model_request(self):
        models = await self.serving_models.show_available_models()
        return models.model_dump()

    async def _handle_chat_or_completion_request(self, openai_request: JobInput):
        if openai_request.openai_route == "/v1/chat/completions":
            request_class = ChatCompletionRequest
            generator_function = self.chat_engine.create_chat_completion
        else:
            request_class = CompletionRequest
            generator_function = self.completion_engine.create_completion

        try:
            request = request_class(**openai_request.openai_input)
        except Exception as e:
            yield create_error_response(str(e)).model_dump()
            return

        dummy_request = DummyRequest()
        response_generator = await generator_function(request, raw_request=dummy_request)

        if (not openai_request.openai_input.get("stream")) or isinstance(
            response_generator, ErrorResponse
        ):
            yield response_generator.model_dump()
            return

        # streaming batching
        batch = []
        batch_token_counter = 0
        batch_size_state = BatchSize(
            self.default_batch_size, self.min_batch_size, self.batch_size_growth_factor
        )

        async for chunk_str in response_generator:
            if "data" not in chunk_str:
                continue

            if self.raw_openai_output:
                data = chunk_str
            elif "[DONE]" in chunk_str:
                continue
            else:
                # chunk_str is something like: "data: {...}\n\n"
                data = json.loads(chunk_str.removeprefix("data: ").rstrip("\n\n"))

            batch.append(data)
            batch_token_counter += 1

            if batch_token_counter >= batch_size_state.current_batch_size:
                if self.raw_openai_output:
                    yield "".join(batch)
                else:
                    yield batch
                batch = []
                batch_token_counter = 0
                batch_size_state.update()

        if batch:
            if self.raw_openai_output:
                yield "".join(batch)
            else:
                yield batch
