"""A small instruct model running in-process. The provider that needs no account.

OFFLINE ONLY. See ``arag/local/__init__.py`` for why this is not in ``arag.agent``.
"""

from __future__ import annotations

from typing import cast, TYPE_CHECKING

from arag.agent.llm import LLMError, LLMErrorKind, LLMResponse
from arag.obs import get_logger

if TYPE_CHECKING:
    from arag.agent.llm import LLMRequest

log = get_logger(__name__)


class LocalTransformersProvider:
    """A small instruct model running in-process on CPU. No account, no quota, no network.

    ## Why this exists

    Three attempts to measure generation died on account state nobody in this repository
    controls: a retired model (404), a 20/day free quota, an un-prefixed env var, and a
    suspended Google project (403). The engine was ready each time; the blocker never was.

    This provider makes the pipeline verifiable without anyone's billing console. It is a
    **test instrument, not the production generator** - DESIGN §5.6 chooses hosted Gemini
    for the online path and that is unchanged. A 1.5B model's answers are not Gemini's, so
    numbers from here verify that the ENGINE works end to end; they are not the v3 quality
    measurement and must never be recorded as one.

    ## Why a weak model is a good test instrument

    Counter-intuitively it is better than a strong one for this job. The paths that matter
    in ``GeneratingEngine`` are the adversarial ones - malformed JSON, fabricated citations,
    truncation - and a frontier model at temperature 0 almost never produces them. A small
    model produces them readily, against a real tokeniser, a real sampler and a real context
    window. If the verification layer holds here it is being exercised, not bypassed.

    ## The import boundary

    ``transformers`` and ``torch`` are offline-only (DESIGN §9) and ``arag.agent`` is an
    ONLINE package, so this module must not import them at module scope -
    ``tests/test_import_closure.py`` fails the build if it does. The import is inside
    ``__init__``, which is the same deliberate laziness ``GeminiProvider`` uses for httpx.
    """

    name = "local-transformers"

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        *,
        device: str = "cpu",
        max_input_tokens: int = 8192,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = model
        self._torch = torch
        # local_files_only, and it is not an optimisation.
        #
        # `from_pretrained` contacts huggingface.co to check for updates even when every
        # weight is already cached. The first smoke-test run died on
        # `[Errno 11001] getaddrinfo failed` while HEADing `added_tokens.json` - a file that
        # does not exist for this model - and then burned its remaining time on five
        # retries without generating a single token.
        #
        # That made this class's own docstring false: it promises "no account, no quota, no
        # network", and the whole reason the class exists is that three measurement attempts
        # died on external availability. A provider built to remove a network dependency
        # must not have one.
        self._tokeniser = AutoTokenizer.from_pretrained(model, local_files_only=True)
        # `.to(device)` rather than `device_map=`: device_map pulls in `accelerate`, which
        # this project does not depend on and does not need. Multi-device placement is
        # meaningless for a 1.5B model on one CPU.
        # transformers' stubs misreport this chain: `.to(device)` resolves to a wrapped
        # descriptor whose __call__ is typed to expect a PreTrainedModel, so a plain device
        # string is flagged. The call is correct and documented; the stub is wrong, so it is
        # narrowly ignored rather than reshaped around.
        self._model = AutoModelForCausalLM.from_pretrained(
            model, dtype=torch.float32, local_files_only=True
        ).to(device)  # type: ignore[arg-type]
        self._model.eval()
        self._max_input_tokens = max_input_tokens
        log.info("local_model_loaded", model=model, device=device)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        import asyncio

        # Generation is seconds-to-minutes of blocking CPU. Run it off the event loop so an
        # async caller is not silently serialised behind it - the engine is async because
        # the real provider is network-bound, and a sync call here would make that a lie.
        return await asyncio.to_thread(self._complete_sync, request)

    def _complete_sync(self, request: LLMRequest) -> LLMResponse:
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.user})

        text = self._tokeniser.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokeniser([text], return_tensors="pt")
        input_tokens = int(inputs["input_ids"].shape[1])
        if input_tokens > self._max_input_tokens:
            # Refused rather than silently truncated. A quietly clipped prompt drops
            # passages the model is then asked to cite, which manufactures exactly the
            # fabricated-citation failure the engine exists to catch - from our own code.
            raise LLMError(
                LLMErrorKind.MALFORMED,
                f"prompt is {input_tokens} tokens, over the {self._max_input_tokens} "
                f"limit for {self.model}. Reduce top_k rather than truncating: a clipped "
                f"prompt removes passages the model is still asked to cite.",
            )

        with self._torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=request.max_output_tokens,
                do_sample=request.temperature > 0,
                temperature=request.temperature if request.temperature > 0 else None,
                pad_token_id=self._tokeniser.eos_token_id,
            )
        new_tokens = generated[0][input_tokens:]
        output_tokens = int(new_tokens.shape[0])
        # decode of a single token sequence returns str; the stub widens the
        # return to str | list[str] (that is the batch_decode shape), so narrow it.
        answer = cast(str, self._tokeniser.decode(new_tokens, skip_special_tokens=True))

        # Same MAX_TOKENS semantics as the hosted provider, so the engine's truncation
        # branch is exercised identically rather than being a Gemini-only path.
        truncated = output_tokens >= request.max_output_tokens
        return LLMResponse(
            text=answer,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason="MAX_TOKENS" if truncated else "STOP",
            truncated=truncated,
        )
