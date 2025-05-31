import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# === Phoenix API token ===
PHOENIX_API_KEY = os.getenv("PHOENIX_API_KEY")
PHOENIX_PROJECT = "fastapi-mistral-load-test"

# === Ensure CodeCarbon output directory exists ===
LOG_DIR = "emissions_logs"
os.makedirs(LOG_DIR, exist_ok=True)

# === Energy + Tracing ===
from codecarbon import EmissionsTracker
from openinference.semconv.trace import SpanAttributes

# === OpenTelemetry Setup ===
from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# === FastAPI App ===
app = FastAPI(title="Mistral MathChat Inference")

# === Instrument FastAPI with OpenTelemetry ===
otlp_exporter = OTLPSpanExporter(
    endpoint="https://app.phoenix.arize.com/v1/traces",
    headers={"api_key": PHOENIX_API_KEY}
)

trace.set_tracer_provider(
    TracerProvider(resource=Resource.create({SERVICE_NAME: PHOENIX_PROJECT}))
)
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(otlp_exporter)
)
tracer = trace.get_tracer(__name__)
FastAPIInstrumentor().instrument_app(app)

# === Hugging Face Login ===
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(HF_TOKEN)
    print("✅ Successfully logged in to Hugging Face!")
else:
    raise ValueError("❌ HF_TOKEN environment variable is not set!")

# === Load tokenizer + model once at startup ===
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=False, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_side = "right"

model = AutoModelForCausalLM.from_pretrained(
     repo_id,
     torch_dtype=torch.float16,
     device_map="auto",
     token=HF_TOKEN,
 )
model.eval()

# === Optional: Track total energy for app lifetime ===
startup_tracker = EmissionsTracker(
    project_name=PHOENIX_PROJECT,
    output_dir=LOG_DIR,
    measure_power_secs=5,
    log_level="ERROR",
)

@app.on_event("startup")
def on_startup():
    startup_tracker.start()

@app.on_event("shutdown")
def on_shutdown():
    startup_tracker.stop()

# === Request Schema ===
class InferenceRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.95

# === Energy-tracked model inference ===
def generate_with_energy(
    prompt: str,
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.95,
):
    tracker = EmissionsTracker(
        project_name=PHOENIX_PROJECT,
        output_dir=LOG_DIR,
        measure_power_secs=1,
        log_level="error",
    )

    with tracer.start_as_current_span(
        "mistral_generate",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: "llm-inference",
            "custom.prompt": prompt
        },
    ) as span:
        tracker.start()

        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].cuda()
        attention_mask = inputs["attention_mask"].cuda()

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=tokenizer.eos_token_id,
            )

        tracker.stop()
        emissions_kg = tracker.final_emissions
        energy_kwh = tracker.final_emissions_data.energy_consumed  # same as _total_energy.kWh


        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract useful metadata
        emissions_data = tracker.final_emissions_data

        # Send all attributes to Phoenix
        span.set_attribute("custom.prompt", prompt)
        span.set_attribute("custom.response", response)
        span.set_attribute("custom.energy.kwh", round(energy_kwh, 6))
        span.set_attribute("custom.emissions.kg", round(emissions_kg, 6))
        span.set_attribute("custom.country", emissions_data.country_name)
        span.set_attribute("custom.region", emissions_data.region)
        span.set_attribute("custom.gpu_model", emissions_data.gpu_model)
        span.set_attribute("custom.cpu_model", emissions_data.cpu_model)
        span.set_attribute("custom.gpu_power", round(emissions_data.gpu_power, 2))
        span.set_attribute("custom.cpu_power", round(emissions_data.cpu_power, 2))
        span.set_attribute("custom.ram_power", round(emissions_data.ram_power, 2))
        span.set_attribute("custom.total_energy_kwh", round(emissions_data.energy_consumed, 6))
        span.set_attribute("custom.duration_sec", round(emissions_data.duration, 3))

    return response

# === Inference Endpoint ===
@app.post("/generate")
async def generate(req: InferenceRequest):
    try:
        response = generate_with_energy(
            req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )
        return {"response": response}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
