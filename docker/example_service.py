"""Example service demonstrating Pulse integration"""

import random
import time
from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes

app = FastAPI()

# Setup OpenTelemetry
resource = Resource.create({
    ResourceAttributes.SERVICE_NAME: "example-api",
    ResourceAttributes.SERVICE_VERSION: "1.0.0",
})

provider = TracerProvider(resource=resource)
processor = BatchSpanProcessor(OTLPSpanExporter())
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer(__name__)


@app.get("/")
async def root():
    with tracer.start_as_current_span("root-handler") as span:
        span.set_attribute("handler", "root")
        return {"message": "Hello from example API"}


@app.get("/users/{user_id}")
async def get_user(user_id: int):
    with tracer.start_as_current_span("get-user") as span:
        span.set_attribute("user.id", user_id)
        
        # Simulate processing
        time.sleep(random.uniform(0.01, 0.1))
        
        # Maybe add an error
        if random.random() < 0.05:
            span.set_attribute("error", True)
            span.set_status(trace.Status(trace.StatusCode.ERROR, "User not found"))
            return {"error": "User not found"}, 404
        
        return {"user_id": user_id, "name": f"User {user_id}"}


@app.get("/orders/{order_id}")
async def get_order(order_id: int):
    with tracer.start_as_current_span("get-order") as span:
        span.set_attribute("order.id", order_id)
        
        # Call user service
        user_id = random.randint(1, 100)
        
        with tracer.start_as_current_span("fetch-user") as fetch_span:
            fetch_span.set_attribute("user.id", user_id)
            time.sleep(random.uniform(0.01, 0.05))
        
        # Simulate order processing
        time.sleep(random.uniform(0.02, 0.15))
        
        return {
            "order_id": order_id,
            "user_id": user_id,
            "total": random.uniform(10, 1000),
        }


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
